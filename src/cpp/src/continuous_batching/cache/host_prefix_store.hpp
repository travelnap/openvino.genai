// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"

namespace ov::genai {

inline ov::Tensor copy_cache_block_to_host(const ov::Tensor& cache, size_t block_index) {
    const ov::Shape shape = cache.get_shape();
    OPENVINO_ASSERT(!shape.empty() && block_index < shape[0], "Cache block index is outside the allocated pool");
    ov::Shape block_shape = shape;
    block_shape[0] = 1;
    ov::Coordinate start(shape.size(), 0);
    ov::Coordinate end(shape.begin(), shape.end());
    start[0] = block_index;
    end[0] = block_index + 1;
    ov::Tensor roi(cache, start, end);
    ov::Tensor host(cache.get_element_type(), block_shape);
    roi.copy_to(host);
    return host;
}

inline void copy_host_block_to_cache(const ov::Tensor& host, ov::Tensor& cache, size_t block_index) {
    const ov::Shape shape = cache.get_shape();
    OPENVINO_ASSERT(!shape.empty() && block_index < shape[0], "Cache block index is outside the allocated pool");
    ov::Coordinate start(shape.size(), 0);
    ov::Coordinate end(shape.begin(), shape.end());
    start[0] = block_index;
    end[0] = block_index + 1;
    ov::Tensor roi(cache, start, end);
    host.copy_to(roi);
}

inline bool cache_block_can_spill(const ov::Tensor& cache) {
    if (cache.get_size() == 0) {
        return false;
    }
    const auto precision = cache.get_element_type();
    const bool sub_byte = precision == ov::element::u4 || precision == ov::element::i4;
    return !(sub_byte && cache.is<ov::RemoteTensor>());
}

// Bytes of prefix blocks that no longer fit in the device pool. One budget is shared by every
// cache type so a hybrid model cannot keep KV blocks for a chat whose linear-attention state was dropped.
struct HostPrefixBudget {
    std::mutex mutex;
    size_t limit = 0;
    size_t used = 0;
    std::vector<std::pair<size_t, std::function<bool()>>> droppers;
};

class HostPrefixStore {
public:
    HostPrefixStore(ICacheManager* cache, std::shared_ptr<HostPrefixBudget> budget, std::function<void(uint64_t)> on_drop)
        : m_cache(cache), m_budget(std::move(budget)), m_on_drop(std::move(on_drop)) {
        m_id = m_budget->droppers.size() + 1;
        m_budget->droppers.emplace_back(m_id, [this]() {
            return drop_oldest();
        });
    }

    ~HostPrefixStore() {
        m_on_drop = nullptr;
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        for (const auto& entry : m_entries) {
            if (m_budget->used >= entry.second.bytes) {
                m_budget->used -= entry.second.bytes;
            } else {
                m_budget->used = 0;
            }
        }
        m_entries.clear();
        m_order.clear();
        auto& droppers = m_budget->droppers;
        droppers.erase(std::remove_if(droppers.begin(), droppers.end(), [this](const auto& dropper) {
            return dropper.first == m_id;
        }), droppers.end());
    }

    HostPrefixStore(const HostPrefixStore&) = delete;
    HostPrefixStore& operator=(const HostPrefixStore&) = delete;

    void save(uint64_t hash, int block_index) {
        if (block_index < 0 || m_budget->limit == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        if (m_entries.count(hash) > 0) {
            return;
        }
        std::vector<ov::Tensor> tensors;
        m_cache->export_block(static_cast<size_t>(block_index), tensors);
        if (tensors.empty()) {
            return;
        }
        size_t bytes = 0;
        for (const auto& tensor : tensors) {
            bytes += tensor.get_byte_size();
        }
        if (bytes == 0 || bytes > m_budget->limit) {
            return;
        }
        while (m_budget->used + bytes > m_budget->limit) {
            if (!drop_any_oldest()) {
                note_once(m_announced_skip, "[host-prefix] a prefix block does not fit in the remaining DRAM cap\n");
                return;
            }
        }
        m_budget->used += bytes;
        m_order.push_back(hash);
        m_entries.emplace(hash, Entry{std::move(tensors), bytes});
        note_once(m_announced_save, "[host-prefix] spilled a GPU prefix block to DRAM\n");
    }

    bool contains(uint64_t hash) const {
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        return m_entries.count(hash) > 0;
    }

    // False means the snapshot is gone. The caller must not keep the device block under this hash.
    bool load(uint64_t hash, int block_index) {
        if (block_index < 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        const auto it = m_entries.find(hash);
        if (it == m_entries.end()) {
            return false;
        }
        m_cache->import_block(static_cast<size_t>(block_index), it->second.tensors);
        note_once(m_announced_load, "[host-prefix] restored a prefix block from DRAM\n");
        return true;
    }

    void pin(uint64_t hash) {
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        m_pinned.insert(hash);
    }

    void unpin() {
        std::lock_guard<std::mutex> lock(m_budget->mutex);
        m_pinned.clear();
    }

private:
    struct Entry {
        std::vector<ov::Tensor> tensors;
        size_t bytes = 0;
    };

    bool drop_oldest() {
        auto order = m_order.begin();
        while (order != m_order.end() && m_pinned.count(*order) > 0) {
            ++order;
        }
        if (order == m_order.end()) {
            return false;
        }
        const uint64_t hash = *order;
        m_order.erase(order);
        const auto it = m_entries.find(hash);
        if (it == m_entries.end()) {
            return false;
        }
        OPENVINO_ASSERT(m_budget->used >= it->second.bytes, "Host prefix byte accounting underflow");
        m_budget->used -= it->second.bytes;
        m_entries.erase(it);
        if (m_on_drop) {
            m_on_drop(hash);
        }
        note_once(m_announced_drop, "[host-prefix] DRAM prefix cap is full; dropping the oldest spilled block\n");
        return true;
    }

    bool drop_any_oldest() {
        for (auto& dropper : m_budget->droppers) {
            if (dropper.second()) {
                return true;
            }
        }
        return false;
    }

    static void note_once(std::atomic<bool>& flag, const char* message) {
        bool expected = false;
        if (flag.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "%s", message);
            std::fflush(stderr);
        }
    }

    ICacheManager* m_cache = nullptr;
    std::shared_ptr<HostPrefixBudget> m_budget;
    std::function<void(uint64_t)> m_on_drop;
    std::map<uint64_t, Entry> m_entries;
    std::list<uint64_t> m_order;
    std::unordered_set<uint64_t> m_pinned;
    size_t m_id = 0;
    std::atomic<bool> m_announced_save{false};
    std::atomic<bool> m_announced_load{false};
    std::atomic<bool> m_announced_drop{false};
    std::atomic<bool> m_announced_skip{false};
};

inline size_t host_prefix_budget_bytes() {
    const char* env = std::getenv("OV_GENAI_HOST_PREFIX_BYTES");
    if (env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end != env) {
            return static_cast<size_t>(parsed);
        }
    }
    return 16ULL << 30;
}

}  // namespace ov::genai
