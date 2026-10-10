// spsc_ring.hpp — Lock-free Single-Producer/Single-Consumer ring buffer.
//
// Dùng để nối các thread của open-loop harness (scheduler -> sender -> receiver)
// mà không cần mutex. Bộ nhớ cấp phát 1 lần ở constructor; push/pop không allocate.
// push() KHÔNG BAO GIỜ block: nếu đầy thì trả false để caller tự quyết định
// (scheduler sẽ ghi nhận "dropped" thay vì dừng lịch).
#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace tl {

template <typename T>
class SpscRing {
public:
    explicit SpscRing(size_t capacity_pow2) : mask_(capacity_pow2 - 1), buf_(capacity_pow2) {}

    bool push(const T& v) {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_cache_ > mask_) {  // có vẻ đầy -> đọc lại tail thật
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (h - tail_cache_ > mask_) return false;
        }
        buf_[h & mask_] = v;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_cache_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (t == head_cache_) return false;
        }
        out = buf_[t & mask_];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    size_t size_approx() const {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

private:
    const size_t mask_;
    std::vector<T> buf_;
    // Tách head/tail ra 2 cache line khác nhau để tránh false sharing.
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) size_t tail_cache_ = 0;  // chỉ producer dùng
    alignas(64) std::atomic<size_t> tail_{0};
    alignas(64) size_t head_cache_ = 0;  // chỉ consumer dùng
};

}  // namespace tl
