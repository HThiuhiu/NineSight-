// timer.hpp — nanosecond clock dựa trên TSC (rdtsc), calibrate với steady_clock.
//
// Tại sao TSC: đọc rdtsc tốn ~20 cycle, không syscall, độ phân giải < 1ns.
// Điều kiện: CPU có "invariant TSC" (mọi CPU x86 hiện đại đều có) để tần số
// không đổi theo P-state và đồng bộ giữa các core.
// Trên kiến trúc không phải x86 -> fallback steady_clock.
#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <x86intrin.h>
  #define TL_HAS_TSC 1
#else
  #define TL_HAS_TSC 0
#endif

namespace tl {

inline uint64_t steady_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class TscClock {
public:
    // Calibrate: đo số tick TSC trong một khoảng thời gian steady_clock đã biết.
    // TODO(step 3): lặp nhiều lần và lấy median để giảm nhiễu do context switch.
    void calibrate(std::chrono::milliseconds window = std::chrono::milliseconds(200)) {
#if TL_HAS_TSC
        uint64_t t0 = steady_ns(), c0 = __rdtsc();
        std::this_thread::sleep_for(window);
        uint64_t t1 = steady_ns(), c1 = __rdtsc();
        ns_per_tick_ = double(t1 - t0) / double(c1 - c0);
        base_tick_ = c1;
        base_ns_ = t1;
#endif
    }

    // Raw tick — dùng trong hot path (rẻ nhất có thể).
    inline uint64_t ticks() const {
#if TL_HAS_TSC
        return __rdtsc();
#else
        return steady_ns();
#endif
    }

    inline uint64_t ticks_to_ns(uint64_t t) const {
#if TL_HAS_TSC
        return base_ns_ + (uint64_t)((double)(int64_t)(t - base_tick_) * ns_per_tick_);
#else
        return t;
#endif
    }

    inline uint64_t now_ns() const { return ticks_to_ns(ticks()); }

    double ghz() const { return 1.0 / ns_per_tick_; }

private:
    double ns_per_tick_ = 1.0;
    uint64_t base_tick_ = 0;
    uint64_t base_ns_ = 0;
};

// Busy-wait tới thời điểm deadline_ns. Dùng cho scheduler open-loop:
// sleep_for() trên Windows có granularity ~1-15ms, không đủ cho 10k RPS (100µs/req).
inline void spin_until(const TscClock& clk, uint64_t deadline_ns) {
    while (clk.now_ns() < deadline_ns) {
#if TL_HAS_TSC
        _mm_pause();
#endif
    }
}

}  // namespace tl
