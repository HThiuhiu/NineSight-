// test_hdr.cpp — kiểm tra HdrHistogram so với percentile chính xác (sort toàn bộ mẫu).
// Yêu cầu: sai số tương đối <= 0.1% (3 significant digits) trên toàn dải 1µs .. 10s.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../common/hdr_histogram.hpp"
#include "../common/spsc_ring.hpp"

using tl::HdrHistogram;

static int failures = 0;
#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++failures; } \
    } while (0)

static void test_percentiles_lognormal() {
    std::mt19937_64 rng(7);
    std::lognormal_distribution<double> d(std::log(200000.0), 2.0);  // median 200µs, đuôi rất dài
    std::vector<int64_t> v;
    HdrHistogram h;
    for (int i = 0; i < 1000000; ++i) {
        int64_t x = std::min<int64_t>((int64_t)d(rng) + 1, 10LL * 1000000000);
        v.push_back(x);
        h.record(x);
    }
    std::sort(v.begin(), v.end());
    for (double p : {50.0, 90.0, 99.0, 99.9, 99.99, 99.999}) {
        size_t rank = (size_t)std::ceil(p / 100.0 * v.size()) - 1;
        double exact = (double)v[rank];
        double got = (double)h.value_at_percentile(p);
        double rel = std::fabs(got - exact) / exact;
        CHECK(rel <= 1e-3, "p%g exact=%.0f hdr=%.0f rel=%.5f", p, exact, got, rel);
    }
    CHECK(h.total() == (int64_t)v.size(), "total count mismatch");
    CHECK(h.max() == v.back(), "max mismatch");
}

static void test_corrected_recording() {
    // 1 mẫu 100ms với interval kỳ vọng 1ms => thêm 99 mẫu 99ms, 98ms, ..., 1ms.
    HdrHistogram h;
    h.record_corrected(100000000, 1000000);
    CHECK(h.total() == 100, "corrected total = %lld, want 100", (long long)h.total());
    double p50 = (double)h.value_at_percentile(50);
    CHECK(std::fabs(p50 - 50e6) / 50e6 < 2e-3, "corrected p50 = %.0f, want ~50ms", p50);
}

static void test_merge() {
    HdrHistogram a, b;
    for (int i = 1; i <= 1000; ++i) (i % 2 ? a : b).record(i * 1000);
    a.merge(b);
    CHECK(a.total() == 1000, "merged total");
    CHECK(std::fabs((double)a.value_at_percentile(50) - 500000) / 500000 < 1e-3, "merged p50");
}

static void test_spsc_ring() {
    tl::SpscRing<int> r(4);
    int x;
    CHECK(!r.pop(x), "empty pop");
    for (int i = 0; i < 4; ++i) CHECK(r.push(i), "push %d", i);
    CHECK(!r.push(99), "push on full ring must fail (non-blocking)");
    for (int i = 0; i < 4; ++i) CHECK(r.pop(x) && x == i, "FIFO order at %d", i);
}

int main() {
    test_percentiles_lognormal();
    test_corrected_recording();
    test_merge();
    test_spsc_ring();
    std::printf(failures ? "%d test(s) FAILED\n" : "all C++ tests passed\n", failures);
    return failures ? 1 : 0;
}
