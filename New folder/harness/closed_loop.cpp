// closed_loop.cpp — Task 2: Naive Benchmark (closed-loop, đo SAI có chủ đích).
//
//   loop: t0 = now(); send(req); wait(resp); record(now() - t0);
//
// Vấn đề (Coordinated Omission): khi server đứng 500ms, client cũng đứng theo
// => trong 500ms đó lẽ ra phải có hàng nghìn request (ở tải thực tế) bị chậm,
// nhưng closed-loop chỉ ghi nhận ĐÚNG 1 mẫu chậm. Đuôi phân phối bị "xóa sổ".
//
// Usage: closed_loop --host 127.0.0.1 --port 8080 --requests 20000 --out results/closed

#include <cstdio>
#include <string>

#include "../common/args.hpp"
#include "../common/hdr_histogram.hpp"
#include "../common/net.hpp"
#include "../common/timer.hpp"

using namespace tl;

int main(int argc, char** argv) {
    const char* host = arg_str(argc, argv, "--host", "127.0.0.1");
    int port = (int)arg_int(argc, argv, "--port", 8080);
    long long requests = arg_int(argc, argv, "--requests", 20000);
    std::string out = arg_str(argc, argv, "--out", "results/closed");
    // Nếu > 0: tạo thêm histogram "đã bù CO" kiểu HdrHistogram (chỉ để so sánh).
    long long expected_interval_us = arg_int(argc, argv, "--expected-interval-us", 0);

    NetInit net;
    TscClock clk;
    clk.calibrate();

    socket_t s = tcp_connect(host, (uint16_t)port);
    if (s == TL_INVALID_SOCKET) { std::fprintf(stderr, "connect failed\n"); return 1; }

    // Pre-allocate trước khi đo.
    HdrHistogram hist, corrected;
    HttpResponseParser parser;
    char buf[4096];
    const int req_len = (int)sizeof(kPingRequest) - 1;

    std::printf("[closed_loop] %lld requests -> %s:%d\n", requests, host, port);
    uint64_t t_begin = clk.now_ns();
    for (long long i = 0; i < requests; ++i) {
        uint64_t t0 = clk.ticks();
        if (!send_all(s, kPingRequest, req_len)) { std::fprintf(stderr, "send failed\n"); return 1; }
        int got = 0;
        while (got == 0) {
            int n = recv(s, buf, (int)sizeof(buf), 0);
            if (n <= 0) { std::fprintf(stderr, "recv failed\n"); return 1; }
            got = parser.feed(buf, n);
        }
        uint64_t t1 = clk.ticks();
        int64_t lat = (int64_t)(clk.ticks_to_ns(t1) - clk.ticks_to_ns(t0));
        hist.record(lat);
        if (expected_interval_us > 0) corrected.record_corrected(lat, expected_interval_us * 1000);
    }
    double elapsed_s = (double)(clk.now_ns() - t_begin) / 1e9;
    tl_close(s);

    std::printf("achieved throughput: %.1f req/s (client is paced by the server = coordinated omission)\n",
                (double)requests / elapsed_s);
    // Spec Task 2: in p50/p90/p99 — p99 trông "đẹp". In thêm phần đuôi để thấy sự thật.
    std::printf("p50 = %.3f ms | p90 = %.3f ms | p99 = %.3f ms   <-- what a typical dashboard shows\n",
                hist.value_at_percentile(50) / 1e6, hist.value_at_percentile(90) / 1e6,
                hist.value_at_percentile(99) / 1e6);
    hist.print_summary("closed-loop (raw)");
    if (expected_interval_us > 0) corrected.print_summary("closed-loop (HdrHistogram CO-corrected)");

    if (FILE* f = std::fopen((out + ".hgrm").c_str(), "w")) { hist.write_hgrm(f); std::fclose(f); }
    if (FILE* f = std::fopen((out + "_buckets.csv").c_str(), "w")) { hist.write_buckets_csv(f); std::fclose(f); }
    std::printf("wrote %s.hgrm, %s_buckets.csv\n", out.c_str(), out.c_str());
    return 0;
}
