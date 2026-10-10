// open_loop.cpp — Task 3: Tail-latency Harness (open-loop + TSC + HdrHistogram).
//
// Kiến trúc (1 + 2N thread, N = --connections), nối bằng SPSC ring lock-free:
//
//                       ┌─ringA[0]─> [Sender 0] ─ringB[0]─> [Receiver 0]  (connection 0)
//   [Scheduler] ─round─┤
//                robin  └─ringA[N-1]─> [Sender N-1] ─ringB[N-1]─> [Receiver N-1]
//
//   Scheduler: intended_i = start + i * (1e9 / rate). Busy-spin tới intended_i rồi push
//              vào ringA của connection (i mod N). KHÔNG BAO GIỜ chạm socket => không bao
//              giờ bị mạng/server chặn. ringA đầy => ghi nhận "dropped" và đi tiếp theo lịch.
//              Đồng thời đo "scheduler lag" (now - intended) để CHỨNG MINH harness không
//              phải bottleneck.
//   Sender:    pop ringA, push (intended, actual_send) vào ringB, rồi send(). Nếu server
//              stall và TCP buffer đầy, send() block — chỉ Sender block, lịch vẫn chạy.
//   Receiver:  HTTP pipelining trả theo FIFO => response thứ k ứng với phần tử thứ k của ringB.
//              response_time = recv_time - INTENDED_send_time   (đo đúng, không CO)
//
// Kết thúc: sau khi lịch xong, chờ tối đa --drain-timeout-ms. Request chưa có response được
// tính là TIMEOUT và vẫn được ghi vào histogram với latency = end_time - intended (cận dưới),
// để phần đuôi không bị "giấu" đi.
//
// Usage: open_loop --rate 2000 --duration-s 10 --connections 1 --warmup-s 1
//                  --out results/open [--raw-out results/open_raw.csv --raw-max 1000000 --raw-every 1]
//
// --raw-every K: chỉ ghi raw 1/K request (histogram vẫn nhận TẤT CẢ) — giống sampled tracing
// trong production: giữ nguyên tải thật nhưng lưu ít mẫu, trải đều theo thời gian.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/args.hpp"
#include "../common/hdr_histogram.hpp"
#include "../common/net.hpp"
#include "../common/spsc_ring.hpp"
#include "../common/timer.hpp"

using namespace tl;

struct InFlight {
    uint64_t intended_ns;
    uint64_t actual_ns;
};

struct Sample {
    uint64_t intended_ns;
    int64_t latency_ns;
};

// Mọi trạng thái của một connection — cấp phát toàn bộ trước khi đo.
struct Conn {
    Conn(size_t ring_cap, size_t raw_cap) : ringA(ring_cap), ringB(ring_cap), raw(raw_cap) {}
    socket_t s = TL_INVALID_SOCKET;
    SpscRing<uint64_t> ringA;
    SpscRing<InFlight> ringB;
    HdrHistogram resp, send_based;
    std::vector<Sample> raw;
    size_t raw_n = 0;
    uint64_t raw_seen = 0;
    std::atomic<uint64_t> scheduled{0}, received{0};
    bool drain_timed_out = false;
};

int main(int argc, char** argv) {
    const char* host = arg_str(argc, argv, "--host", "127.0.0.1");
    int port = (int)arg_int(argc, argv, "--port", 8080);
    double rate = arg_dbl(argc, argv, "--rate", 2000.0);
    double duration_s = arg_dbl(argc, argv, "--duration-s", 10.0);
    double warmup_s = arg_dbl(argc, argv, "--warmup-s", 0.0);
    int nconn = (int)std::max(1LL, arg_int(argc, argv, "--connections", 1));
    long long drain_timeout_ms = arg_int(argc, argv, "--drain-timeout-ms", 5000);
    int ring_pow2 = (int)arg_int(argc, argv, "--queue-pow2", 18);  // 2^18 = 262k request backlog / conn
    std::string out = arg_str(argc, argv, "--out", "results/open");
    const char* raw_out = arg_str(argc, argv, "--raw-out", nullptr);
    long long raw_max = arg_int(argc, argv, "--raw-max", 1000000);
    const uint64_t raw_every = (uint64_t)std::max(1LL, arg_int(argc, argv, "--raw-every", 1));
    // TODO: pin thread vào core (SetThreadAffinityMask / pthread_setaffinity_np).

    const uint64_t interval_ns = (uint64_t)(1e9 / rate);
    const uint64_t total = (uint64_t)(rate * (duration_s + warmup_s));

    NetInit net;
    TscClock clk;
    clk.calibrate();

    // ---------- Pre-allocate MỌI THỨ trước khi bắt đầu đo ----------
    const size_t raw_per_conn = raw_out ? (size_t)(raw_max / nconn + 1) : 0;
    std::vector<std::unique_ptr<Conn>> conns;
    for (int i = 0; i < nconn; ++i) {
        conns.emplace_back(new Conn((size_t)1 << ring_pow2, raw_per_conn));
        conns.back()->s = tcp_connect(host, (uint16_t)port);
        if (conns.back()->s == TL_INVALID_SOCKET) { std::fprintf(stderr, "connect #%d failed\n", i); return 1; }
    }
    HdrHistogram sched_lag;
    std::atomic<bool> sched_done{false}, stop{false};
    std::atomic<uint64_t> dropped{0};
    const int req_len = (int)sizeof(kPingRequest) - 1;

    std::printf("[open_loop] rate=%.0f rps, conns=%d, duration=%.1fs (+%.1fs warmup), total=%llu, "
                "interval=%lluns, tsc=%.3fGHz\n",
                rate, nconn, duration_s, warmup_s, (unsigned long long)total,
                (unsigned long long)interval_ns, clk.ghz());

    const uint64_t start_ns = clk.now_ns() + 50000000ULL;  // bắt đầu sau 50ms để các thread sẵn sàng
    const uint64_t record_from_ns = start_ns + (uint64_t)(warmup_s * 1e9);

    // ---------- Scheduler ----------
    std::thread scheduler([&] {
#ifdef _WIN32
        // Giảm khả năng bị OS preempt giữa lúc spin (spike lag hàng chục ms trên run dài).
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
        // TODO(Linux): SCHED_FIFO + pin vào core cô lập (isolcpus) — cần quyền root.
        for (uint64_t i = 0; i < total; ++i) {
            const uint64_t intended = start_ns + i * interval_ns;
            spin_until(clk, intended);
            Conn& c = *conns[i % (uint64_t)nconn];
            if (c.ringA.push(intended)) c.scheduled.fetch_add(1, std::memory_order_release);
            else dropped.fetch_add(1, std::memory_order_relaxed);  // không dừng lịch
            sched_lag.record((int64_t)(clk.now_ns() - intended));
        }
        sched_done.store(true, std::memory_order_release);
    });

    // ---------- Sender / Receiver cho từng connection ----------
    auto sender_fn = [&](Conn& c) {
        uint64_t intended;
        unsigned idle = 0;
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) return;
            if (c.ringA.pop(intended)) {
                idle = 0;
                InFlight f{intended, clk.now_ns()};
                // push trước send() để receiver luôn tìm thấy phần tử tương ứng
                while (!c.ringB.push(f))
                    if (stop.load(std::memory_order_relaxed)) return;
                if (!send_all(c.s, kPingRequest, req_len)) return;  // socket đóng / server chết
            } else if (sched_done.load(std::memory_order_acquire) && c.ringA.size_approx() == 0) {
                return;
            } else if (++idle < 2000) {
                _mm_pause();  // spin ngắn: request kế tiếp thường tới trong vài µs
            } else {
                std::this_thread::yield();  // rảnh lâu: nhường CPU cho scheduler/receiver/server
            }
        }
    };

    auto receiver_fn = [&](Conn& c) {
        HttpResponseParser parser;
        char buf[64 * 1024];
        uint64_t last_progress = clk.now_ns();
        for (;;) {
            const bool done_sched = sched_done.load(std::memory_order_acquire);
            if (done_sched && c.received.load(std::memory_order_relaxed) >=
                                  c.scheduled.load(std::memory_order_acquire))
                break;
            if (done_sched && clk.now_ns() - last_progress > (uint64_t)drain_timeout_ms * 1000000ULL) {
                c.drain_timed_out = true;
                break;
            }
            fd_set rf;
            FD_ZERO(&rf);
            FD_SET(c.s, &rf);
            timeval tv{0, 100000};  // poll 100ms để còn kiểm tra điều kiện dừng
            if (select((int)c.s + 1, &rf, nullptr, nullptr, &tv) <= 0) {
                if (!done_sched) last_progress = clk.now_ns();  // drain timeout chỉ tính sau khi lịch xong
                continue;
            }
            int n = recv(c.s, buf, (int)sizeof(buf), 0);
            if (n <= 0) { std::fprintf(stderr, "recv failed/closed\n"); break; }
            const uint64_t now = clk.now_ns();  // mọi response trong cùng recv() được quan sát cùng lúc
            int done = parser.feed(buf, n);
            for (int k = 0; k < done; ++k) {
                InFlight f;
                if (!c.ringB.pop(f)) break;  // không nên xảy ra
                if (f.intended_ns < record_from_ns) continue;  // warmup
                const int64_t lat = (int64_t)(now - f.intended_ns);
                c.resp.record(lat);
                c.send_based.record((int64_t)(now - f.actual_ns));
                if (c.raw_seen++ % raw_every == 0 && c.raw_n < c.raw.size()) c.raw[c.raw_n++] = {f.intended_ns, lat};
            }
            if (done) {
                c.received.fetch_add((uint64_t)done, std::memory_order_relaxed);
                last_progress = now;
            }
        }
    };

    std::vector<std::thread> senders, receivers;
    for (auto& c : conns) {
        senders.emplace_back(sender_fn, std::ref(*c));
        receivers.emplace_back(receiver_fn, std::ref(*c));
    }

    scheduler.join();
    for (auto& t : receivers) t.join();
    // Receiver xong => đóng socket để giải phóng sender còn kẹt trong send().
    stop.store(true);
    for (auto& c : conns) { tl_shutdown(c->s); tl_close(c->s); }
    for (auto& t : senders) t.join();

    // ---------- Hậu xử lý (đã hết đo, được phép allocate) ----------
    const uint64_t end_ns = clk.now_ns();
    HdrHistogram resp, send_based;
    std::vector<Sample> raw;
    uint64_t scheduled = 0, received = 0, timeouts = 0;
    for (auto& c : conns) {
        // Request đã gửi/chưa gửi mà không có response => timeout, ghi latency cận dưới.
        InFlight f;
        uint64_t intended;
        auto add_timeout = [&](uint64_t intended_ns) {
            ++timeouts;
            if (intended_ns < record_from_ns) return;
            const int64_t lat = (int64_t)(end_ns - intended_ns);
            c->resp.record(lat);
            raw.push_back({intended_ns, lat});
        };
        while (c->ringB.pop(f)) add_timeout(f.intended_ns);
        while (c->ringA.pop(intended)) add_timeout(intended);
        resp.merge(c->resp);
        send_based.merge(c->send_based);
        raw.insert(raw.end(), c->raw.begin(), c->raw.begin() + (long)c->raw_n);
        scheduled += c->scheduled.load();
        received += c->received.load();
    }

    std::printf("scheduled=%llu received=%llu dropped=%llu timeouts=%llu\n", (unsigned long long)scheduled,
                (unsigned long long)received, (unsigned long long)dropped.load(), (unsigned long long)timeouts);
    resp.print_summary("open-loop RESPONSE time (from intended start)");
    send_based.print_summary("time from ACTUAL send (differs only if the client itself lags)");
    sched_lag.print_summary("harness self-check: scheduler lag (now - intended)", 1e3, "us");
    // Lag của scheduler ĐÃ nằm trong latency đo được (vì tính từ intended time) => chỉ làm kết quả
    // bảo thủ hơn. Nó chỉ đáng lo khi không còn nhỏ so với chính phần đuôi ta đang đo.
    const int64_t lag999 = sched_lag.value_at_percentile(99.9);
    const int64_t resp999 = resp.value_at_percentile(99.9);
    std::printf("harness overhead: scheduler lag p99.9 = %.1fus = %.3f%% of response p99.9\n", lag999 / 1e3,
                resp999 ? 100.0 * (double)lag999 / (double)resp999 : 0.0);
    if (resp999 && lag999 * 20 > resp999)
        std::printf("WARNING: scheduler lag > 5%% of measured p99.9 -> harness may distort the tail\n");

    if (FILE* f = std::fopen((out + ".hgrm").c_str(), "w")) { resp.write_hgrm(f); std::fclose(f); }
    if (FILE* f = std::fopen((out + "_buckets.csv").c_str(), "w")) { resp.write_buckets_csv(f); std::fclose(f); }
    if (raw_out) {
        // Sắp theo intended time: EVT cần thứ tự thời gian để phát hiện cụm (declustering).
        std::sort(raw.begin(), raw.end(), [](const Sample& a, const Sample& b) { return a.intended_ns < b.intended_ns; });
        if (FILE* f = std::fopen(raw_out, "w")) {
            std::fprintf(f, "intended_offset_ns,latency_ns\n");
            for (const Sample& s : raw)
                std::fprintf(f, "%llu,%lld\n", (unsigned long long)(s.intended_ns - record_from_ns), (long long)s.latency_ns);
            std::fclose(f);
        }
    }
    std::printf("wrote %s.hgrm, %s_buckets.csv%s%s\n", out.c_str(), out.c_str(), raw_out ? ", " : "",
                raw_out ? raw_out : "");
    return timeouts > 0 ? 2 : 0;
}
