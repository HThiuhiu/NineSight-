// dummy_server.cpp — Task 1: Dummy Target System.
//
// HTTP/1.1 server tối giản, chỉ phục vụ GET /ping (keep-alive + pipelining).
// Mô phỏng 2 kiểu "stall":
//
//   --stall-mode request  (mặc định, đúng spec): mỗi request có xác suất
//        --stall-prob (0.01) bị delay U[--stall-min-ms, --stall-max-ms] = U[100,500]ms.
//        Request còn lại "bận" --service-us micro-giây (busy-spin, < 1ms).
//
//   --stall-mode none   : không stall — dùng để đo "sàn" độ trễ của chính harness + loopback.
//
//   --stall-mode global : mô phỏng GC stop-the-world. Một thread nền cứ mỗi
//        ~--gc-interval-ms lại "đóng băng" TOÀN BỘ server trong U[100,500]ms.
//        Đây là kịch bản kinh điển mà Gil Tene dùng để minh họa Coordinated Omission.
//
// LƯU Ý capacity (mode request): thời gian phục vụ trung bình
//   E[S] ≈ 0.99 * 0.1ms + 0.01 * 300ms ≈ 3.1ms  =>  ~320 RPS / connection.
// Open-loop với rate > capacity sẽ làm hàng đợi tăng vô hạn (đó cũng là sự thật
// mà closed-loop không bao giờ cho bạn thấy).
//
// Usage: dummy_server --port 8080 --stall-mode request|global|none

#include <atomic>
#include <cstdio>
#include <random>
#include <thread>

#include "../common/args.hpp"
#include "../common/net.hpp"
#include "../common/timer.hpp"

using namespace tl;

struct Config {
    int port;
    bool global_mode;
    double stall_prob;
    int stall_min_ms, stall_max_ms;
    int service_us;
    int gc_interval_ms;
};

static Config g_cfg;
static TscClock g_clk;
static std::atomic<uint64_t> g_stall_until_ns{0};  // dùng cho global mode
static std::atomic<uint64_t> g_requests{0}, g_stalls{0};

static const char kResponse[] =
    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n\r\npong";

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// Xử lý 1 request: áp dụng độ trễ giả lập rồi mới trả lời.
static void handle_one(std::mt19937_64& rng) {
    g_requests.fetch_add(1, std::memory_order_relaxed);

    if (g_cfg.global_mode) {
        // Stop-the-world: nếu đang trong "GC pause" thì đứng chờ tới khi hết.
        uint64_t until = g_stall_until_ns.load(std::memory_order_acquire);
        while (g_clk.now_ns() < until) {
            sleep_ms(1);
            until = g_stall_until_ns.load(std::memory_order_acquire);
        }
    } else {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        if (u01(rng) < g_cfg.stall_prob) {
            std::uniform_int_distribution<int> d(g_cfg.stall_min_ms, g_cfg.stall_max_ms);
            g_stalls.fetch_add(1, std::memory_order_relaxed);
            sleep_ms(d(rng));
            return;
        }
    }
    // Fast path: busy-spin service time (sleep trên Windows quá thô cho < 1ms).
    spin_until(g_clk, g_clk.now_ns() + (uint64_t)g_cfg.service_us * 1000);
}

static void serve_connection(socket_t c, uint64_t seed) {
    set_nodelay(c);
    std::mt19937_64 rng(seed);
    char buf[16 * 1024];
    int have = 0;
    for (;;) {
        int n = recv(c, buf + have, (int)sizeof(buf) - have, 0);
        if (n <= 0) break;
        have += n;
        // Tách các request pipelined: mỗi request kết thúc bằng "\r\n\r\n".
        int start = 0;
        for (int i = 3; i < have; ++i) {
            if (buf[i - 3] == '\r' && buf[i - 2] == '\n' && buf[i - 1] == '\r' && buf[i] == '\n') {
                // TODO: route theo path; hiện mọi request đều coi là /ping.
                handle_one(rng);
                if (!send_all(c, kResponse, (int)sizeof(kResponse) - 1)) { tl_close(c); return; }
                start = i + 1;
            }
        }
        // Dời phần request chưa đầy đủ về đầu buffer.
        if (start > 0) {
            std::memmove(buf, buf + start, (size_t)(have - start));
            have -= start;
        }
        if (have == (int)sizeof(buf)) break;  // request quá lớn -> đóng kết nối
    }
    tl_close(c);
}

// Thread nền mô phỏng GC: ngủ ~interval rồi đặt cửa sổ stall cho toàn server.
static void gc_thread() {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> pause(g_cfg.stall_min_ms, g_cfg.stall_max_ms);
    std::uniform_int_distribution<int> jitter(-g_cfg.gc_interval_ms / 4, g_cfg.gc_interval_ms / 4);
    for (;;) {
        sleep_ms(g_cfg.gc_interval_ms + jitter(rng));
        int p = pause(rng);
        g_stall_until_ns.store(g_clk.now_ns() + (uint64_t)p * 1000000ULL, std::memory_order_release);
        g_stalls.fetch_add(1, std::memory_order_relaxed);
        sleep_ms(p);
    }
}

int main(int argc, char** argv) {
    g_cfg.port = (int)arg_int(argc, argv, "--port", 8080);
    const std::string mode = arg_str(argc, argv, "--stall-mode", "request");
    g_cfg.global_mode = mode == "global";
    g_cfg.stall_prob = mode == "none" ? 0.0 : arg_dbl(argc, argv, "--stall-prob", 0.01);
    g_cfg.stall_min_ms = (int)arg_int(argc, argv, "--stall-min-ms", 100);
    g_cfg.stall_max_ms = (int)arg_int(argc, argv, "--stall-max-ms", 500);
    g_cfg.service_us = (int)arg_int(argc, argv, "--service-us", 100);
    g_cfg.gc_interval_ms = (int)arg_int(argc, argv, "--gc-interval-ms", 2000);

    NetInit net;
    g_clk.calibrate();

    const char* bind_host = arg_str(argc, argv, "--bind", "127.0.0.1");
    socket_t ls = tcp_listen(bind_host, (uint16_t)g_cfg.port);
    if (ls == TL_INVALID_SOCKET) { std::fprintf(stderr, "listen failed on port %d\n", g_cfg.port); return 1; }

    std::printf("[dummy_server] port=%d mode=%s stall_prob=%.4f stall=[%d,%d]ms service=%dus tsc=%.3fGHz\n",
                g_cfg.port, mode.c_str(), g_cfg.stall_prob,
                g_cfg.stall_min_ms, g_cfg.stall_max_ms, g_cfg.service_us, g_clk.ghz());
    std::fflush(stdout);

    if (g_cfg.global_mode) std::thread(gc_thread).detach();

    uint64_t seed = 1;
    for (;;) {
        socket_t c = accept(ls, nullptr, nullptr);
        if (c == TL_INVALID_SOCKET) continue;
        std::thread(serve_connection, c, seed++).detach();  // thread-per-connection
    }
}
