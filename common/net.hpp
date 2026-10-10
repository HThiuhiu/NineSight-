// net.hpp — lớp bọc socket tối giản cho Windows (Winsock2) và POSIX.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using socket_t = SOCKET;
  #define TL_INVALID_SOCKET INVALID_SOCKET
  #define tl_close closesocket
  #define TL_SHUT_RDWR SD_BOTH
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using socket_t = int;
  #define TL_INVALID_SOCKET (-1)
  #define tl_close close
  #define TL_SHUT_RDWR SHUT_RDWR
#endif

namespace tl {

struct NetInit {
    NetInit() {
#ifdef _WIN32
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
#endif
    }
    ~NetInit() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

// shutdown() trước close() để đánh thức thread đang block trong send()/recv().
inline void tl_shutdown(socket_t s) { shutdown(s, TL_SHUT_RDWR); }

inline void set_nodelay(socket_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
}

inline socket_t tcp_listen(const char* host, uint16_t port) {
    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == TL_INVALID_SOCKET) return s;
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host, &a.sin_addr);
    if (bind(s, (sockaddr*)&a, sizeof(a)) != 0 || listen(s, 128) != 0) {
        tl_close(s);
        return TL_INVALID_SOCKET;
    }
    return s;
}

inline socket_t tcp_connect(const char* host, uint16_t port) {
    socket_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == TL_INVALID_SOCKET) return s;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host, &a.sin_addr);
    if (connect(s, (sockaddr*)&a, sizeof(a)) != 0) {
        tl_close(s);
        return TL_INVALID_SOCKET;
    }
    set_nodelay(s);
    return s;
}

inline bool send_all(socket_t s, const char* buf, int len) {
    while (len > 0) {
        int n = send(s, buf, len, 0);
        if (n <= 0) return false;
        buf += n;
        len -= n;
    }
    return true;
}

// Bộ đếm HTTP response cho kết nối keep-alive/pipelined.
// Đọc stream byte, đếm số response hoàn chỉnh (header + Content-Length body).
// Không allocate: buffer cố định do caller cấp.
class HttpResponseParser {
public:
    // Nạp thêm n byte, trả về số response hoàn chỉnh vừa parse được.
    int feed(const char* data, int n) {
        int done = 0;
        for (int i = 0; i < n; ++i) {
            char c = data[i];
            if (body_left_ > 0) {
                if (--body_left_ == 0) { ++done; reset(); }
                continue;
            }
            // Bắt dòng header "Content-Length: N" (đơn giản hóa cho dummy server).
            if (line_len_ < (int)sizeof(line_) - 1) line_[line_len_++] = c;
            if (c == '\n') {
                line_[line_len_] = 0;
                if (line_len_ <= 2) {  // dòng trống => hết header
                    if (content_length_ == 0) { ++done; reset(); }
                    else body_left_ = content_length_;
                } else if (std::strncmp(line_, "Content-Length:", 15) == 0) {
                    content_length_ = std::atoi(line_ + 15);
                }
                line_len_ = 0;
            }
        }
        return done;
    }

private:
    void reset() { content_length_ = 0; body_left_ = 0; line_len_ = 0; }
    char line_[256];
    int line_len_ = 0;
    int content_length_ = 0;
    int body_left_ = 0;
};

static const char kPingRequest[] = "GET /ping HTTP/1.1\r\nHost: localhost\r\n\r\n";

}  // namespace tl
