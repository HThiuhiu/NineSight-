// args.hpp — parse tham số dòng lệnh dạng "--key value" (tối giản).
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

namespace tl {

inline const char* arg_str(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}
inline long long arg_int(int argc, char** argv, const char* key, long long def) {
    const char* v = arg_str(argc, argv, key, nullptr);
    return v ? std::atoll(v) : def;
}
inline double arg_dbl(int argc, char** argv, const char* key, double def) {
    const char* v = arg_str(argc, argv, key, nullptr);
    return v ? std::atof(v) : def;
}

}  // namespace tl
