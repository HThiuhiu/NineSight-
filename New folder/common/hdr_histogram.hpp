// hdr_histogram.hpp — HdrHistogram tối giản (header-only), cùng thuật toán bucket
// với bản gốc của Gil Tene (https://github.com/HdrHistogram/HdrHistogram_c).
//
// Ý tưởng: lưu count theo dải log-linear. Mỗi "bucket" ứng với một lũy thừa 2,
// bên trong chia thành sub_bucket_count ô tuyến tính -> sai số tương đối cố định
// (3 significant digits => sai số <= 0.1%) trên toàn dải [1ns, 1 giờ].
// Bộ nhớ cố định (~ vài trăm KB) bất kể ghi bao nhiêu triệu mẫu => không OOM.
//
// Toàn bộ bộ nhớ được cấp phát MỘT LẦN trong constructor. record() là O(1),
// không allocate, không lock (mỗi thread giữ histogram riêng, merge sau khi đo).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace tl {

class HdrHistogram {
public:
    HdrHistogram(int64_t lowest = 1, int64_t highest = 3600LL * 1000000000LL, int sig_figs = 3)
        : lowest_(lowest), highest_(highest), sig_figs_(sig_figs) {
        int64_t largest_single_unit = 2 * (int64_t)std::pow(10, sig_figs);
        int sub_bucket_count_magnitude = (int)std::ceil(std::log2((double)largest_single_unit));
        sub_bucket_half_count_magnitude_ = (sub_bucket_count_magnitude > 1 ? sub_bucket_count_magnitude : 1) - 1;
        unit_magnitude_ = (int)std::floor(std::log2((double)lowest));
        sub_bucket_count_ = (int64_t)1 << (sub_bucket_half_count_magnitude_ + 1);
        sub_bucket_half_count_ = sub_bucket_count_ / 2;
        sub_bucket_mask_ = (sub_bucket_count_ - 1) << unit_magnitude_;

        // Số bucket cần để phủ tới `highest`.
        int64_t smallest_untrackable = sub_bucket_count_ << unit_magnitude_;
        bucket_count_ = 1;
        while (smallest_untrackable <= highest) {
            if (smallest_untrackable > INT64_MAX / 2) { bucket_count_++; break; }
            smallest_untrackable <<= 1;
            bucket_count_++;
        }
        counts_.assign((size_t)((bucket_count_ + 1) * sub_bucket_half_count_), 0);  // allocate duy nhất
    }

    // Hot path: không allocate, không branch nặng.
    inline bool record(int64_t value, int64_t count = 1) {
        if (value < 0 || value > highest_) { overflow_ += count; return false; }
        counts_[(size_t)counts_index(value)] += count;
        total_ += count;
        if (value > max_) max_ = value;
        if (value < min_) min_ = value;
        return true;
    }

    // Cơ chế "bù CO" kinh điển của HdrHistogram cho closed-loop: nếu mẫu dài hơn
    // khoảng chờ kỳ vọng, chèn thêm các mẫu "lẽ ra phải có" (value - k*interval).
    // Chỉ dùng để SO SÁNH; open-loop đo đúng ngay từ đầu nên không cần.
    void record_corrected(int64_t value, int64_t expected_interval) {
        record(value);
        if (expected_interval <= 0) return;
        for (int64_t v = value - expected_interval; v >= expected_interval; v -= expected_interval)
            record(v);
    }

    void merge(const HdrHistogram& o) {
        for (size_t i = 0; i < counts_.size() && i < o.counts_.size(); ++i) counts_[i] += o.counts_[i];
        total_ += o.total_;
        overflow_ += o.overflow_;
        if (o.max_ > max_) max_ = o.max_;
        if (o.min_ < min_) min_ = o.min_;
    }

    int64_t total() const { return total_; }
    int64_t max() const { return max_; }
    int64_t overflow() const { return overflow_; }

    double mean() const {
        if (total_ == 0) return 0;
        double s = 0;
        for (size_t i = 0; i < counts_.size(); ++i)
            if (counts_[i]) s += (double)counts_[i] * (double)median_equivalent(value_at_index((int64_t)i));
        return s / (double)total_;
    }

    double stddev() const {
        if (total_ == 0) return 0;
        const double m = mean();
        double s = 0;
        for (size_t i = 0; i < counts_.size(); ++i)
            if (counts_[i]) {
                const double d = (double)median_equivalent(value_at_index((int64_t)i)) - m;
                s += (double)counts_[i] * d * d;
            }
        return std::sqrt(s / (double)total_);
    }

    // Giá trị tại percentile p (0..100), trả về highest-equivalent value như bản gốc.
    int64_t value_at_percentile(double p) const {
        if (total_ == 0) return 0;
        if (p > 100.0) p = 100.0;
        int64_t target = (int64_t)std::ceil(p / 100.0 * (double)total_);
        if (target < 1) target = 1;
        int64_t cum = 0;
        for (size_t i = 0; i < counts_.size(); ++i) {
            cum += counts_[i];
            if (cum >= target) return highest_equivalent(value_at_index((int64_t)i));
        }
        return max_;
    }

    // Xuất định dạng .hgrm (tương thích HdrHistogram plotter).
    // value_scale: chia giá trị cho hệ số này (vd 1e6 để ra millisecond).
    void write_hgrm(FILE* f, double value_scale = 1e6, int ticks_per_half_distance = 5) const {
        std::fprintf(f, "%12s %14s %10s %14s\n\n", "Value", "Percentile", "TotalCount", "1/(1-Percentile)");
        if (total_ == 0) return;
        double level = 0.0;
        while (true) {
            int64_t v = value_at_percentile(level);
            int64_t cnt = count_at_or_below(v);
            double pct = (double)cnt / (double)total_;
            if (level >= 100.0 || cnt >= total_) {
                std::fprintf(f, "%12.3f %1.12f %10lld\n", (double)v / value_scale, 1.0, (long long)total_);
                break;
            }
            std::fprintf(f, "%12.3f %1.12f %10lld %14.2f\n", (double)v / value_scale, pct,
                         (long long)cnt, 1.0 / (1.0 - pct));
            // Bước nhảy giảm một nửa mỗi khi khoảng cách tới 100% giảm một nửa
            // => mật độ điểm tăng dần ở phía đuôi (p99, p99.9, p99.99...).
            double half_distance = std::pow(2.0, std::floor(std::log2(100.0 / (100.0 - level))) + 1.0);
            level += 100.0 / (ticks_per_half_distance * half_distance);
        }
        std::fprintf(f, "#[Mean    = %12.3f, StdDeviation   = %12.3f]\n", mean() / value_scale, stddev() / value_scale);
        std::fprintf(f, "#[Max     = %12.3f, Total count    = %12lld]\n", (double)max_ / value_scale, (long long)total_);
        std::fprintf(f, "#[Buckets = %12d, SubBuckets     = %12lld]\n", bucket_count_, (long long)sub_bucket_count_);
    }

    // Xuất bucket thô (value_ns,count) — Python đọc file này để tái tạo phân phối chính xác.
    void write_buckets_csv(FILE* f) const {
        std::fprintf(f, "value_ns,count\n");
        for (size_t i = 0; i < counts_.size(); ++i)
            if (counts_[i])
                std::fprintf(f, "%lld,%lld\n", (long long)median_equivalent(value_at_index((int64_t)i)),
                             (long long)counts_[i]);
    }

    void print_summary(const char* title, double value_scale = 1e6, const char* unit = "ms") const {
        const double ps[] = {50, 90, 99, 99.9, 99.99, 99.999, 100};
        std::printf("== %s (n=%lld, overflow=%lld) ==\n", title, (long long)total_, (long long)overflow_);
        for (double p : ps)
            std::printf("  p%-8g = %10.3f %s\n", p, (double)value_at_percentile(p) / value_scale, unit);
    }

private:
    inline int64_t counts_index(int64_t value) const {
        int pow2ceiling = 64 - __builtin_clzll((uint64_t)(value | sub_bucket_mask_));
        int bucket_index = pow2ceiling - unit_magnitude_ - (sub_bucket_half_count_magnitude_ + 1);
        int64_t sub_bucket_index = value >> (bucket_index + unit_magnitude_);
        return ((int64_t)(bucket_index + 1) << sub_bucket_half_count_magnitude_) +
               (sub_bucket_index - sub_bucket_half_count_);
    }

    // Trả về (giá trị thấp nhất của ô, độ rộng ô) cho counts index i.
    struct Range { int64_t lo; int64_t width; };
    inline Range value_at_index(int64_t i) const {
        int bucket_index = (int)(i >> sub_bucket_half_count_magnitude_) - 1;
        int64_t sub_bucket_index = (i & (sub_bucket_half_count_ - 1)) + sub_bucket_half_count_;
        if (bucket_index < 0) { sub_bucket_index -= sub_bucket_half_count_; bucket_index = 0; }
        return {sub_bucket_index << (bucket_index + unit_magnitude_), (int64_t)1 << (bucket_index + unit_magnitude_)};
    }
    static int64_t highest_equivalent(Range r) { return r.lo + r.width - 1; }
    static int64_t median_equivalent(Range r) { return r.lo + r.width / 2; }

    int64_t count_at_or_below(int64_t v) const {
        int64_t idx = counts_index(v), cum = 0;
        for (int64_t i = 0; i <= idx && i < (int64_t)counts_.size(); ++i) cum += counts_[(size_t)i];
        return cum;
    }

    int64_t lowest_, highest_;
    int sig_figs_;
    int unit_magnitude_ = 0, sub_bucket_half_count_magnitude_ = 0, bucket_count_ = 0;
    int64_t sub_bucket_count_ = 0, sub_bucket_half_count_ = 0, sub_bucket_mask_ = 0;
    std::vector<int64_t> counts_;
    int64_t total_ = 0, overflow_ = 0, max_ = 0, min_ = INT64_MAX;
};

}  // namespace tl
