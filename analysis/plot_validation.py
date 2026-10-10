"""
plot_validation.py — Task 5: Validation & Visualization.

  fig1  CDF (trục đuôi log): Closed-loop vs Open-loop                 => Coordinated Omission
  fig2  EVT từ sample nhỏ (dày vs thưa) vs Ground Truth (+ lý thuyết)  => sai số + 95% CI
  fig3  Chẩn đoán ngưỡng: Mean Residual Life + độ ổn định ξ            => chọn u có cơ sở
  fig4  Sai số theo cỡ mẫu: EVT vs percentile thực nghiệm              => "EVT cần ít mẫu hơn bao nhiêu lần"
  summary.json  mọi con số chính (để đưa vào README / CV)

Trục x của fig1/fig2 dùng thang 1/(1-p) (log) như HdrHistogram plotter: mỗi nấc = thêm một số 9.

MÔ HÌNH LÝ THUYẾT (server --stall-mode global): pause D ~ U[a, b], chu kỳ trung bình
E[C] = gc_interval + E[D]. Request đến trong một pause dài D phải chờ phần còn lại của pause, nên
latency ≈ phần dư, phân phối đều trên [0, D]. Do đó với x > 0:

    P(L > x) = E[(D - x)^+] / E[C]
             = (E[D] - x) / E[C]                  nếu x <= a
             = (b - x)^2 / (2 (b - a) E[C])       nếu a < x < b

Nhánh thứ hai chính là một GPD với ξ = -1/2 và cận trên b — EVT fit đúng phải tìm lại được ξ ≈ -0.5.

Giới hạn: công thức trên là xấp xỉ ρ = λ·S → 0. Ở tải thật, sau mỗi pause server còn phải xả backlog
(λ·D request), nên latency đo được CAO HƠN lý thuyết ở vùng p90–p99 (phần thân đuôi), nhưng DẠNG
(b - x)^2 ở sát cận trên — và do đó ξ = -1/2 — vẫn giữ nguyên. Vì vậy chỉ dùng lý thuyết để kiểm tra ξ
(fig3) và ghi tham khảo vào summary.json, không dùng làm ground truth.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evt_gpd import TARGETS, analyze, describe, fit_gpd, gpd_quantile, load_raw, mean_residual_life, shape_stability

# Bảng màu categorical đã validate (slot 1-3) + mực chữ trung tính.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ------------------------------------------------------------------------------- data

class BucketDist:
    """Phân phối tái tạo từ file *_buckets.csv (value_ns,count) của HdrHistogram."""

    def __init__(self, path: str):
        d = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        self.values = d[:, 0]
        self.cum = np.cumsum(d[:, 1])
        self.total = int(self.cum[-1])

    def quantile(self, p):
        idx = np.searchsorted(self.cum, np.ceil(np.asarray(p) * self.total), side="left")
        return self.values[np.clip(idx, 0, self.values.size - 1)]


def theory_quantile(p: float, a_ms: float, b_ms: float, interval_ms: float) -> float:
    """Quantile lý thuyết (ns) cho mode global — xem docstring đầu file."""
    ed = (a_ms + b_ms) / 2
    ec = interval_ms + ed
    tail = 1 - p
    x = b_ms - np.sqrt(2 * (b_ms - a_ms) * ec * tail)
    if x < a_ms:
        x = ed - ec * tail
    return max(x, 0.0) * 1e6


def stationarity_check(raw_path: str, window_s: float = 30.0, tol: float = 0.25) -> dict:
    """Ground truth chỉ có nghĩa nếu run DỪNG (stationary): chia run thành các cửa sổ thời gian và gắn cờ
    cửa sổ có p99.9 lệch > tol so với median các cửa sổ — dấu hiệu một sự kiện ngoại lai (OS freeze,
    tiến trình nền...) không thuộc hệ thống đang đo. Cờ này KHÔNG tự loại dữ liệu; nó báo run cần chạy lại."""
    d = np.loadtxt(raw_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    t, x = d[:, 0] / 1e9, d[:, -1]
    starts = np.arange(0, t.max(), window_s)
    p999 = np.array([np.quantile(x[(t >= a) & (t < a + window_s)], 0.999) for a in starts])
    med = float(np.median(p999))
    bad = [(float(a), float(v) / 1e6) for a, v in zip(starts, p999) if abs(v - med) / med > tol]
    return {"window_s": window_s, "median_window_p999_ms": med / 1e6,
            "window_p999_ms": [float(v) / 1e6 for v in p999], "outlier_windows": bad, "stationary": not bad}


def supported(p: float, n: int, min_beyond: int = 10) -> bool:
    """Percentile p chỉ 'đo được' từ n mẫu nếu có ít nhất min_beyond mẫu vượt qua nó."""
    return (1 - p) * n >= min_beyond


# ------------------------------------------------------------------------------- styling

def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold")


def nines_axis(ax, max_nines: int) -> None:
    ax.set_xscale("log")
    ticks = [10.0 ** k for k in range(0, max_nines + 1)]
    labels = ["0%"] + [f"{100 * (1 - 1 / t):.{max(0, k - 2)}f}%" for k, t in enumerate(ticks) if k > 0]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Percentile", color=INK_2)


def percentile_curve(dist: BucketDist, max_nines: int):
    inv = np.logspace(0, max_nines, 400)          # 1/(1-p)
    p = np.clip(1.0 - 1.0 / inv, 0, 1 - 1.0 / dist.total)
    return inv, dist.quantile(p) / 1e6


def save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------- fig1

def plot_co(closed: BucketDist, opened: BucketDist, path: str) -> dict:
    max_nines = int(np.floor(np.log10(min(closed.total, opened.total))))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for dist, color, name in ((closed, C_BLUE, "Closed-loop (đo sai)"), (opened, C_ORANGE, "Open-loop (đo đúng)")):
        x, y = percentile_curve(dist, max_nines)
        ax.plot(x, y, color=color, linewidth=2, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    nines_axis(ax, max_nines)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log)", color=INK_2)
    c99, o99 = float(closed.quantile(0.99)) / 1e6, float(opened.quantile(0.99)) / 1e6
    style(ax, f"Coordinated Omission: p99 closed-loop {c99:.2f} ms vs open-loop {o99:.0f} ms")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    save(fig, path)
    return {"closed_p99_ms": c99, "open_p99_ms": o99, "under_report_factor_p99": o99 / c99}


# ------------------------------------------------------------------------------- fig2

def plot_evt(truth: BucketDist, samples: dict[str, np.ndarray], tail_fraction: float, n_boot: int,
             path: str) -> dict:
    max_nines = int(np.floor(np.log10(truth.total)))
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)

    x, y = percentile_curve(truth, max_nines)
    ax.plot(x, y, color=C_BLUE, linewidth=2, label=f"Ground truth (long-run, n={truth.total:,})")

    out = {}
    colors = [C_ORANGE, C_AQUA]
    for (name, xs), color, dy in zip(samples.items(), colors, (-16, 10)):
        rep = analyze(xs, tail_fraction, n_boot=n_boot)
        fit = rep.fit
        inv_tail = np.logspace(np.log10(1 / fit.zeta_u) + 1e-6, max_nines, 200)
        ax.plot(inv_tail, [gpd_quantile(fit, 1 - 1 / v) / 1e6 for v in inv_tail], color=color, linewidth=2,
                label=f"EVT · {name} (n={xs.size:,}, {rep.clusters.n_clusters} sự kiện độc lập)")
        res = {"n": int(xs.size), "n_clusters": rep.clusters.n_clusters, "theta": rep.clusters.theta,
               "xi": fit.xi, "sigma_ms": fit.sigma / 1e6, "u_ms": fit.u / 1e6, "targets": {}}
        for p, est in rep.estimates.items():
            inv = 1 / (1 - p)
            if inv > 10 ** max_nines or not supported(p, truth.total):
                continue
            gt = float(truth.quantile(p))
            err = (est - gt) / gt * 100
            lo, hi = rep.ci.get(p, (est, est))
            ax.errorbar([inv], [est / 1e6], yerr=[[(est - lo) / 1e6], [(hi - est) / 1e6]], fmt="o", color=color,
                        markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.5, capsize=4)
            ax.annotate(f"{err:+.1f}%", (inv, est / 1e6), xytext=(8, dy), textcoords="offset points",
                        color=INK, fontsize=8)
            res["targets"][f"p{p*100:g}"] = {"truth_ms": gt / 1e6, "evt_ms": est / 1e6, "error_pct": err,
                                             "ci95_ms": [lo / 1e6, hi / 1e6]}
        out[name] = res
        print(f"  [{name}] {describe(fit)} | n_c={rep.clusters.n_clusters}")
        for k, v in res["targets"].items():
            print(f"     {k:<9} truth={v['truth_ms']:8.2f} ms  EVT={v['evt_ms']:8.2f} ms  "
                  f"err={v['error_pct']:+6.1f}%  CI95=[{v['ci95_ms'][0]:.1f}, {v['ci95_ms'][1]:.1f}]")

    for p in TARGETS:  # chấm ground truth tại các mốc
        if supported(p, truth.total) and 1 / (1 - p) <= 10 ** max_nines:
            ax.plot([1 / (1 - p)], [float(truth.quantile(p)) / 1e6], "o", color=C_BLUE, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)

    nines_axis(ax, max_nines)
    ax.set_ylabel("Latency (ms)", color=INK_2)
    style(ax, "EVT từ 10k mẫu vs Ground Truth (nhãn = sai số, thanh = 95% block-bootstrap CI)")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", fontsize=8.5)
    save(fig, path)
    return out


# ------------------------------------------------------------------------------- fig3

def plot_diagnostics(x: np.ndarray, chosen_fraction: float, path: str) -> None:
    fractions = np.linspace(0.005, 0.12, 24)
    thresholds = np.quantile(x, 1 - fractions)
    mrl = mean_residual_life(x, thresholds)
    xis = shape_stability(x, fractions)
    u_sel = np.quantile(x, 1 - chosen_fraction)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    a1.plot(thresholds / 1e6, mrl / 1e6, color=C_BLUE, linewidth=2)
    a1.axvline(u_sel / 1e6, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    a1.annotate(f"u chọn = {u_sel/1e6:.0f} ms", (u_sel / 1e6, np.nanmax(mrl) / 1e6), xytext=(6, -4),
                textcoords="offset points", color=INK, fontsize=8)
    a1.set_xlabel("Ngưỡng u (ms)", color=INK_2)
    a1.set_ylabel("E[X − u | X > u] (ms)", color=INK_2)
    style(a1, "Mean Residual Life (tuyến tính ⇒ GPD hợp lệ)")

    a2.plot(fractions * 100, xis, color=C_BLUE, linewidth=2, marker="o", markersize=3)
    a2.axhline(-0.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    a2.annotate("ξ lý thuyết = −0.5", (fractions[-1] * 100, -0.5), xytext=(-4, 6), textcoords="offset points",
                ha="right", color=INK, fontsize=8)
    a2.axvline(chosen_fraction * 100, color=INK_2, linewidth=1, linestyle=(0, (1, 2)))
    a2.set_xlabel("Tail fraction dùng để fit (%)", color=INK_2)
    a2.set_ylabel("ξ̂ (shape)", color=INK_2)
    style(a2, "Độ ổn định của ξ theo ngưỡng")
    save(fig, path)


# ------------------------------------------------------------------------------- fig4

def equal_error_n(sizes: list[int], errs: list[float], target: float) -> float | None:
    """n mà đường sai số (log-log) của một phương pháp chạm mức `target` — nội suy giữa 2 điểm kề nhau."""
    for (n0, e0), (n1, e1) in zip(zip(sizes, errs), zip(sizes[1:], errs[1:])):
        if np.isnan(e0) or np.isnan(e1):
            continue
        if e0 >= target >= e1 and e0 > e1:
            t = (np.log(e0) - np.log(target)) / (np.log(e0) - np.log(e1))
            return float(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0))))
    return None


def plot_convergence(raw_long: np.ndarray, tail_fraction: float, path: str, reps: int = 40,
                     seed: int = 0) -> dict:
    """Lấy mẫu ngẫu nhiên n điểm từ NỬA ĐẦU long-run (≈ lấy mẫu thưa, i.i.d. theo phân phối biên);
    ground truth lấy từ NỬA SAU (độc lập về thời gian) để không 'chấm điểm bằng chính đề bài'.

    Metric "sample efficiency": với mỗi n của EVT, empirical cần bao nhiêu mẫu để đạt CÙNG median
    sai số (nội suy log-log) => tỷ lệ n_emp / n_evt. Chỉ tính khi sai số EVT còn cao hơn hẳn sàn nhiễu
    của ground truth (đuôi phải của đường cong phẳng ra vì truth cũng chỉ là một mẫu hữu hạn)."""
    rng = np.random.default_rng(seed)
    half = raw_long.size // 2
    pool, truth_part = raw_long[:half], raw_long[half:]
    ps = [p for p in (0.999, 0.9999) if supported(p, truth_part.size, 20)]
    sizes = [n for n in (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000) if n <= pool.size // 2]
    result = {"pool_n": int(pool.size), "truth_n": int(truth_part.size), "reps": reps, "percentiles": {}}

    fig, axes = plt.subplots(1, len(ps), figsize=(5.5 * len(ps), 4.3), facecolor=SURFACE, squeeze=False)
    for ax, p in zip(axes[0], ps):
        gt = float(np.quantile(truth_part, p))
        med = {"EVT/GPD": [], "Empirical": []}
        for n in sizes:
            errs = {"EVT/GPD": [], "Empirical": []}
            for _ in range(reps):
                s = rng.choice(pool, size=n, replace=False)
                errs["Empirical"].append(abs(np.quantile(s, p) - gt) / gt)
                try:
                    f = fit_gpd(s, tail_fraction)
                    if 1 - p < f.zeta_u:
                        errs["EVT/GPD"].append(abs(gpd_quantile(f, p) - gt) / gt)
                except ValueError:
                    pass
            for k in med:
                med[k].append(float(np.median(errs[k])) if errs[k] else np.nan)

        for (k, v), color in zip(med.items(), (C_ORANGE, C_BLUE)):
            ax.plot(sizes, np.array(v) * 100, color=color, linewidth=2, marker="o", markersize=5, label=k)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%" if v in (2, 3, 5) else ""))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Số mẫu n", color=INK_2)
        ax.set_ylabel("Median |sai số tương đối|", color=INK_2)

        # Sàn nhiễu ≈ sai số nhỏ nhất đạt được; chỉ so hiệu suất khi EVT còn > 1.5x sàn.
        floor = float(np.nanmin(med["EVT/GPD"] + med["Empirical"]))
        ratios = {}
        for n, e in zip(sizes, med["EVT/GPD"]):
            if np.isnan(e) or e < 1.5 * floor:
                continue
            n_emp = equal_error_n(sizes, med["Empirical"], e)
            if n_emp:
                ratios[n] = n_emp / n
        eff = float(np.median(list(ratios.values()))) if ratios else None
        key = f"p{p*100:g}"
        result["percentiles"][key] = {"truth_ms": gt / 1e6, "sizes": sizes, "median_err_evt": med["EVT/GPD"],
                                      "median_err_empirical": med["Empirical"], "noise_floor": floor,
                                      "equal_error_ratio_by_n": ratios, "median_sample_efficiency": eff}
        style(ax, f"{key}: EVT ≈ {eff:.1f}× ít mẫu hơn" if eff else f"{key}")
        ax.text(0.98, 0.97, f"truth = {gt/1e6:.1f} ms\nsàn nhiễu ground truth ≈ {floor*100:.2f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK_2)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower left")
        print(f"  [convergence {key}] truth={gt/1e6:.2f} ms | equal-error ratio n_emp/n_evt = "
              f"{ {k: round(v, 2) for k, v in ratios.items()} } | median = {eff}")
    save(fig, path)
    return result


# ------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", required=True, help="closed-loop *_buckets.csv")
    ap.add_argument("--open", required=True, help="open-loop long-run *_buckets.csv (ground truth)")
    ap.add_argument("--long-raw", help="raw CSV của long-run (cho fig4)")
    ap.add_argument("--dense-raw", help="raw CSV run ngắn, rate cao (mẫu 'dày')")
    ap.add_argument("--sparse-raw", help="raw CSV run dài, rate thấp (mẫu 'thưa')")
    ap.add_argument("--sample-size", type=int, default=10_000)
    ap.add_argument("--tail-fraction", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--theory", help="a_ms,b_ms,interval_ms — vẽ đường lý thuyết cho mode global, vd 100,500,2000")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)  # noqa: E731
    summary = {}

    closed, opened = BucketDist(args.closed), BucketDist(args.open)
    summary["coordinated_omission"] = plot_co(closed, opened, out("fig1_coordinated_omission.png"))
    print(f"[fig1] closed p99 = {summary['coordinated_omission']['closed_p99_ms']:.3f} ms | "
          f"open p99 = {summary['coordinated_omission']['open_p99_ms']:.1f} ms")

    theory = tuple(float(v) for v in args.theory.split(",")) if args.theory else None
    if theory:
        summary["theory_rho0_vs_measured"] = {f"p{p*100:g}": {"theory_ms": theory_quantile(p, *theory) / 1e6,
                                                         "measured_ms": float(opened.quantile(p)) / 1e6}
                                         for p in TARGETS if supported(p, opened.total)}

    samples = {}
    if args.dense_raw:
        samples["mẫu dày"] = load_raw(args.dense_raw)[: args.sample_size]
    if args.sparse_raw:
        samples["mẫu thưa"] = load_raw(args.sparse_raw)[: args.sample_size]
    if samples:
        print("[fig2] EVT vs ground truth:")
        summary["evt"] = plot_evt(opened, samples, args.tail_fraction, args.bootstrap,
                                  out("fig2_evt_vs_truth.png"))
        diag = samples.get("mẫu thưa", next(iter(samples.values())))
        plot_diagnostics(diag, args.tail_fraction, out("fig3_threshold_diagnostics.png"))

    if args.long_raw:
        st = stationarity_check(args.long_raw)
        summary["stationarity"] = st
        print(f"[stationarity] p99.9 theo cửa sổ {st['window_s']:.0f}s: "
              f"{[round(v) for v in st['window_p999_ms']]} ms")
        if not st["stationary"]:
            print(f"  WARNING: ground truth KHÔNG dừng — cửa sổ ngoại lai (start_s, p99.9_ms): "
                  f"{st['outlier_windows']}. Nên chạy lại long-run.")
        print("[fig4] convergence:")
        summary["convergence"] = plot_convergence(load_raw(args.long_raw), args.tail_fraction,
                                                  out("fig4_sample_efficiency.png"))

    with open(out("summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"[plot] wrote figures + {out('summary.json')}")


if __name__ == "__main__":
    main()
