"""
evt_gpd.py — Task 4: Ước lượng percentile cực hạn bằng Extreme Value Theory (Peaks-Over-Threshold).

=============================== NỀN TẢNG TOÁN HỌC ===============================

1) Định lý Pickands–Balkema–de Haan:
   Với X là latency, ngưỡng u đủ cao, phân phối của phần VƯỢT ngưỡng Y = X - u (điều kiện X > u)
   hội tụ về Generalized Pareto Distribution (GPD):

        P(Y <= y | X > u)  ≈  H(y; ξ, σ) = 1 - (1 + ξ·y/σ)^(-1/ξ)      (ξ ≠ 0)
                                          = 1 - exp(-y/σ)              (ξ = 0)

   - σ > 0  : scale  — "độ rộng" của đuôi (đơn vị: ns).
   - ξ      : shape  — quyết định độ "béo" của đuôi:
        ξ > 0  đuôi nặng (Pareto-like, không có giới hạn trên) — thường gặp khi có queueing/stall;
        ξ = 0  đuôi mũ (exponential);
        ξ < 0  đuôi có cận trên hữu hạn: x_max = u - σ/ξ   (vd: stall bị chặn ở 500ms).

2) Ghép đuôi GPD với phần thân thực nghiệm:
   Gọi ζ_u = P(X > u) ≈ n_u / n   (tỷ lệ mẫu vượt ngưỡng). Với x > u:

        P(X > x) = P(X > u) · P(X > x | X > u)  =  ζ_u · (1 + ξ·(x-u)/σ)^(-1/ξ)

3) Đảo ngược để lấy quantile x_p (vd p = 0.99999), đặt P(X > x_p) = 1 - p:

        1 - p = ζ_u · (1 + ξ·(x_p - u)/σ)^(-1/ξ)
   =>   x_p   = u + (σ/ξ) · [ ((1 - p)/ζ_u)^(-ξ) - 1 ]          (ξ ≠ 0)
        x_p   = u - σ · ln((1 - p)/ζ_u)                         (ξ → 0)

   Điều kiện dùng được: 1 - p < ζ_u, tức percentile mục tiêu phải nằm TRONG đuôi đã fit.

4) Fit tham số bằng Maximum Likelihood (scipy.stats.genpareto.fit) trên các excess y_i = x_i - u,
   cố định loc = 0 (vì đã trừ u). Log-likelihood:

        ℓ(ξ, σ) = -n_u·ln σ - (1 + 1/ξ) · Σ ln(1 + ξ·y_i/σ)

   Lưu ý scipy dùng ký hiệu c = ξ.

5) PHỤ THUỘC THỜI GIAN — vì sao latency open-loop KHÔNG phải i.i.d.:
   Một stall dài D làm mọi request đến trong cửa sổ đó có latency ≈ D, D - Δ, D - 2Δ, ... (một "dốc").
   Các exceedance vì vậy tụ thành CỤM. Hệ quả:
     (a) Ước lượng ζ_u, ξ, σ cho phân phối BIÊN (marginal) vẫn nhất quán (consistent) — percentile
         là thuộc tính của phân phối biên, nên vẫn fit trên toàn bộ exceedance.
     (b) Nhưng LƯỢNG THÔNG TIN thật = số cụm độc lập n_c, không phải n_u. 10.000 mẫu liên tiếp
         chứa 2 stall chỉ cho biết 2 giá trị độ dài stall — không phương pháp nào ngoại suy đúng từ đó.
     (c) Khoảng tin cậy phải dùng BLOCK bootstrap (resample cả khối liên tiếp) để giữ cấu trúc cụm;
         bootstrap từng điểm i.i.d. sẽ cho CI hẹp một cách giả tạo.

   Runs declustering (Smith & Weissman 1994): hai exceedance thuộc cùng cụm nếu giữa chúng có
   ít hơn r quan sát không vượt ngưỡng. Extremal index:  θ̂ = n_c / n_u  ∈ (0, 1],
   1/θ̂ ≈ kích thước cụm trung bình (θ = 1 nghĩa là không tụ cụm).

Tham khảo: Coles (2001), "An Introduction to Statistical Modeling of Extreme Values", ch. 4–5.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import genpareto

# Console Windows mặc định cp1252 -> không in được tiếng Việt / ký hiệu ξ, σ, ζ.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TARGETS = [0.99, 0.999, 0.9999, 0.99999]
MIN_CLUSTERS = 30  # dưới mức này, ngoại suy EVT không đáng tin (quy tắc thực nghiệm)


@dataclass
class GpdFit:
    u: float        # threshold (ns)
    xi: float       # shape ξ
    sigma: float    # scale σ (ns)
    zeta_u: float   # P(X > u)
    n: int          # tổng số mẫu
    n_u: int        # số mẫu vượt ngưỡng


@dataclass
class Clusters:
    n_clusters: int
    theta: float              # extremal index θ̂ = n_c / n_u
    maxima: np.ndarray        # max của từng cụm
    spans: np.ndarray         # số quan sát mà mỗi cụm trải dài


@dataclass
class EvtReport:
    fit: GpdFit
    clusters: Clusters
    estimates: dict[float, float]
    ci: dict[float, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------- I/O

def load_raw(path: str) -> np.ndarray:
    """Đọc raw CSV từ open_loop --raw-out. Cột cuối là latency_ns; các dòng đã sắp theo intended time."""
    d = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    return d[:, -1]


# ------------------------------------------------------------------------------- GPD core

def fit_gpd(samples: np.ndarray, tail_fraction: float = 0.05) -> GpdFit:
    """Peaks-over-threshold: u = quantile (1 - tail_fraction), fit GPD lên excess bằng MLE."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    u = float(np.quantile(x, 1.0 - tail_fraction))
    excess = x[x > u] - u                     # y_i = x_i - u  (> 0)
    n_u = excess.size
    if n_u < 30:
        raise ValueError(f"Quá ít điểm vượt ngưỡng ({n_u}); tăng sample hoặc tail_fraction.")
    xi, _loc, sigma = genpareto.fit(excess, floc=0.0)  # MLE, scipy: c ≡ ξ
    return GpdFit(u=u, xi=float(xi), sigma=float(sigma), zeta_u=n_u / n, n=n, n_u=n_u)


def gpd_quantile(fit: GpdFit, p: float) -> float:
    """x_p = u + (σ/ξ)·[((1-p)/ζ_u)^(-ξ) - 1]   (hoặc dạng log khi ξ ≈ 0)."""
    tail_prob = 1.0 - p
    if tail_prob >= fit.zeta_u:
        raise ValueError(f"p={p} không nằm trong vùng đuôi (cần 1-p < ζ_u={fit.zeta_u:.4g}).")
    ratio = tail_prob / fit.zeta_u
    if abs(fit.xi) < 1e-9:
        return fit.u - fit.sigma * np.log(ratio)
    return fit.u + (fit.sigma / fit.xi) * (ratio ** (-fit.xi) - 1.0)


# ------------------------------------------------------------------------------- dependence

def decluster_runs(x: np.ndarray, u: float, run_gap: int = 100) -> Clusters:
    """Runs declustering: exceedance cách nhau < run_gap quan sát -> cùng một cụm.

    x phải theo THỨ TỰ THỜI GIAN (open_loop đã sắp theo intended time).
    """
    idx = np.flatnonzero(x > u)
    if idx.size == 0:
        return Clusters(0, float("nan"), np.empty(0), np.empty(0))
    # Ranh giới cụm: nơi khoảng cách giữa 2 exceedance liên tiếp >= run_gap.
    breaks = np.flatnonzero(np.diff(idx) >= run_gap) + 1
    groups = np.split(idx, breaks)
    maxima = np.array([x[g].max() for g in groups])
    spans = np.array([g[-1] - g[0] + 1 for g in groups])
    return Clusters(len(groups), len(groups) / idx.size, maxima, spans)


def block_bootstrap_ci(x: np.ndarray, ps: list[float], tail_fraction: float, block_len: int,
                       n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> dict[float, tuple[float, float]]:
    """Moving-block bootstrap (Künsch 1989): ghép các khối liên tiếp dài block_len được chọn ngẫu nhiên.

    Khối đủ dài (> độ dài một cụm) => mỗi stall được resample NGUYÊN VẸN, giữ đúng cấu trúc phụ thuộc.
    CI = percentile [α/2, 1-α/2] của các ước lượng x_p trên mẫu bootstrap.
    """
    rng = np.random.default_rng(seed)
    n = x.size
    block_len = int(min(max(block_len, 1), n))
    n_blocks = int(np.ceil(n / block_len))
    est = {p: [] for p in ps}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        xb = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        try:
            fb = fit_gpd(xb, tail_fraction)
        except ValueError:
            continue
        for p in ps:
            if 1 - p < fb.zeta_u:
                est[p].append(gpd_quantile(fb, p))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {p: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for p, v in est.items() if v}


# ------------------------------------------------------------------------------- threshold diagnostics

def mean_residual_life(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """e(u) = E[X - u | X > u]. Nếu đuôi là GPD thì e(u) = (σ_u0 + ξ·u)/(1 - ξ) — TUYẾN TÍNH theo u.
    Chọn u ở đầu đoạn mà đồ thị bắt đầu thẳng."""
    return np.array([np.mean(x[x > u] - u) if np.any(x > u) else np.nan for u in thresholds])


def shape_stability(x: np.ndarray, tail_fractions: np.ndarray) -> np.ndarray:
    """ξ̂ theo từng ngưỡng. Nếu GPD đúng trên u0 thì ξ không đổi với mọi u > u0 -> chọn vùng ξ ổn định."""
    out = []
    for f in tail_fractions:
        try:
            out.append(fit_gpd(x, float(f)).xi)
        except ValueError:
            out.append(np.nan)
    return np.array(out)


# ------------------------------------------------------------------------------- pipeline

def analyze(x: np.ndarray, tail_fraction: float = 0.05, run_gap: int = 100,
            n_boot: int = 200, ps: list[float] | None = None) -> EvtReport:
    ps = ps or TARGETS
    fit = fit_gpd(x, tail_fraction)
    cl = decluster_runs(x, fit.u, run_gap)
    estimates = {p: gpd_quantile(fit, p) for p in ps if 1 - p < fit.zeta_u}
    rep = EvtReport(fit=fit, clusters=cl, estimates=estimates)

    if cl.n_clusters < MIN_CLUSTERS:
        rep.warnings.append(
            f"chỉ có {cl.n_clusters} sự kiện cực trị độc lập (< {MIN_CLUSTERS}): mẫu quá 'dày' theo thời gian, "
            f"ngoại suy KHÔNG đáng tin. Hãy lấy mẫu thưa hơn trong khoảng thời gian dài hơn.")
    if n_boot > 0:
        # Khối dài gấp đôi cụm dài nhất để không cắt đôi một stall.
        block = int(max(2 * (cl.spans.max() if cl.spans.size else 1), run_gap, x.size // 50))
        rep.ci = block_bootstrap_ci(x, list(estimates), tail_fraction, block, n_boot)
    return rep


def describe(fit: GpdFit) -> str:
    kind = "đuôi nặng" if fit.xi > 0.05 else ("đuôi bị chặn" if fit.xi < -0.05 else "≈ exponential")
    s = (f"u = {fit.u/1e6:.3f} ms | ξ = {fit.xi:+.4f} ({kind}) | σ = {fit.sigma/1e6:.3f} ms | "
         f"ζ_u = {fit.zeta_u:.4f} ({fit.n_u}/{fit.n})")
    if fit.xi < 0:
        s += f" | x_max = u - σ/ξ = {(fit.u - fit.sigma / fit.xi)/1e6:.3f} ms"
    return s


def print_report(rep: EvtReport, x: np.ndarray, label: str = "") -> None:
    print(f"[EVT{(' ' + label) if label else ''}] n = {x.size} mẫu")
    print("  " + describe(rep.fit))
    print(f"  cụm độc lập n_c = {rep.clusters.n_clusters}, extremal index θ̂ = {rep.clusters.theta:.4f} "
          f"(kích thước cụm TB ≈ {1/rep.clusters.theta:.0f})")
    for p, est in rep.estimates.items():
        emp = np.quantile(x, p) if (1 - p) * x.size >= 1 else float("nan")
        ci = rep.ci.get(p)
        ci_s = f" [95% CI {ci[0]/1e6:8.2f} – {ci[1]/1e6:8.2f}]" if ci else ""
        print(f"  p{p*100:<8g}: GPD = {est/1e6:9.3f} ms{ci_s} | empirical = {emp/1e6:9.3f} ms")
    for w in rep.warnings:
        print("  WARNING: " + w)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit GPD lên đuôi latency và ước lượng percentile cực hạn")
    ap.add_argument("raw_csv", help="raw CSV từ open_loop --raw-out")
    ap.add_argument("--sample-size", type=int, default=10_000, help="chỉ dùng N mẫu đầu")
    ap.add_argument("--tail-fraction", type=float, default=0.05, help="tỷ lệ top dữ liệu dùng làm đuôi")
    ap.add_argument("--run-gap", type=int, default=100, help="tham số r của runs declustering")
    ap.add_argument("--bootstrap", type=int, default=200, help="số lần block bootstrap (0 = tắt)")
    args = ap.parse_args()

    x = load_raw(args.raw_csv)[: args.sample_size]
    print_report(analyze(x, args.tail_fraction, args.run_gap, args.bootstrap), x)


if __name__ == "__main__":
    main()
