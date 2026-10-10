"""Unit test cho analysis/evt_gpd.py — chạy: python -m unittest discover -s tests"""

import os
import sys
import unittest

import numpy as np
from scipy.stats import genpareto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from evt_gpd import GpdFit, analyze, decluster_runs, fit_gpd, gpd_quantile  # noqa: E402


class TestGpdQuantile(unittest.TestCase):
    def test_matches_scipy_ppf_when_threshold_is_zero(self):
        # u = 0, ζ_u = 1 => x_p chính là quantile của GPD thuần.
        for xi in (-0.4, 0.0, 0.3):
            fit = GpdFit(u=0.0, xi=xi, sigma=2.0, zeta_u=1.0, n=1, n_u=1)
            for p in (0.5, 0.9, 0.999):
                self.assertAlmostEqual(gpd_quantile(fit, p), genpareto.ppf(p, xi, scale=2.0), places=6)

    def test_rejects_percentile_outside_tail(self):
        fit = GpdFit(u=1.0, xi=0.1, sigma=1.0, zeta_u=0.05, n=100, n_u=5)
        with self.assertRaises(ValueError):
            gpd_quantile(fit, 0.9)  # 1 - 0.9 = 0.1 >= ζ_u


class TestFit(unittest.TestCase):
    def test_recovers_parameters(self):
        rng = np.random.default_rng(1)
        y = genpareto.rvs(0.2, scale=5.0, size=50_000, random_state=rng)
        fit = fit_gpd(y, tail_fraction=0.999999)  # u ≈ min => gần như toàn bộ dữ liệu là đuôi
        self.assertAlmostEqual(fit.xi, 0.2, delta=0.03)
        self.assertAlmostEqual(fit.sigma, 5.0, delta=0.3)

    def test_extrapolates_beyond_small_sample(self):
        # Thân exponential + đuôi Pareto (ξ = 0.25). p99.99 thật tính giải tích được.
        rng = np.random.default_rng(2)
        n = 10_000
        body = rng.exponential(1.0, size=n)
        is_tail = rng.random(n) < 0.05
        u_true = -np.log(0.05)  # quantile 95% của Exp(1)
        body[is_tail] = u_true + genpareto.rvs(0.25, scale=1.0, size=is_tail.sum(), random_state=rng)
        truth = u_true + genpareto.ppf(1 - 1e-4 / 0.05, 0.25, scale=1.0)
        est = gpd_quantile(fit_gpd(body, 0.05), 0.9999)
        self.assertLess(abs(est - truth) / truth, 0.30)  # ngoại suy 100x ra ngoài vùng có dữ liệu


class TestDeclustering(unittest.TestCase):
    def test_counts_stall_ramps_as_clusters(self):
        # 3 stall: mỗi stall là một dốc tuyến tính 300 mẫu (giống open-loop khi server đứng).
        x = np.full(20_000, 0.1)
        for start in (2_000, 9_000, 15_000):
            x[start:start + 300] = np.linspace(400, 1, 300)
        cl = decluster_runs(x, u=50.0, run_gap=100)
        self.assertEqual(cl.n_clusters, 3)
        self.assertLess(cl.theta, 0.02)  # tụ cụm mạnh: θ ≈ 3 / 789

    def test_analyze_warns_when_few_independent_events(self):
        x = np.full(10_000, 0.1)
        x[5_000:5_600] = np.linspace(500, 1, 600)
        rep = analyze(x, tail_fraction=0.05, n_boot=0)
        self.assertTrue(any("độc lập" in w for w in rep.warnings))


if __name__ == "__main__":
    unittest.main()
