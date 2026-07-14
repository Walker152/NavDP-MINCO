from __future__ import annotations

import math
import numpy as np


def sample_rule(n):
    return "CASES_ONLY" if n < 5 else "DESCRIPTIVE_BOOTSTRAP" if n < 10 else "PAIRED_TEST_ALLOWED"


def bootstrap_ci(values, confidence=0.95, iterations=2000, seed=0, statistic=np.mean):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return float("nan"), float("nan")
    rng = np.random.default_rng(seed); samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    estimates = np.apply_along_axis(statistic, 1, samples); alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def proportion_ci(successes, count, confidence=0.95):
    if count <= 0: return {"low": float("nan"), "high": float("nan"), "method": "wilson"}
    z = 1.959963984540054; p = successes / count; denom = 1 + z*z/count
    center = (p + z*z/(2*count)) / denom; margin = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denom
    return {"low": max(0.0, center-margin), "high": min(1.0, center+margin), "method": "wilson"}


def mcnemar_test(b, c):
    n = int(b + c)
    if n == 0: return {"statistic": 0.0, "p_value": 1.0, "method": "exact_binomial"}
    k = int(min(b, c)); tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"statistic": float(abs(b-c)), "p_value": min(1.0, 2*tail), "method": "exact_binomial"}


def wilcoxon_paired(baseline, method):
    baseline = np.asarray(baseline, float); method = np.asarray(method, float)
    valid = np.isfinite(baseline) & np.isfinite(method); baseline, method = baseline[valid], method[valid]
    if len(baseline) < 10: return {"statistic":float("nan"), "p_value":float("nan"), "method":"not_run_small_sample"}
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(baseline, method)
        return {"statistic":float(result.statistic), "p_value":float(result.pvalue), "method":"scipy_wilcoxon"}
    except Exception:
        delta = method - baseline; nonzero = delta[delta != 0]
        if not len(nonzero): return {"statistic":0.0, "p_value":1.0, "method":"normal_approximation"}
        positives = int(np.sum(nonzero > 0)); n = len(nonzero); z = (positives - n/2) / math.sqrt(n/4)
        p = math.erfc(abs(z) / math.sqrt(2))
        return {"statistic":float(positives), "p_value":float(p), "method":"normal_approximation"}
