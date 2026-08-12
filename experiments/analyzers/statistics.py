from __future__ import annotations

import math
from statistics import NormalDist
from typing import Mapping
import numpy as np


def sample_rule(n):
    return "CASES_ONLY" if n < 5 else "DESCRIPTIVE_BOOTSTRAP" if n < 10 else "PAIRED_TEST_ALLOWED"


def bootstrap_ci(values, confidence=0.95, iterations=2000, seed=0, statistic=np.mean):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return float("nan"), float("nan")
    rng = np.random.default_rng(seed); samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    estimates = np.apply_along_axis(statistic, 1, samples); alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for one binomial outcome denominator."""

    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def paired_bootstrap_ci(
    baseline_by_key: Mapping[str, float],
    method_by_key: Mapping[str, float],
    *,
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 20260812,
) -> dict[str, float | int | str]:
    """Bootstrap the paired mean delta after exact episode/case-key joining."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    keys = sorted(set(baseline_by_key) & set(method_by_key))
    deltas = []
    for key in keys:
        try:
            baseline = float(baseline_by_key[key])
            method = float(method_by_key[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(baseline) and math.isfinite(method):
            deltas.append(method - baseline)
    if not deltas:
        return {
            "estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "paired_count": 0,
            "confidence": confidence,
            "iterations": iterations,
            "seed": seed,
            "cluster_unit": "episode_or_case_key",
        }
    values = np.asarray(deltas, dtype=float)
    low, high = bootstrap_ci(
        values,
        confidence=confidence,
        iterations=iterations,
        seed=seed,
    )
    return {
        "estimate": float(np.mean(values)),
        "ci_low": low,
        "ci_high": high,
        "paired_count": len(values),
        "confidence": confidence,
        "iterations": iterations,
        "seed": seed,
        "cluster_unit": "episode_or_case_key",
    }


def proportion_ci(successes, count, confidence=0.95):
    if count <= 0: return {"low": float("nan"), "high": float("nan"), "method": "wilson"}
    low, high = wilson_interval(successes, count, confidence)
    return {"low": low, "high": high, "method": "wilson"}


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
