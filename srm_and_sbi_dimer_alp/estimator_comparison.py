"""General estimator-comparison diagnostic for the DIMER pipeline's two mirrored workflows.

Decides whether one trained estimator generalizes better than another by the **paired
log-score on the shared `(task, sim)` subset** of the held-out set, tested with the
Diebold-Mariano statistic and, as heavy-tail-robust companions, the Wilcoxon signed-rank
test and a paired bootstrap. Workflow-agnostic: it operates on two per-video loss arrays
keyed by `(task, sim)`, which both workflows' Test-Loss-Distribution artifacts provide.

**Why paired, not two means.** The per-video negative log-density decomposes into an
intrinsic posterior-entropy floor (a property of the video, not the estimator) plus the
estimator's KL error. Pairing on the *same* videos cancels the common entropy term, so the
log-score *difference* isolates the difference of the two estimators' KL divergences to the
truth (Amisano & Giacomini 2007) -- the Diebold-Mariano test of equal predictive accuracy
(Diebold & Mariano 1995). Comparing the two estimators' *mean* losses on different-sized
sets is invalid: it confounds the sets' intrinsic-entropy content. Because these are
per-video (exchangeable) forecasts rather than a time series, the Diebold-Mariano long-run
variance is just the i.i.d. variance, so the statistic is the paired z-test on the loss
differential; the per-video loss has a heavy upper tail, so the Wilcoxon signed-rank test
(median-based) and the paired bootstrap (of the mean) are reported alongside it.

Sign convention: ``improvement`` is ``loss_B - loss_A`` per video, so a **positive** mean
improvement means estimator **A** has the lower loss (A is better).

Pure analysis: numpy, with scipy imported lazily. Imports nothing from
``parameterization`` / ``artifacts``, so it is unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    """Paired log-score comparison of estimator A vs estimator B."""

    label_a: str
    label_b: str
    n_a: int                      # videos in A's loss set
    n_b: int                      # videos in B's loss set
    n_shared: int                 # videos in the shared (task, sim) subset (the paired sample)
    mean_improvement: float       # mean(loss_B - loss_A); > 0 => A better (lower loss)
    median_improvement: float     # median(loss_B - loss_A); robust to the heavy tail
    frac_a_better: float          # fraction of shared videos where A has the lower loss
    dm_stat: float                # Diebold-Mariano statistic (mean / se); sign = mean_improvement's
    dm_pvalue: float              # two-sided p-value (H0: equal predictive accuracy)
    wilcoxon_stat: float          # Wilcoxon signed-rank statistic on the improvement
    wilcoxon_pvalue: float        # two-sided p-value (H0: symmetric about zero)
    boot_ci_low: float            # paired-bootstrap CI of the mean improvement
    boot_ci_high: float
    alpha: float
    verdict: str                  # "A better" | "B better" | "no significant difference"
    improvement: np.ndarray       # (n_shared,) per-video loss_B - loss_A on the shared subset


def align_by_key(keys_a, loss_a, keys_b, loss_b):
    """Intersect two per-video loss sets on their ``(task, sim)`` keys.

    Args:
        keys_a / keys_b: ``(N, 2)`` integer arrays of ``(task_index, sim_index)``.
        loss_a / loss_b: ``(N,)`` per-video losses, row-aligned to their keys.

    Returns:
        ``(shared_keys (M, 2), la (M,), lb (M,))`` -- the losses on the shared subset, in a
        common, deterministic key order. Alignment is by key identity, never by position,
        so two sets stored in different orders or with different coverage still pair
        correctly.
    """
    keys_a = np.asarray(keys_a)
    keys_b = np.asarray(keys_b)
    index_a = {tuple(int(v) for v in k): i for i, k in enumerate(keys_a)}
    index_b = {tuple(int(v) for v in k): i for i, k in enumerate(keys_b)}
    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        raise ValueError("estimator comparison: the two loss sets share no (task, sim) keys")
    ia = [index_a[k] for k in shared]
    ib = [index_b[k] for k in shared]
    return (np.array(shared, dtype=int),
            np.asarray(loss_a, dtype=float)[ia],
            np.asarray(loss_b, dtype=float)[ib])


def diebold_mariano(improvement):
    """Diebold-Mariano statistic + two-sided p-value on the loss differential.

    ``improvement = loss_B - loss_A``. With per-video (exchangeable) forecasts the long-run
    variance is the i.i.d. variance, so ``stat = mean / (std / sqrt(n))`` and the p-value is
    the two-sided normal tail -- the paired z-test of equal predictive accuracy.
    """
    from scipy.stats import norm  # lazy

    d = np.asarray(improvement, dtype=float)
    n = d.size
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    stat = mean / se if se > 0 else float("nan")
    pvalue = float(2.0 * norm.sf(abs(stat))) if np.isfinite(stat) else float("nan")
    return stat, pvalue


def wilcoxon_test(improvement):
    """Wilcoxon signed-rank test that the improvement is symmetric about zero (robust)."""
    from scipy.stats import wilcoxon  # lazy

    d = np.asarray(improvement, dtype=float)
    nonzero = d[d != 0.0]
    if nonzero.size < 1:
        return float("nan"), float("nan")
    res = wilcoxon(nonzero)
    return float(res.statistic), float(res.pvalue)


def paired_bootstrap(improvement, *, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean improvement (resamples video pairs)."""
    d = np.asarray(improvement, dtype=float)
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean(d[rng.integers(0, n, n)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare(keys_a, loss_a, keys_b, loss_b, *,
            label_a: str = "A", label_b: str = "B",
            n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> ComparisonResult:
    """Full paired log-score comparison of estimator A vs estimator B."""
    n_a = int(np.asarray(loss_a).size)
    n_b = int(np.asarray(loss_b).size)
    _, la, lb = align_by_key(keys_a, loss_a, keys_b, loss_b)
    improvement = lb - la                                  # loss_B - loss_A; > 0 => A better

    dm_stat, dm_p = diebold_mariano(improvement)
    w_stat, w_p = wilcoxon_test(improvement)
    ci_lo, ci_hi = paired_bootstrap(improvement, n_boot=n_boot, seed=seed, alpha=alpha)
    mean_impr = float(np.mean(improvement))

    if np.isfinite(dm_p) and dm_p < alpha:
        verdict = f"{label_a} better" if mean_impr > 0 else f"{label_b} better"
    else:
        verdict = "no significant difference"

    return ComparisonResult(
        label_a=label_a, label_b=label_b, n_a=n_a, n_b=n_b, n_shared=int(improvement.size),
        mean_improvement=mean_impr,
        median_improvement=float(np.median(improvement)),
        frac_a_better=float(np.mean(improvement > 0)),
        dm_stat=dm_stat, dm_pvalue=dm_p,
        wilcoxon_stat=w_stat, wilcoxon_pvalue=w_p,
        boot_ci_low=ci_lo, boot_ci_high=ci_hi, alpha=alpha, verdict=verdict,
        improvement=improvement,
    )


def summarize(result: ComparisonResult) -> list[dict]:
    """Flatten a :class:`ComparisonResult` into printable rows."""
    r = result
    return [
        {"metric": "shared (task,sim) videos", "value": f"{r.n_shared}  (A={r.n_a}, B={r.n_b})"},
        {"metric": f"mean improvement (loss_{r.label_b} - loss_{r.label_a})",
         "value": f"{r.mean_improvement:+.4f}  (>0 => {r.label_a} better)"},
        {"metric": "median improvement (robust)", "value": f"{r.median_improvement:+.4f}"},
        {"metric": f"fraction where {r.label_a} wins", "value": f"{r.frac_a_better:.3f}"},
        {"metric": "Diebold-Mariano", "value": f"stat={r.dm_stat:+.3f}, p={r.dm_pvalue:.3g}"},
        {"metric": "Wilcoxon signed-rank", "value": f"stat={r.wilcoxon_stat:.4g}, p={r.wilcoxon_pvalue:.3g}"},
        {"metric": f"bootstrap {int((1-r.alpha)*100)}% CI of mean improvement",
         "value": f"[{r.boot_ci_low:+.4f}, {r.boot_ci_high:+.4f}]"},
        {"metric": "verdict", "value": r.verdict},
    ]
