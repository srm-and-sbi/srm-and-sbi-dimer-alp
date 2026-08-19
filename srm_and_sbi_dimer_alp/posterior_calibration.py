"""General posterior-calibration diagnostics for the DIMER pipeline's two mirrored workflows.

Both workflows -- biology (infers the reaction-diffusion parameters) and detector
(infers the imaging parameters) -- train a neural posterior ``q(theta | x)`` over
their own target-theta vector. This module scores how well-calibrated that posterior
is with four established simulation-based-calibration diagnostics, and does so for
either workflow unchanged: it operates purely on pre-drawn quantities in the active
workflow's target-theta space, so it never needs to know which workflow produced them.

Diagnostics (each wraps sbi's validated statistics; citations inline):

  - **SBC** -- simulation-based calibration (Talts et al. 2018). The rank of the true
    theta among ``L`` posterior samples is uniform on ``{0..L}`` for every marginal iff
    the posterior is calibrated. Uniformity is scored with sbi's ``check_sbc``
    (Kolmogorov-Smirnov test + classifier two-sample test).
  - **Expected coverage** (Deistler et al. 2022 / Hermans et al. 2022). The rank of
    the true theta's log-density among the posterior samples' log-densities is likewise
    uniform when calibrated; the empirical-versus-nominal coverage curve reads off
    over- or under-confidence directly.
  - **TARP** -- tests of accuracy with random points (Lemos et al. 2023), a necessary
    and sufficient coverage test in the full joint space, via sbi's ``_run_tarp`` /
    ``check_tarp``.
  - **L-C2ST** -- local classifier two-sample test (Linhart et al. 2023), the only
    per-observation diagnostic: does ``q(theta | x)`` match the true posterior *locally*
    at each ``x``? Via sbi's ``LC2ST``.

**Why pre-drawn arrays, not the posterior + videos.** sbi's ``run_sbc`` / ``run_tarp``
take the observed data ``xs`` as one in-memory tensor and draw internally -- but here
each ``x`` is a full microscopy video (tens of MB), so materializing ~10^4 of them is
infeasible, and sbi's diagnostics assume low-dimensional summary statistics, not raw
videos. Instead the runner streams the EVAL videos once (exactly as the Evaluation
stage does), drawing samples and log-densities per video, and hands this module only
the small results. Correspondingly, the three theta-space tests (SBC, coverage, TARP)
never touch ``x`` at all; L-C2ST -- which must condition on ``x`` -- uses the learned
``Complex3DCNN`` **embedding** (the summary the flow actually conditions on), not the
raw video. Only sbi's pre-drawn-input entry points are used.

**Stratification.** Calibration can hold on average yet fail in a subregion, so every
diagnostic is also reported *stratified*: the EVAL videos are binned into equal-count
bins along one target-theta dimension and the diagnostic is recomputed per bin. The
axis is the posterior's **inferred value** for that dimension (the per-video posterior
median), never the latent ``theta_true`` -- a deliberate and load-bearing choice. The
rank of the truth is uniform *conditional on the observation ``x``*, hence conditional
on any function of ``x`` such as the inferred value, so a calibrated posterior stays
uniform within every inferred-value bin; binning on ``theta_true`` instead would
confound Bayesian shrinkage (correct behavior, and strongest exactly where the data is
uninformative) with miscalibration and flag an honest posterior. The stratifying
dimension is any the caller names -- fully general, since both workflows have a
target-theta vector, with no workflow-specific covariate baked in -- and the question
it answers is "for the videos the posterior *infers* to sit in this region, is it
calibrated there?"

Pure analysis: numpy + torch, with sbi's diagnostics and (for L-C2ST) scikit-learn
imported lazily inside the functions that need them, so the theta-space helpers
(``sbc_ranks``, ``coverage_ranks``, ``coverage_curve``, ``stratify_by_dim``) import and
unit-test without sbi. It imports nothing from ``parameterization`` / ``artifacts``,
so it is testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


# =============================================================================
# Inputs and results
# =============================================================================

@dataclass(frozen=True)
class CalibrationInputs:
    """Pre-drawn diagnostic inputs the runner collects in one streaming pass.

    All arrays span the same ``N`` EVAL videos and live in the active workflow's
    target-theta space (``d = 10`` biology, ``6`` detector). The kernel never sees a
    raw video: the runner draws these while streaming the EVAL set, then drops the
    videos.

    Attributes:
        truths: ``(N, d)`` ground-truth theta (drawn from the prior, so the EVAL set is
            itself a proper SBC sample).
        samples: ``(N, L, d)`` -- ``L`` posterior samples per video.
        theta_keys: the ``d`` target-parameter keys, for labeling and stratification.
        prior_samples: ``(N, d)`` prior draws -- the baseline for ``check_sbc``'s
            data-averaged-posterior check.
        truth_log_probs: ``(N,)`` ``log q(theta_true | x_i)`` -- needed for coverage.
        sample_log_probs: ``(N, L)`` ``log q(theta_l | x_i)`` -- needed for coverage.
        embeddings: ``(N, e)`` learned ``x``-embeddings -- needed only for L-C2ST.
    """

    truths: np.ndarray
    samples: np.ndarray
    theta_keys: Sequence[str]
    prior_samples: np.ndarray
    truth_log_probs: Optional[np.ndarray] = None
    sample_log_probs: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None

    def __post_init__(self):
        truths = np.asarray(self.truths)
        samples = np.asarray(self.samples)
        if truths.ndim != 2:
            raise ValueError(f"truths must be (N, d); got {truths.shape}")
        if samples.ndim != 3:
            raise ValueError(f"samples must be (N, L, d); got {samples.shape}")
        n, d = truths.shape
        if samples.shape[0] != n or samples.shape[2] != d:
            raise ValueError(f"samples {samples.shape} inconsistent with truths {truths.shape}")
        if len(self.theta_keys) != d:
            raise ValueError(f"theta_keys has {len(self.theta_keys)} keys; expected d={d}")
        if np.asarray(self.prior_samples).shape != (n, d):
            raise ValueError(f"prior_samples must be (N, d)=({n}, {d})")
        if self.truth_log_probs is not None and np.asarray(self.truth_log_probs).shape != (n,):
            raise ValueError(f"truth_log_probs must be (N,)=({n},)")
        if self.sample_log_probs is not None and np.asarray(self.sample_log_probs).shape != (n, samples.shape[1]):
            raise ValueError(f"sample_log_probs must be (N, L)=({n}, {samples.shape[1]})")
        if self.embeddings is not None and np.asarray(self.embeddings).shape[0] != n:
            raise ValueError(f"embeddings must have N={n} rows")

    @property
    def n_videos(self) -> int:
        return int(np.asarray(self.truths).shape[0])

    @property
    def num_posterior_samples(self) -> int:
        return int(np.asarray(self.samples).shape[1])

    @property
    def dim(self) -> int:
        return int(np.asarray(self.truths).shape[1])


@dataclass(frozen=True)
class SBCResult:
    """Simulation-based-calibration result (per marginal)."""

    ks_pvals: np.ndarray          # (d,) KS p-value of rank uniformity; small => miscalibrated
    ks_stats: np.ndarray          # (d,) KS D statistic = max |rank CDF - uniform|; the EFFECT SIZE
    rank_hist: np.ndarray         # (d, n_bins) rank histogram per marginal (for plotting)
    rank_hist_edges: np.ndarray   # (n_bins + 1,) shared bin edges in rank units [0, L]
    num_posterior_samples: int
    theta_keys: Sequence[str]
    c2st_ranks: Optional[np.ndarray] = None  # (d,) opt-in C2ST rank-vs-uniform; ~0.5 good, ->1 bad
    c2st_dap: Optional[np.ndarray] = None    # (d,) opt-in C2ST prior-vs-data-averaged-posterior


@dataclass(frozen=True)
class CoverageResult:
    """Expected-coverage result (log-density based; Deistler/Hermans)."""

    levels: np.ndarray            # (M,) nominal credibility levels in [0, 1]
    empirical: np.ndarray         # (M,) empirical coverage at each level
    ks_pval: float                # KS p-value that the log-prob rank is Uniform(0, 1)
    cov_ranks: np.ndarray         # (N,) log-prob coverage rank per video, in [0, L]
    max_gap: float                # max |empirical - nominal| over the levels; the EFFECT SIZE
    max_gap_level: float          # the nominal level where that worst gap occurs


@dataclass(frozen=True)
class TARPResult:
    """TARP result (Lemos et al. 2023)."""

    alpha: np.ndarray             # (K,) credibility levels
    ecp: np.ndarray               # (K,) expected coverage probability
    atc: float                    # area-to-curve; >0 over-dispersed, <0 under-dispersed
    ks_pval: float                # KS p-value that ecp == alpha


@dataclass(frozen=True)
class LC2STResult:
    """Local-C2ST result (Linhart et al. 2023), evaluated at a subset of observations."""

    n_eval: int                   # number of observations evaluated
    statistics: np.ndarray        # (n_eval,) local L-C2ST statistics (0 = indistinguishable)
    p_values: np.ndarray          # (n_eval,) local p-values under the trained null
    alpha: float
    reject_fraction: float        # fraction of observations rejecting calibration at alpha
    median_p_value: float
    eval_index: Optional[np.ndarray] = None   # which videos those per-observation results are for


def lc2st_over_subset(result: "LC2STResult", index) -> Optional["LC2STResult"]:
    """Restrict an L-C2ST result to a subset of videos, without retraining.

    L-C2ST is already a *local* test: it trains once and then yields a p-value **per
    observation**. The calibration of a region is therefore obtained by selecting that
    region's observations from the results already computed -- no separate classifier per
    region. Doing it this way is both far cheaper (one training instead of one per stratum)
    and statistically sounder: a classifier refitted on a few hundred videos would be much
    weaker than the one trained on the whole set, so per-stratum retraining would confound a
    real regional defect with the shrinking power of a smaller training set.

    Returns ``None`` when too few of the evaluated observations fall in the subset for the
    fraction to mean anything.
    """
    if result is None or result.eval_index is None:
        return None
    wanted = np.asarray(index)
    keep = np.isin(np.asarray(result.eval_index), wanted)
    n = int(keep.sum())
    if n < 20:                      # too thin to read a rate from
        return None
    p = np.asarray(result.p_values)[keep]
    return LC2STResult(
        n_eval=n,
        statistics=np.asarray(result.statistics)[keep],
        p_values=p,
        alpha=result.alpha,
        reject_fraction=float(np.mean(p < result.alpha)),
        median_p_value=float(np.median(p)),
        eval_index=np.asarray(result.eval_index)[keep],
    )


@dataclass(frozen=True)
class StratumResult:
    """One diagnostic run restricted to a quantile bin of a target-theta dimension."""

    key: str                      # the target-theta dimension stratified on
    lo: float                     # bin lower edge (theta value)
    hi: float                     # bin upper edge (theta value)
    n: int                        # videos in this bin
    sbc: Optional[SBCResult] = None
    coverage: Optional[CoverageResult] = None
    tarp: Optional[TARPResult] = None
    lc2st: Optional[LC2STResult] = None

    @property
    def label(self) -> str:
        return f"inferred {self.key} in [{self.lo:.4g}, {self.hi:.4g})"


@dataclass(frozen=True)
class CalibrationResult:
    """Overall + stratified calibration for one workflow's posterior."""

    n_videos: int
    num_posterior_samples: int
    theta_keys: Sequence[str]
    tests: Sequence[str]
    overall_sbc: Optional[SBCResult] = None
    overall_coverage: Optional[CoverageResult] = None
    overall_tarp: Optional[TARPResult] = None
    overall_lc2st: Optional[LC2STResult] = None
    strata: Sequence[StratumResult] = field(default_factory=tuple)
    # Coverage bookkeeping: a bin holding fewer than ``min_stratum`` videos is not scored
    # (its rank statistics would be unreliable). Both counts are reported so a truncated
    # stratification announces itself instead of looking like clean full coverage.
    strata_total: int = 0            # bins formed across all stratifying dimensions
    strata_skipped: int = 0          # of those, how many were skipped as too small
    min_stratum: int = 0             # the threshold that was applied
    diagnosis: Sequence[ParameterDiagnosis] = field(default_factory=tuple)
    pairwise: Optional[np.ndarray] = None   # (d, d) |ATC| over 1-D and 2-D marginals


# =============================================================================
# SBC -- Talts et al. 2018 (https://arxiv.org/abs/1804.06788)
# =============================================================================

def sbc_ranks(samples, truths) -> np.ndarray:
    """Rank of each true theta among its ``L`` posterior samples, per marginal.

    Args:
        samples: ``(N, L, d)`` posterior samples.
        truths: ``(N, d)`` ground-truth theta.

    Returns:
        ``(N, d)`` integer ranks in ``[0, L]``:
        ``rank[i, k] = #{ samples[i, :, k] < truths[i, k] }``. Uniform on ``{0..L}``
        for every marginal iff the posterior is calibrated.
    """
    samples = np.asarray(samples)
    truths = np.asarray(truths)
    return (samples < truths[:, None, :]).sum(axis=1)


def run_sbc(inputs: CalibrationInputs, *, n_rank_bins: int = 20, compute_c2st: bool = False) -> SBCResult:
    """SBC ranks + rank-uniformity statistics, per marginal.

    Always runs sbi's Kolmogorov-Smirnov uniformity test
    (``check_uniformity_frequentist``) -- the fast primary check -- alongside the KS **D
    statistic**, the sample-size-independent *effect size* (max deviation of the rank CDF
    from uniform). Report the effect size, not the p-value: at production N the p-value is
    ~0 for any detectable deviation however small, whereas D says how large it is.
    ``compute_c2st=True`` additionally runs the slower classifier two-sample tests --
    rank-vs-uniform and prior-vs-data-averaged-posterior (Talts et al.'s second, more
    sensitive check) -- which train an MLP per marginal, so they are off by default and best
    reserved for a focused follow-up on a flagged parameter.
    """
    from scipy.stats import kstest, uniform  # lazy
    from sbi.diagnostics.sbc import (  # lazy: keeps the theta-space helpers sbi-free
        check_prior_vs_dap, check_uniformity_c2st, check_uniformity_frequentist)

    samples = np.asarray(inputs.samples)
    ranks = sbc_ranks(samples, inputs.truths)                 # (N, d)
    n_samples = inputs.num_posterior_samples
    ranks_t = torch.as_tensor(ranks, dtype=torch.float32)     # the KS/C2ST paths need float ranks
    ks_pvals = check_uniformity_frequentist(ranks_t, n_samples).detach().cpu().numpy()
    # D under the SAME transform sbi's p-value uses (ranks vs Uniform(0, L)), so the two
    # are the statistic and the p-value of one identical test.
    ks_stats = np.array([
        float(kstest(ranks[:, k], uniform(loc=0, scale=n_samples).cdf).statistic)
        for k in range(ranks.shape[1])
    ])

    c2st_ranks = c2st_dap = None
    if compute_c2st:
        c2st_ranks = check_uniformity_c2st(ranks_t, n_samples).detach().cpu().numpy()
        c2st_dap = check_prior_vs_dap(
            torch.as_tensor(np.asarray(inputs.prior_samples), dtype=torch.float32),
            torch.as_tensor(samples[:, 0, :], dtype=torch.float32),   # DAP: one draw per video
        ).detach().cpu().numpy()

    edges = np.linspace(0.0, n_samples, n_rank_bins + 1)
    hist = np.stack([np.histogram(ranks[:, k], bins=edges)[0] for k in range(ranks.shape[1])])
    return SBCResult(
        ks_pvals=ks_pvals,
        ks_stats=ks_stats,
        rank_hist=hist,
        rank_hist_edges=edges,
        num_posterior_samples=n_samples,
        theta_keys=tuple(inputs.theta_keys),
        c2st_ranks=c2st_ranks,
        c2st_dap=c2st_dap,
    )


# =============================================================================
# Expected coverage -- Deistler et al. 2022 / Hermans et al. 2022 (2210.04815)
# =============================================================================

def coverage_ranks(truth_log_probs, sample_log_probs) -> np.ndarray:
    """Rank of the true theta's log-density among the samples' log-densities.

    Args:
        truth_log_probs: ``(N,)`` ``log q(theta_true | x_i)``.
        sample_log_probs: ``(N, L)`` ``log q(theta_l | x_i)``.

    Returns:
        ``(N,)`` integer ranks in ``[0, L]``:
        ``#{ sample_lp[i, :] < truth_lp[i] }``. Uniform on ``[0, L]`` iff the
        posterior's highest-density credible regions have nominal coverage.
    """
    tlp = np.asarray(truth_log_probs)
    slp = np.asarray(sample_log_probs)
    return (slp < tlp[:, None]).sum(axis=1)


def coverage_curve(cov_ranks, num_posterior_samples, levels) -> np.ndarray:
    """Empirical coverage at each nominal credibility level, from the log-prob ranks.

    ``theta_true`` lies inside the ``c``-credible (highest-density) region iff its
    log-prob rank is at least ``(1 - c) * L``, so
    ``empirical_coverage(c) = mean_i[ rank_i / L >= 1 - c ]``. Equal to ``c`` for a
    calibrated posterior; above the diagonal is conservative, below is overconfident.
    """
    r = np.asarray(cov_ranks) / float(num_posterior_samples)
    return np.array([float(np.mean(r >= (1.0 - c))) for c in levels])


def run_coverage(inputs: CalibrationInputs, *, levels: Optional[np.ndarray] = None) -> CoverageResult:
    """Expected-coverage curve + a KS test that the log-prob rank is uniform."""
    if inputs.truth_log_probs is None or inputs.sample_log_probs is None:
        raise ValueError("expected coverage needs truth_log_probs and sample_log_probs")
    from scipy.stats import kstest  # lazy

    n_samples = inputs.num_posterior_samples
    ranks = coverage_ranks(inputs.truth_log_probs, inputs.sample_log_probs)   # (N,)
    if levels is None:
        levels = np.linspace(0.0, 1.0, 21)
    empirical = coverage_curve(ranks, n_samples, levels)
    # Continuity correction maps integer ranks [0, L] to (0, 1) for the uniform KS test.
    ks = kstest((ranks + 0.5) / (n_samples + 1), "uniform")
    # Effect size: the largest gap between empirical and nominal coverage, i.e. "the
    # c-credible interval really covers (c +/- max_gap)". Read this, not the p-value:
    # at production N the p-value is ~0 for any detectable deviation however small.
    levels_arr = np.asarray(levels, dtype=float)
    gaps = np.abs(empirical - levels_arr)
    worst = int(np.argmax(gaps))
    return CoverageResult(
        levels=levels_arr,
        empirical=empirical,
        ks_pval=float(ks.pvalue),
        cov_ranks=ranks,
        max_gap=float(gaps[worst]),
        max_gap_level=float(levels_arr[worst]),
    )


# =============================================================================
# TARP -- Lemos, Coogan et al. 2023 (https://arxiv.org/abs/2302.03026)
# =============================================================================

def run_tarp(inputs: CalibrationInputs, *, num_bins: int = 30, z_score_theta: bool = True) -> TARPResult:
    """TARP expected-coverage curve via sbi's samples-based ``_run_tarp`` + ``check_tarp``."""
    from sbi.diagnostics import check_tarp  # lazy
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references

    truths = torch.as_tensor(np.asarray(inputs.truths), dtype=torch.float32)
    # _run_tarp wants samples-first: (L, N, d).
    samples = torch.as_tensor(np.asarray(inputs.samples), dtype=torch.float32).permute(1, 0, 2)
    references = get_tarp_references(truths)
    ecp, alpha = _run_tarp(samples, truths, references, num_bins=num_bins, z_score_theta=z_score_theta)
    atc, ks_pval = check_tarp(ecp, alpha)
    return TARPResult(
        alpha=alpha.detach().cpu().numpy(),
        ecp=ecp.detach().cpu().numpy(),
        atc=float(atc),
        ks_pval=float(ks_pval),
    )


# =============================================================================
# L-C2ST -- Linhart et al. 2023 (local classifier two-sample test)
# =============================================================================

def run_lc2st(
    inputs: CalibrationInputs,
    *,
    n_eval: int = 1000,
    num_trials_null: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
    classifier: str = "mlp",
    device: str = "cpu",
) -> LC2STResult:
    """Local C2ST on the learned embedding: does ``q(theta | x)`` match locally?

    Trains sbi's ``LC2ST`` to tell posterior draws from prior draws as a function of the
    embedded observation, then evaluates the local statistic / p-value at ``n_eval``
    randomly chosen observations. ``reject_fraction`` is the share of observations whose
    local test rejects calibration at ``alpha``. This is the heaviest diagnostic (it
    fits a main classifier plus ``num_trials_null`` null classifiers), but that training
    cost is independent of ``n_eval`` -- evaluating one observation is ~``num_trials_null``
    cheap forward passes -- so ``n_eval`` is set generously: ``reject_fraction``'s standard
    error is ``sqrt(p (1 - p) / n_eval)``, about 0.016 at ``n_eval = 1000`` versus 0.035 at
    200. Observations are drawn uniformly at random, so the estimate is unbiased.
    """
    if inputs.embeddings is None:
        raise ValueError("L-C2ST needs embeddings (the learned x-summary), not raw videos")
    from sbi.diagnostics import LC2ST  # lazy

    # L-C2ST's reference class is the prior-simulator JOINT (theta_true, x): the ground-truth
    # theta that generated each x -- not independent prior draws. It classifies that joint
    # against the posterior joint (theta ~ q(.|x), x); under calibration they are the same.
    truths = torch.as_tensor(np.asarray(inputs.truths), dtype=torch.float32)
    embeddings = torch.as_tensor(np.asarray(inputs.embeddings), dtype=torch.float32)
    post_one = torch.as_tensor(np.asarray(inputs.samples)[:, 0, :], dtype=torch.float32)  # one draw/video

    # num_folds > 1 makes sbi cross-validate the classifier (train on k-1 folds, score the
    # held-out fold), so per-observation statistics are no longer read off a classifier that
    # memorized the very observation being scored. The default (1) trains a single in-sample
    # classifier, which overstates confidence at evaluated training points.
    lc2st = LC2ST(
        thetas=truths,
        xs=embeddings,
        posterior_samples=post_one,
        seed=seed,
        num_trials_null=num_trials_null,
        num_folds=5,
        classifier=classifier,
        device=device,
    )
    lc2st.train_on_observed_data(verbosity=0)
    lc2st.train_under_null_hypothesis()

    n = embeddings.shape[0]
    n_eval = int(min(n_eval, n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_eval, replace=False)
    statistics = np.empty(n_eval)
    p_values = np.empty(n_eval)
    samples_all = torch.as_tensor(np.asarray(inputs.samples), dtype=torch.float32)
    for j, i in enumerate(idx):
        # Evaluate on the FULL stored posterior cloud for this observation, not a single
        # draw: sbi's statistic is the mean classifier discrepancy over the posterior
        # sample at x_o, and a 1-sample estimate of that mean is needlessly noisy when
        # the L draws are already on disk.
        theta_o = samples_all[i]
        x_o = embeddings[i:i + 1]
        statistics[j] = float(lc2st.get_statistic_on_observed_data(theta_o, x_o))
        p_values[j] = float(lc2st.p_value(theta_o, x_o))
    return LC2STResult(
        n_eval=n_eval,
        statistics=statistics,
        p_values=p_values,
        alpha=alpha,
        reject_fraction=float(np.mean(p_values < alpha)),
        median_p_value=float(np.median(p_values)),
        eval_index=np.asarray(idx),
    )


# =============================================================================
# Diagnosis -- separating a location error from a width error
# =============================================================================

@dataclass(frozen=True)
class ParameterDiagnosis:
    """Where a parameter's calibration error comes from: its location or its width.

    The rank-based measures report *that* a posterior is miscalibrated; they do not say
    whether it sits in the wrong place or merely claims the wrong precision. Those are
    different defects with different remedies, and the standardized error
    ``z = (theta_true - posterior_median) / posterior_sd`` separates them: for a calibrated
    posterior ``z`` has mean 0 and standard deviation 1, so its first two moments read
    directly as a location error and a width error.
    """

    key: str
    bias_z: float          # mean(z): systematic offset in units of the posterior's own sd
    spread_z: float        # sd(z): actual error / claimed sd. 1 = honest, >1 = too narrow
    sharpness: float       # posterior sd / prior width: how much the data constrained it
    bias_theta: float      # the same offset in theta units (log10 here)
    bias_factor: float     # and as a multiplicative factor on the physical parameter
    defect: str            # "ok" | "location (high|low)" | "width (too narrow|too wide)"


# A location error counts once it is an appreciable fraction of the posterior's own width;
# a width error once the claimed interval is off by more than about 15%. Both bars are
# practical, sample-size independent, and identical for every parameter and workflow.
_BIAS_Z_LIMIT = 0.30
_SPREAD_Z_LOW, _SPREAD_Z_HIGH = 0.87, 1.15


def diagnose(inputs: CalibrationInputs) -> list:
    """Per-parameter location/width diagnosis from the standardized error.

    Returns one :class:`ParameterDiagnosis` per target-theta dimension. ``sharpness`` uses
    the prior width recovered from the ground-truth values themselves: the held-out theta
    are prior draws, and for the box-uniform priors both workflows use, ``width =
    sqrt(12) * sd``, so no prior object has to be threaded in.
    """
    truths = np.asarray(inputs.truths)
    samples = np.asarray(inputs.samples)
    median = np.median(samples, axis=1)                     # (N, d)
    post_sd = samples.std(axis=1)                           # (N, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(post_sd > 0, (truths - median) / post_sd, np.nan)
    prior_width = np.sqrt(12.0) * truths.std(axis=0)        # exact for a uniform prior

    out = []
    for j, key in enumerate(inputs.theta_keys):
        zj = z[:, j][np.isfinite(z[:, j])]
        bias_z = float(np.mean(zj)) if zj.size else float("nan")
        spread_z = float(np.std(zj)) if zj.size else float("nan")
        mean_sd = float(np.mean(post_sd[:, j]))
        # Report the offset with the posterior's sign convention: positive = the posterior
        # sits ABOVE the truth, i.e. the parameter is over-estimated.
        bias_theta = -bias_z * mean_sd
        out.append(ParameterDiagnosis(
            key=str(key),
            bias_z=-bias_z,
            spread_z=spread_z,
            sharpness=(mean_sd / prior_width[j]) if prior_width[j] > 0 else float("nan"),
            bias_theta=bias_theta,
            bias_factor=float(10.0 ** abs(bias_theta)),
            defect=_defect_label(-bias_z, spread_z),
        ))
    return out


def _defect_label(bias_z: float, spread_z: float) -> str:
    """Name the dominant defect, comparing each error against its own practical bar."""
    if not (np.isfinite(bias_z) and np.isfinite(spread_z)):
        return "undetermined"
    bias_excess = abs(bias_z) / _BIAS_Z_LIMIT
    width_excess = max(spread_z / _SPREAD_Z_HIGH, _SPREAD_Z_LOW / max(spread_z, 1e-9))
    if bias_excess <= 1.0 and width_excess <= 1.0:
        return "ok"
    if bias_excess >= width_excess:
        return f"location ({'high' if bias_z > 0 else 'low'})"
    return f"width ({'too narrow' if spread_z > 1 else 'too wide'})"


# =============================================================================
# Pairwise (two-dimensional) calibration
# =============================================================================

def pairwise_calibration(inputs: CalibrationInputs, *, num_bins: int = 50,
                         n_jobs: Optional[int] = None, max_videos: int = 4000,
                         seed: int = 0) -> np.ndarray:
    """Calibration of every one- and two-dimensional marginal, as a ``(d, d)`` matrix.

    Rank uniformity of the one-dimensional marginals is *necessary but not sufficient* for
    a calibrated posterior (Talts et al. 2018; Modrak et al. 2023): a posterior can have
    every marginal perfectly calibrated and still misstate how the parameters covary, and
    such a defect is invisible to marginal SBC. Checking the two-dimensional marginals as
    well raises the bar -- it catches pairwise dependence errors -- without yet being
    sufficient either; only the full joint test (TARP over all dimensions) is, in the
    population limit. Reporting both levels therefore says how far up that ladder the
    posterior has been verified.

    Each entry is an effect size on the same 0-scale as the rest of the report:

      - off-diagonal ``[i, j]`` -- ``|ATC|`` of the TARP test restricted to the
        two-dimensional subspace spanned by parameters ``i`` and ``j``.
      - diagonal ``[i, i]`` -- ``|ATC|`` of the same test on parameter ``i`` alone, so the
        diagonal and the off-diagonal are directly comparable.

    Args:
        num_bins: credibility bins for each subspace TARP curve.
        n_jobs: worker processes (see :func:`resolve_workers`).
        max_videos: cap on the videos used per subspace. ``ATC`` is an average over videos,
            so its precision goes as ``1/sqrt(n)`` and a few thousand already resolve it far
            below the practical threshold; capping keeps the ``d (d + 1) / 2`` subspace fits
            affordable and their payloads small. ``0`` uses every video.
        seed: fixes the subsample, so the matrix is reproducible.

    Returns:
        ``(d, d)`` symmetric array of ``|ATC|``; larger means worse.
    """
    d = inputs.dim
    truths = np.asarray(inputs.truths)
    samples = np.asarray(inputs.samples)
    if 0 < max_videos < truths.shape[0]:
        pick = np.random.default_rng(seed).choice(truths.shape[0], max_videos, replace=False)
        truths, samples = truths[pick], samples[pick]

    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    payloads = [(truths[:, sorted({i, j})], samples[:, :, sorted({i, j})], num_bins)
                for i, j in pairs]
    values = _parallel_map(_subspace_atc, payloads, resolve_workers(n_jobs))

    matrix = np.zeros((d, d), dtype=float)
    for (i, j), v in zip(pairs, values):
        matrix[i, j] = matrix[j, i] = v
    return matrix


def _subspace_atc(payload) -> float:
    """``|ATC|`` of the TARP coverage test restricted to one subspace of theta."""
    truths_sub, samples_sub, num_bins = payload
    from sbi.diagnostics import check_tarp
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references

    truths_t = torch.as_tensor(truths_sub, dtype=torch.float32)
    samples_t = torch.as_tensor(samples_sub, dtype=torch.float32).permute(1, 0, 2)
    ecp, alpha = _run_tarp(samples_t, truths_t, get_tarp_references(truths_t),
                           num_bins=num_bins, z_score_theta=True)
    atc, _ = check_tarp(ecp, alpha)
    return float(abs(atc))


# =============================================================================
# Stratification -- general over any target-theta dimension
# =============================================================================

def stratify_by_dim(values, dim, *, n_strata: int = 5):
    """Quantile-bin the videos by the inferred value of one target-theta dimension.

    Args:
        values: ``(N, d)`` per-video inferred point estimate (the posterior median) --
            a function of the observation ``x``. Binning on this axis, not the latent
            ``theta_true``, is what keeps a calibrated posterior's within-bin ranks
            uniform: the rank is uniform conditional on ``x`` and hence on any function
            of ``x``, whereas conditioning on ``theta_true`` confounds Bayesian shrinkage
            with miscalibration.
        dim: the target-theta dimension index to stratify on.
        n_strata: number of equal-count bins.

    Returns:
        A list of ``(lo, hi, indices)`` for each non-empty bin along ``values[:, dim]``.
        Bins are equal-count quantiles; when the estimate is effectively discrete
        (repeated quantile edges collapse) fewer, wider bins are returned rather than
        empty ones. The stratifying axis is fully general -- the caller chooses which
        target-theta dimension -- so no workflow-specific covariate is assumed.
    """
    col = np.asarray(values)[:, dim]
    edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_strata + 1)))
    if edges.size < 2:  # a constant column -- a single degenerate bin
        return [(float(col.min()), float(np.nextafter(col.max(), np.inf)), np.arange(col.size))]
    edges = edges.astype(float)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max in the last bin
    out = []
    for b in range(edges.size - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        index = np.where((col >= lo) & (col < hi))[0]
        if index.size:
            out.append((lo, hi, index))
    return out


def _subset(inputs: CalibrationInputs, index: np.ndarray) -> CalibrationInputs:
    """Restrict every input array to ``index`` (a view onto the same theta space)."""
    def take(a):
        return None if a is None else np.asarray(a)[index]

    return CalibrationInputs(
        truths=np.asarray(inputs.truths)[index],
        samples=np.asarray(inputs.samples)[index],
        theta_keys=inputs.theta_keys,
        prior_samples=np.asarray(inputs.prior_samples)[index],
        truth_log_probs=take(inputs.truth_log_probs),
        sample_log_probs=take(inputs.sample_log_probs),
        embeddings=take(inputs.embeddings),
    )


# =============================================================================
# Orchestration
# =============================================================================

_ALL_TESTS = ("sbc", "coverage", "tarp", "lc2st")

# Cap on the stratum-loop worker pool. The per-stratum work is dominated by L-C2ST, which
# fits one classifier plus ``num_trials_null`` null classifiers per stratum, so the loop is
# embarrassingly parallel and CPU-bound; 16 is enough to make stratification cheap without
# monopolizing a shared node.
_MAX_WORKERS = 16


def resolve_workers(requested: Optional[int] = None, *, cap: int = _MAX_WORKERS) -> int:
    """Resolve the stratum-loop worker count: the largest power of two the job may use.

    Steps down ``16 -> 8 -> 4 -> 2 -> 1`` so the pool always divides evenly into the cores
    actually available. Reads the cores **allocated to this process**
    (``os.sched_getaffinity``, which honors a Slurm cgroup / ``--cpus-per-task``) rather than
    the machine's total, so a job given 64 cores on a 288-core node does not oversubscribe.

    Args:
        requested: explicit override; ``None`` or a non-positive value means auto-detect.
            An explicit value is honored as given (not snapped to a power of two), so a
            caller can pin an exact pool size.
        cap: never return more than this.

    Returns:
        The worker count (>= 1). ``1`` means run the loop serially.
    """
    import os

    if requested is not None and requested > 0:
        return int(requested)
    try:
        avail = len(os.sched_getaffinity(0))       # Linux: respects the Slurm allocation
    except AttributeError:                        # pragma: no cover - non-Linux fallback
        avail = os.cpu_count() or 1
    n = max(1, min(cap, avail))
    return 1 << (n.bit_length() - 1)              # snap down: 12 -> 8, 7 -> 4, 3 -> 2


def _parallel_map(worker, payloads, workers: int):
    """Map ``worker`` over ``payloads``, in parallel when that is safe, else serially.

    Uses a **spawn** context rather than the platform default. On Linux that default is
    ``fork``, which duplicates the parent mid-flight -- including the thread pools torch and
    the BLAS libraries have already started. A forked child that then re-enters those
    libraries can deadlock, and does so here: the workers sit at zero CPU forever while the
    parent waits on them. Spawn starts each worker from a clean interpreter, paying a
    re-import per worker (seconds) to remove that class of hang entirely.

    Order is preserved, and any pool failure degrades to a serial run rather than losing the
    result.

    One caller obligation comes with spawn: every worker re-imports the entry-point module,
    so a caller that runs :func:`calibrate` from module level -- without an
    ``if __name__ == "__main__":`` guard -- has its whole body re-executed once per worker,
    each re-execution spawning workers of its own. The two shipped entry points guard their
    bodies; an ad-hoc script that calls this must do the same.
    """
    if workers <= 1 or len(payloads) <= 1:
        return [worker(p) for p in payloads]
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads)),
                                 mp_context=mp.get_context("spawn")) as pool:
            return list(pool.map(worker, payloads))
    except Exception:
        return [worker(p) for p in payloads]


def _stratum_worker(payload):
    """Score one stratum. Module-level and picklable, so a process pool can run it.

    Pins each worker's BLAS/OpenMP pool to a single thread while it runs: the strata are
    already spread across processes, so letting each one also spawn a full thread pool
    would oversubscribe the node and run *slower* than serial. ``threadpoolctl`` ships with
    scikit-learn (which L-C2ST uses); if it is somehow absent the work still runs, just
    without the pinning.
    """
    sub_inputs, tests, num_bins, levels, lc2st_kwargs, sbc_c2st = payload
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                            # pragma: no cover - optional guard
        return _run_selected(sub_inputs, tests, num_bins=num_bins, levels=levels,
                             lc2st_kwargs=lc2st_kwargs, sbc_c2st=sbc_c2st)
    with threadpool_limits(limits=1):
        return _run_selected(sub_inputs, tests, num_bins=num_bins, levels=levels,
                             lc2st_kwargs=lc2st_kwargs, sbc_c2st=sbc_c2st)


def _run_selected(inputs, tests, *, num_bins, levels, lc2st_kwargs, sbc_c2st):
    """Run the requested diagnostics on one (sub)set; return a dict of results."""
    out = {}
    if "sbc" in tests:
        out["sbc"] = run_sbc(inputs, compute_c2st=sbc_c2st)
    if "coverage" in tests:
        out["coverage"] = run_coverage(inputs, levels=levels)
    if "tarp" in tests:
        out["tarp"] = run_tarp(inputs, num_bins=num_bins)
    if "lc2st" in tests:
        out["lc2st"] = run_lc2st(inputs, **(lc2st_kwargs or {}))
    return out


def calibrate(
    inputs: CalibrationInputs,
    *,
    tests: Sequence[str] = _ALL_TESTS,
    stratify_dims: Optional[Sequence[int]] = None,
    n_strata: int = 10,
    num_bins: int = 50,
    levels: Optional[np.ndarray] = None,
    lc2st_kwargs: Optional[dict] = None,
    sbc_c2st: bool = False,
    min_stratum: int = 200,
    n_jobs: Optional[int] = None,
) -> CalibrationResult:
    """Run the requested diagnostics overall and stratified by target-theta dimension.

    Args:
        inputs: pre-drawn diagnostic inputs in the active workflow's theta space.
        tests: any subset of ``("sbc", "coverage", "tarp", "lc2st")``.
        stratify_dims: target-theta dimension indices to stratify by. ``None`` = every
            dimension; ``()`` = overall only.
        n_strata: equal-count bins per stratifying dimension. The default 10 is chosen for
            resolution in log10 space: with 5 bins a parameter whose prior spans 2.5 decades
            (a species count, say) puts a factor of ~3 inside one bin, blurring together
            regimes with quite different identifiability.
        num_bins: TARP credibility bins. This is resolution along the *credibility* axis,
            not a data split -- every video informs every level -- so the default 50 costs
            nothing and simply samples the ECP curve more finely.
        levels: nominal coverage levels (default ``linspace(0, 1, 21)``).
        lc2st_kwargs: forwarded to :func:`run_lc2st`.
        sbc_c2st: also run SBC's slower classifier two-sample tests (off by default).
        min_stratum: skip a bin with fewer than this many videos -- the rank/coverage
            statistics are unreliable on tiny bins. Skipped bins are counted and reported
            (``strata_skipped``), never dropped silently.
        n_jobs: worker processes for the stratum loop. ``None`` auto-detects via
            :func:`resolve_workers` (largest power of two up to 16 that fits the cores
            allocated to this process); ``1`` runs serially. The loop is CPU-bound and
            dominated by L-C2ST's per-stratum classifier fits, so this is what keeps a
            10-bin stratification cheap.

    Returns:
        A :class:`CalibrationResult` with the overall diagnostics, one
        :class:`StratumResult` per scored bin, and the skipped-bin bookkeeping.
    """
    unknown = set(tests) - set(_ALL_TESTS)
    if unknown:
        raise ValueError(f"unknown test(s): {sorted(unknown)}; choose from {_ALL_TESTS}")

    overall = _run_selected(inputs, tests, num_bins=num_bins, levels=levels,
                            lc2st_kwargs=lc2st_kwargs, sbc_c2st=sbc_c2st)

    # Location/width diagnosis (cheap) and the 1-D / 2-D marginal ladder. The pairwise
    # matrix reuses TARP, so it is computed whenever TARP is among the requested tests.
    diagnosis = diagnose(inputs)
    pairwise = (pairwise_calibration(inputs, num_bins=num_bins, n_jobs=n_jobs)
                if "tarp" in tests else None)

    # Stratify by the posterior's INFERRED value per video (a function of the observation),
    # never the latent theta_true. The rank is uniform conditional on x, so an x-derived
    # axis preserves that uniformity within each bin; binning on theta_true instead would
    # confound Bayesian shrinkage (correct behavior, strongest where the data is
    # uninformative) with miscalibration, flagging even a perfectly calibrated posterior.
    inferred = np.median(np.asarray(inputs.samples), axis=1)   # (N, d) per-video posterior median
    if stratify_dims is None:
        stratify_dims = range(inputs.dim)

    # ---- Enumerate the bins first, so the skipped ones can be counted ----
    jobs, n_total, n_skipped = [], 0, 0
    for dim in stratify_dims:
        key = inputs.theta_keys[dim]
        for lo, hi, index in stratify_by_dim(inferred, dim, n_strata=n_strata):
            n_total += 1
            if index.size < min_stratum:
                n_skipped += 1
                continue
            jobs.append((key, lo, hi, index))

    # ---- Score each surviving bin (in parallel; the loop is CPU-bound) ----
    # L-C2ST is deliberately excluded from the per-stratum work and instead restricted from
    # the overall result (see :func:`lc2st_over_subset`): it is already a per-observation
    # test, so a region's rate follows from the p-values already computed. Retraining it per
    # stratum would cost one classifier ensemble per bin -- the dominant expense of the whole
    # diagnostic -- and would weaken the classifier at the same time.
    stratum_tests = tuple(t for t in tests if t != "lc2st")
    payloads = [(_subset(inputs, index), stratum_tests, num_bins, levels, None, sbc_c2st)
                for _, _, _, index in jobs]
    results = _parallel_map(_stratum_worker, payloads, resolve_workers(n_jobs))

    strata = [
        StratumResult(key=key, lo=lo, hi=hi, n=int(index.size),
                      sbc=sub.get("sbc"), coverage=sub.get("coverage"),
                      tarp=sub.get("tarp"),
                      lc2st=(lc2st_over_subset(overall.get("lc2st"), index)
                             if "lc2st" in tests else None))
        for (key, lo, hi, index), sub in zip(jobs, results)
    ]

    return CalibrationResult(
        n_videos=inputs.n_videos,
        num_posterior_samples=inputs.num_posterior_samples,
        theta_keys=tuple(inputs.theta_keys),
        tests=tuple(tests),
        overall_sbc=overall.get("sbc"),
        overall_coverage=overall.get("coverage"),
        overall_tarp=overall.get("tarp"),
        overall_lc2st=overall.get("lc2st"),
        strata=tuple(strata),
        strata_total=n_total,
        strata_skipped=n_skipped,
        min_stratum=int(min_stratum),
        diagnosis=tuple(diagnosis),
        pairwise=pairwise,
    )


# =============================================================================
# Text summary (the CLI renders figures; this is the printable digest)
# =============================================================================

# Practical thresholds on the EFFECT SIZES -- "is the deviation large enough to matter?",
# deliberately not "is it statistically detectable?". They are sample-size independent, so
# the same bar applies to the overall result and to every stratum. All three calibration
# measures share 0.05: a five-percentage-point deviation (of the rank CDF from uniform, of
# empirical from nominal coverage, or of the TARP curve from the diagonal). L-C2ST uses
# twice its significance level, the rate a calibrated posterior would reject at by chance.
EFFECT_THRESHOLDS = {
    "sbc": 0.05,        # KS D: max deviation of the rank CDF from uniform
    "coverage": 0.05,   # max |empirical - nominal| coverage gap
    "tarp": 0.05,       # |ATC|: area between the ECP curve and the diagonal
    "lc2st": 0.10,      # reject fraction (= 2 * alpha at the default alpha = 0.05)
}

def summarize(result: CalibrationResult) -> list[dict]:
    """Flatten a :class:`CalibrationResult` into printable rows.

    Each row is ``{scope, test, statistic, value, flag}``. Every reported value is an
    **effect size**, never a p-value, and ``flag`` ("ok" / "check") compares it against
    :data:`EFFECT_THRESHOLDS` -- a practical bar ("is the deviation big enough to matter?")
    rather than a statistical one ("is it detectable?"). The distinction is decisive at
    production scale: with 10^4 videos a KS p-value is ~0 for any deviation whatsoever, so
    p-value verdicts flag everything and rank nothing. The figures (rank histograms,
    coverage / ECP curves) remain the real read; these numbers order the parameters.
    """
    rows: list[dict] = []

    def add(scope, test, statistic, value, flag):
        rows.append({"scope": scope, "test": test, "statistic": statistic,
                     "value": value, "flag": flag})

    def emit(scope, sbc, cov, tarp, lc2st):
        if sbc is not None:
            d_stats = np.asarray(sbc.ks_stats)
            thr = EFFECT_THRESHOLDS["sbc"]
            worst_i = int(np.argmax(d_stats))
            worst = float(d_stats[worst_i])
            n_over = int(np.sum(d_stats > thr))
            key = sbc.theta_keys[worst_i] if worst_i < len(sbc.theta_keys) else "?"
            add(scope, "sbc", f"max KS D [{key}] ({n_over}/{d_stats.size} >{thr:g})", worst,
                "check" if worst > thr else "ok")
        if cov is not None:
            thr = EFFECT_THRESHOLDS["coverage"]
            add(scope, "coverage", f"max |empirical-nominal| @ {cov.max_gap_level:.2f}",
                cov.max_gap, "check" if cov.max_gap > thr else "ok")
        if tarp is not None:
            thr = EFFECT_THRESHOLDS["tarp"]
            add(scope, "tarp", "ATC (0 ideal)", tarp.atc,
                "check" if abs(tarp.atc) > thr else "ok")
        if lc2st is not None:
            add(scope, "lc2st", "reject fraction", lc2st.reject_fraction,
                "check" if lc2st.reject_fraction > EFFECT_THRESHOLDS["lc2st"] else "ok")

    emit("overall", result.overall_sbc, result.overall_coverage,
         result.overall_tarp, result.overall_lc2st)
    for s in result.strata:
        emit(s.label, s.sbc, s.coverage, s.tarp, s.lc2st)
    return rows


def stratum_effect(stratum: StratumResult, test: str,
                   dim: Optional[int] = None) -> Optional[float]:
    """The effect size one ``test`` contributes for one stratum, or None if not run.

    Single definition shared by the stratified figures and the stratified digest, so a
    plotted bar and its tabulated summary can never disagree about what was measured.

    For SBC the value is the STRATIFYING parameter's OWN rank statistic (``dim``), not the
    worst across all marginals: the question a stratum answers is "is this parameter
    calibrated across its own range?", and a max over marginals would report whichever
    parameter happens to be worst overall in every panel alike, saying nothing about the
    one named. ``dim=None`` falls back to that max, for callers with no dimension in hand.
    The remaining measures are joint by construction and have no per-parameter form.
    """
    if test == "sbc" and stratum.sbc is not None:
        stats = np.asarray(stratum.sbc.ks_stats)
        if dim is not None and dim < stats.size:
            return float(stats[dim])
        return float(np.max(stats))
    if test == "coverage" and stratum.coverage is not None:
        return float(stratum.coverage.max_gap)      # max |empirical - nominal|
    if test == "tarp" and stratum.tarp is not None:
        return abs(float(stratum.tarp.atc))
    if test == "lc2st" and stratum.lc2st is not None:
        return float(stratum.lc2st.reject_fraction)
    return None


# A monotone run of bins is read as a trend only above this |Spearman rho|; below it, the
# profile is instead tested for the two symmetric shapes that matter (worst at the prior
# edges versus worst in the interior), which a correlation cannot express.
_TREND_RHO = 0.60
_TREND_RATIO = 1.25     # edge-versus-interior mean ratio required to call a shape


def _spearman_vs_index(values: np.ndarray) -> float:
    """Spearman rho between a bin's ordinal position and its statistic (ties averaged)."""
    n = values.size
    if n < 3:
        return 0.0
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # Average the ranks of tied values, so a flat profile scores 0 rather than an
    # artefact of the sort order.
    uniq, inv = np.unique(values, return_inverse=True)
    if uniq.size < n:
        for u in range(uniq.size):
            mask = inv == u
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
    idx = np.arange(n, dtype=float)
    rs, ri = ranks.std(), idx.std()
    if rs == 0 or ri == 0:
        return 0.0
    return float(((ranks - ranks.mean()) * (idx - idx.mean())).mean() / (rs * ri))


def _trend_label(values: np.ndarray) -> str:
    """Describe how a statistic varies across one parameter's ordered bins."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return "-"
    rho = _spearman_vs_index(v)
    if rho >= _TREND_RHO:
        return "rises with the inferred value"
    if rho <= -_TREND_RHO:
        return "falls with the inferred value"
    # Not monotone: distinguish a U (worst where the prior is thinly sampled at both ends)
    # from an inverted U (worst in the interior). Both are real and mean opposite things.
    k = max(1, v.size // 4)
    edges = np.concatenate([v[:k], v[-k:]]).mean()
    interior = v[k:-k].mean() if v.size > 2 * k else v.mean()
    if interior > 0 and edges > interior * _TREND_RATIO:
        return "worst at both ends of the range"
    if edges > 0 and interior > edges * _TREND_RATIO:
        return "worst in the middle of the range"
    return "flat, no clear trend"


def stratified_digest(result: CalibrationResult) -> list[dict]:
    """Collapse the per-bin stratified results to one row per (parameter, test).

    The per-bin numbers are a poor table: one row per bin per test per parameter runs to
    hundreds of rows that no reader can hold, and the shape they encode -- whether a
    defect grows across a parameter's range, or concentrates at its ends -- is exactly
    what a long column of numbers hides. The stratified FIGURES carry the per-bin detail
    (and the saved ``.npz`` carries the numbers themselves); this digest carries what the
    figures are for: how many bins flag, which bin is worst, and the shape of the profile.

    Each row is ``{key, test, n_bins, n_flagged, worst, worst_center, trend, flag}``, in
    the stratifying order of ``result.strata``.
    """
    rows: list[dict] = []
    keys: list[str] = []
    for s in result.strata:
        if s.key not in keys:
            keys.append(s.key)
    theta_keys = list(result.theta_keys)

    for key in keys:
        bins = [s for s in result.strata if s.key == key]
        dim = theta_keys.index(key) if key in theta_keys else None
        for test in ("sbc", "coverage", "tarp", "lc2st"):
            vals, centers = [], []
            for s in bins:
                v = stratum_effect(s, test, dim)
                if v is None:
                    continue
                vals.append(v)
                centers.append(0.5 * (s.lo + s.hi))
            if not vals:
                continue
            v = np.asarray(vals, dtype=float)
            thr = EFFECT_THRESHOLDS[test]
            worst_i = int(np.argmax(v))
            rows.append({
                "key": key,
                "test": test,
                "n_bins": int(v.size),
                "n_flagged": int(np.sum(v > thr)),
                "worst": float(v[worst_i]),
                "worst_center": float(centers[worst_i]),
                "trend": _trend_label(v),
                "flag": "check" if float(v[worst_i]) > thr else "ok",
            })
    return rows
