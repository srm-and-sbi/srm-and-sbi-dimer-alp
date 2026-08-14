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
    (``check_uniformity_frequentist``) -- the fast primary check. ``compute_c2st=True``
    additionally runs the slower classifier two-sample tests -- rank-vs-uniform and
    prior-vs-data-averaged-posterior (Talts et al.'s second, more sensitive check) --
    which train an MLP per marginal, so they are off by default and best reserved for a
    focused follow-up on a flagged parameter.
    """
    from sbi.diagnostics.sbc import (  # lazy: keeps the theta-space helpers sbi-free
        check_prior_vs_dap, check_uniformity_c2st, check_uniformity_frequentist)

    samples = np.asarray(inputs.samples)
    ranks = sbc_ranks(samples, inputs.truths)                 # (N, d)
    n_samples = inputs.num_posterior_samples
    ranks_t = torch.as_tensor(ranks, dtype=torch.float32)     # the KS/C2ST paths need float ranks
    ks_pvals = check_uniformity_frequentist(ranks_t, n_samples).detach().cpu().numpy()

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
    return CoverageResult(
        levels=np.asarray(levels, dtype=float),
        empirical=empirical,
        ks_pval=float(ks.pvalue),
        cov_ranks=ranks,
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
    n_eval: int = 200,
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
    fits a main classifier plus ``num_trials_null`` null classifiers), so it evaluates a
    subset by default rather than all ``N``.
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

    lc2st = LC2ST(
        thetas=truths,
        xs=embeddings,
        posterior_samples=post_one,
        seed=seed,
        num_trials_null=num_trials_null,
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
    for j, i in enumerate(idx):
        theta_o = post_one[i:i + 1]
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
    )


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
    n_strata: int = 5,
    num_bins: int = 30,
    levels: Optional[np.ndarray] = None,
    lc2st_kwargs: Optional[dict] = None,
    sbc_c2st: bool = False,
    min_stratum: int = 200,
) -> CalibrationResult:
    """Run the requested diagnostics overall and stratified by target-theta dimension.

    Args:
        inputs: pre-drawn diagnostic inputs in the active workflow's theta space.
        tests: any subset of ``("sbc", "coverage", "tarp", "lc2st")``.
        stratify_dims: target-theta dimension indices to stratify by. ``None`` = every
            dimension; ``()`` = overall only.
        n_strata: equal-count bins per stratifying dimension.
        num_bins: TARP credibility bins.
        levels: nominal coverage levels (default ``linspace(0, 1, 21)``).
        lc2st_kwargs: forwarded to :func:`run_lc2st`.
        sbc_c2st: also run SBC's slower classifier two-sample tests (off by default).
        min_stratum: skip a bin with fewer than this many videos -- the rank/coverage
            statistics are unreliable on tiny bins.

    Returns:
        A :class:`CalibrationResult` with the overall diagnostics and one
        :class:`StratumResult` per non-trivial bin.
    """
    unknown = set(tests) - set(_ALL_TESTS)
    if unknown:
        raise ValueError(f"unknown test(s): {sorted(unknown)}; choose from {_ALL_TESTS}")

    overall = _run_selected(inputs, tests, num_bins=num_bins, levels=levels,
                            lc2st_kwargs=lc2st_kwargs, sbc_c2st=sbc_c2st)

    # Stratify by the posterior's INFERRED value per video (a function of the observation),
    # never the latent theta_true. The rank is uniform conditional on x, so an x-derived
    # axis preserves that uniformity within each bin; binning on theta_true instead would
    # confound Bayesian shrinkage (correct behavior, strongest where the data is
    # uninformative) with miscalibration, flagging even a perfectly calibrated posterior.
    inferred = np.median(np.asarray(inputs.samples), axis=1)   # (N, d) per-video posterior median
    if stratify_dims is None:
        stratify_dims = range(inputs.dim)
    strata = []
    for dim in stratify_dims:
        key = inputs.theta_keys[dim]
        for lo, hi, index in stratify_by_dim(inferred, dim, n_strata=n_strata):
            if index.size < min_stratum:
                continue
            sub = _run_selected(
                _subset(inputs, index), tests, num_bins=num_bins, levels=levels,
                lc2st_kwargs=lc2st_kwargs, sbc_c2st=sbc_c2st,
            )
            strata.append(StratumResult(
                key=key, lo=lo, hi=hi, n=int(index.size),
                sbc=sub.get("sbc"), coverage=sub.get("coverage"),
                tarp=sub.get("tarp"), lc2st=sub.get("lc2st"),
            ))

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
    )


# =============================================================================
# Text summary (the CLI renders figures; this is the printable digest)
# =============================================================================

def summarize(result: CalibrationResult) -> list[dict]:
    """Flatten a :class:`CalibrationResult` into printable rows.

    Each row is ``{scope, test, statistic, value, flag}`` where ``flag`` is a coarse
    verdict ("ok" / "check") from the standard thresholds: SBC/coverage uniformity
    ``ks_pval < 0.05``; TARP ``|atc| > 0.05``; L-C2ST ``reject_fraction > 2*alpha``.
    Thresholds are advisory -- the figures (rank histograms, coverage/ECP curves) are
    the real read.
    """
    rows: list[dict] = []

    def add(scope, test, statistic, value, flag):
        rows.append({"scope": scope, "test": test, "statistic": statistic,
                     "value": value, "flag": flag})

    def emit(scope, sbc, cov, tarp, lc2st):
        if sbc is not None:
            ks = np.asarray(sbc.ks_pvals)
            d = ks.size
            worst = float(ks.min())
            n_fail = int(np.sum(ks < 0.05))
            # Bonferroni over the d marginals: a single MC-low p-value on a well-calibrated
            # posterior is expected, so only flag past the corrected threshold.
            add(scope, "sbc", f"min KS p (marginals; {n_fail}/{d} <0.05)", worst,
                "check" if worst < 0.05 / d else "ok")
        if cov is not None:
            add(scope, "coverage", "KS p (log-prob rank)", cov.ks_pval,
                "check" if cov.ks_pval < 0.05 else "ok")
        if tarp is not None:
            add(scope, "tarp", "ATC (0 ideal)", tarp.atc,
                "check" if abs(tarp.atc) > 0.05 else "ok")
        if lc2st is not None:
            add(scope, "lc2st", "reject fraction", lc2st.reject_fraction,
                "check" if lc2st.reject_fraction > 2 * lc2st.alpha else "ok")

    emit("overall", result.overall_sbc, result.overall_coverage,
         result.overall_tarp, result.overall_lc2st)
    for s in result.strata:
        emit(s.label, s.sbc, s.coverage, s.tarp, s.lc2st)
    return rows
