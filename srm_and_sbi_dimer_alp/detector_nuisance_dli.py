"""Nuisance_DLI: the calibrated-imaging nuisance — artifact, construction, and gate.

Design reference: DETECTOR_WORKFLOW.md, "Nuisance and artifact design".

The Detector calibrates the imaging model; the imaging block is then marginalized during
the production run by drawing from a `Nuisance_DLI`. Unlike the biology nuisance, a
`Nuisance_DLI` cannot be declared a priori — its content *is* the calibration result — so
it is built once, by a user-driven analysis step, and persisted as a self-contained
artifact that generation samples standalone.

Construction. The detector estimator is a REQUIRED prerequisite. The analysis step reads
all experimental recordings, draws the posterior per model-window chunk on the fly, and
pools the draws across every chunk of every recording into the `posterior_sample_pool` — a
mixture that represents how the calibrated imaging varies across the real acquisitions. One
user choice, `posterior_sample_pool_choice`, turns that pool into the samplable artifact:

  - ``raw``               — resample the pool per whole vector (rows). Preserves the joint
                            correlation structure exactly; the faithful default.
  - ``map_estimate_pool`` — resample, per whole vector, the pooled per-chunk MAP estimates
                            (point estimates rather than full posterior draws).
  - ``gaussian``          — a full-covariance multivariate normal fit to the pool. Keeps the
                            linear correlations (via the covariance); loses multimodality
                            and curvature.
  - ``box``               — a per-parameter uniform over quantiles of the pool, clamped to
                            the imaging prior box. Independent per dimension: no correlation.
  - ``box_user``          — a per-parameter uniform over user-set ranges, clamped to the
                            prior box. No correlation.

`pool_mode` (``bounded`` default / ``unrestricted``) is identical to the Evaluation and
Experiment convention: bounded rejection-samples the pool within the imaging prior box;
unrestricted takes the raw-flow draws, whose mass may lie outside it. The empirical and
gaussian forms are faithful to the calibration and are NOT clipped — their support is
governed by `pool_mode`. Only `box`/`box_user` are constrained to the prior box, clamped
(never silently) at build.

Distinction from the RDS nuisance. The reaction-diffusion biology marginalized during
detector calibration is a *transient* `BoxUniform` drawn on the fly during generation
(`detector_parameterization.build_nuisance_prior`); it is never instantiated as an object,
and only its per-simulation draws persist (as the `Nuisance_RDS_Theta_Set`). The two share
the marginalization *role* but not the artifact: the `Nuisance_DLI` is a persisted
calibration result, so it — and only it — is the object defined here.

Enforcement. Downstream consumers call `require_nuisance_dli(...)`, which loads the built
artifact and fails loud (naming the analysis to run) if it is absent — the `Nuisance_DLI`
is a user decision, never an automatic fabrication.
"""
import json
import logging
from pathlib import Path

import numpy as np

try:
    import tomllib                       # stdlib on Python 3.11+ (this env is 3.13)
except ModuleNotFoundError:              # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger(__name__)

SPEC_SUFFIX = "_Nuisance_DLI_Spec.toml"      # the user-authored spec (analysis-emitted)
ARTIFACT_SUFFIX = "_Nuisance_DLI.npz"        # the built, samplable Nuisance_DLI artifact
ARTIFACT_FORMAT_VERSION = 2                  # v2: posterior_sample_pool_choice scheme
POOL_FORMAT_VERSION = 1                      # v1: pool cache carries per-row kind_index/cell/chunk labels

POOL_CHOICES = ("raw", "map_estimate_pool", "gaussian", "box", "box_user")
POOL_MODES = ("bounded", "unrestricted")
DEFAULT_BOX_QUANTILES = (0.05, 0.95)
_EMPIRICAL = ("raw", "map_estimate_pool")    # stored as a sample matrix, resampled per-vector
_BOX = ("box", "box_user")                   # stored as low/high, drawn as a per-param uniform

# Which cached pool each choice draws on. raw/gaussian/box share the posterior-sample pool;
# map_estimate_pool uses the MAP pool; box_user needs none. Caching keyed on this so switching
# among raw/gaussian/box never recomputes, and each pool's GPU cost is paid at most once.
POOL_KINDS = {"raw": "PosteriorSample", "gaussian": "PosteriorSample", "box": "PosteriorSample",
              "map_estimate_pool": "MapEstimate", "box_user": None}

_ANALYSIS = "Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py"


def spec_path(posit_dir, project_alias, timing_label):
    """Path of the user-authored Nuisance_DLI spec (in the Posit subdir)."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}{SPEC_SUFFIX}"


def artifact_path(posit_dir, project_alias, timing_label):
    """Path of the built, samplable Nuisance_DLI artifact."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}{ARTIFACT_SUFFIX}"


# =============================================================================
# The artifact (numpy-only: generation samples/loads it without torch)
# =============================================================================
class NuisanceDLI:
    """Samplable, self-describing calibrated-imaging nuisance (see module docstring).

    One of three underlying representations, selected by ``posterior_sample_pool_choice``:
    ``box``/``box_user`` draw a per-parameter uniform over ``[low, high]`` (independent per
    dimension — no cross-parameter correlation); ``raw``/``map_estimate_pool`` resample a
    stored matrix **per whole vector** (rows), preserving the joint correlation structure
    exactly; ``gaussian`` draws ``N(mean, cov)`` (a full covariance keeps the *linear*
    correlations, but not multimodality or curvature). ``prior_low``/``prior_high`` (the
    imaging prior box) are recorded for provenance; only ``box``/``box_user`` are clamped to
    it. Sampling is numpy-only, so generation needs neither torch nor the estimator.
    """

    def __init__(self, parameter_keys, posterior_sample_pool_choice, pool_mode,
                 prior_low, prior_high, *, low=None, high=None, samples=None,
                 mean=None, cov=None, space="log10"):
        self.parameter_keys = list(parameter_keys)
        self.posterior_sample_pool_choice = posterior_sample_pool_choice
        self.pool_mode = pool_mode
        self.prior_low = np.asarray(prior_low, dtype=float)
        self.prior_high = np.asarray(prior_high, dtype=float)
        self.space = space
        self.low = None if low is None else np.asarray(low, dtype=float)
        self.high = None if high is None else np.asarray(high, dtype=float)
        self.samples = None if samples is None else np.asarray(samples, dtype=float)
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.cov = None if cov is None else np.asarray(cov, dtype=float)

        d = len(self.parameter_keys)
        if self.posterior_sample_pool_choice not in POOL_CHOICES:
            raise ValueError(f"posterior_sample_pool_choice "
                             f"{self.posterior_sample_pool_choice!r} must be one of {POOL_CHOICES}.")
        if self.pool_mode not in POOL_MODES:
            raise ValueError(f"pool_mode {self.pool_mode!r} must be one of {POOL_MODES}.")
        if self.prior_low.shape != (d,) or self.prior_high.shape != (d,):
            raise ValueError(f"prior_low/prior_high must have shape ({d},) to match "
                             f"parameter_keys; got {self.prior_low.shape}, {self.prior_high.shape}.")
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            if (self.low is None or self.high is None
                    or self.low.shape != (d,) or self.high.shape != (d,)):
                raise ValueError(f"choice {choice!r} needs low/high of shape ({d},).")
        elif choice in _EMPIRICAL:
            if (self.samples is None or self.samples.ndim != 2
                    or self.samples.shape[1] != d):
                raise ValueError(f"choice {choice!r} needs a (n, {d}) sample matrix; got "
                                 f"{None if self.samples is None else self.samples.shape}.")
        else:  # gaussian
            if (self.mean is None or self.cov is None
                    or self.mean.shape != (d,) or self.cov.shape != (d, d)):
                raise ValueError(f"choice 'gaussian' needs mean ({d},) and cov ({d}, {d}).")

    # -- builders ----------------------------------------------------------
    @classmethod
    def from_box(cls, parameter_keys, low, high, *, choice="box", pool_mode="bounded",
                 prior_low=None, prior_high=None, space="log10"):
        """A per-parameter uniform over ``[low, high]`` (``box`` from pool quantiles, or
        ``box_user`` from user ranges); ``low``/``high`` are already clamped to the prior box."""
        return cls(parameter_keys, choice, pool_mode, prior_low, prior_high,
                   low=low, high=high, space=space)

    @classmethod
    def from_samples(cls, parameter_keys, samples, *, choice="raw", pool_mode="bounded",
                     prior_low=None, prior_high=None, space="log10"):
        """A stored sample matrix (``raw`` posterior draws or ``map_estimate_pool`` MAP
        vectors), resampled per whole vector at draw time to preserve correlations."""
        return cls(parameter_keys, choice, pool_mode, prior_low, prior_high,
                   samples=samples, space=space)

    @classmethod
    def from_gaussian(cls, parameter_keys, mean, cov, *, pool_mode="bounded",
                      prior_low=None, prior_high=None, space="log10"):
        """A full-covariance multivariate normal fit to the posterior_sample_pool."""
        return cls(parameter_keys, "gaussian", pool_mode, prior_low, prior_high,
                   mean=mean, cov=cov, space=space)

    # -- sampling (numpy only; no clipping — see module docstring) ---------
    def sample(self, n):
        """Draw ``n`` nuisance vectors, shape ``(n, D)`` in the declared ``space`` (log10).

        ``box``/``box_user`` draw each parameter independently within ``[low, high]``;
        ``raw``/``map_estimate_pool`` resample **whole rows** of the stored matrix (so the
        joint correlations are preserved — never per-column); ``gaussian`` draws from
        ``N(mean, cov)``. No clipping: the empirical and gaussian forms are faithful to the
        calibration, and ``box``/``box_user`` are bounded by construction.
        """
        n = int(n)
        rng = np.random.default_rng()
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            return rng.uniform(self.low, self.high, size=(n, self.low.size))
        if choice in _EMPIRICAL:
            idx = rng.integers(0, self.samples.shape[0], size=n)
            return self.samples[idx]                          # whole-vector resample
        return rng.multivariate_normal(self.mean, self.cov, size=n)   # gaussian

    # -- persistence -------------------------------------------------------
    def manifest(self):
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "parameter_keys": self.parameter_keys,
            "posterior_sample_pool_choice": self.posterior_sample_pool_choice,
            "pool_mode": self.pool_mode,
            "space": self.space,
            "n_samples": (int(self.samples.shape[0]) if self.posterior_sample_pool_choice
                          in _EMPIRICAL else None),
        }

    def flush(self, path):
        """Write the self-contained artifact to ``path`` as a compressed ``.npz`` (the
        payload arrays for its choice, the prior box, and the manifest)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"prior_low": self.prior_low, "prior_high": self.prior_high,
                  "manifest": np.asarray(json.dumps(self.manifest()))}
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            arrays["low"] = self.low
            arrays["high"] = self.high
        elif choice in _EMPIRICAL:
            arrays["samples"] = self.samples
        else:                                                 # gaussian
            arrays["mean"] = self.mean
            arrays["cov"] = self.cov
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path):
        """Reload a flushed artifact for sampling (numpy only)."""
        with np.load(str(path), allow_pickle=False) as data:
            m = json.loads(str(data["manifest"]))
            payload = {k: data[k] for k in ("low", "high", "samples", "mean", "cov")
                       if k in data.files}
            return cls(m["parameter_keys"], m["posterior_sample_pool_choice"],
                       m["pool_mode"], data["prior_low"], data["prior_high"],
                       space=m.get("space", "log10"), **payload)


# =============================================================================
# Pool construction (analysis --build; needs the estimator + GPU)
# =============================================================================
def build_posterior_sample_pool(posterior, chunks, device, vista_device, *,
                                 n_per_chunk, theta_prex_batch_size, pool_mode,
                                 chunk_labels=None):
    """Draw ``n_per_chunk`` posterior samples per chunk and pool them, shape ``(N, D)``.

    Uses the Evaluation/Experiment candidate sampler (`collect_theta_prex`) conditioned on
    each chunk in turn, so ``bounded``/``unrestricted`` behave exactly as in those stages,
    and concatenates the draws over every chunk of every recording. The result is a mixture
    over the real acquisitions — the empirical calibrated-imaging distribution. Each draw is
    a whole parameter vector, so the joint correlation structure is retained in the pool.

    Returns ``(pool, labels)``. ``labels`` is ``None`` unless ``chunk_labels`` (one
    ``(kind_index, cell, chunk)`` tuple per chunk, aligned to ``chunks``) is given, in which
    case it is the per-row ``{"kind_index", "cell", "chunk"}`` dict expanded to each chunk's
    ACTUAL draw count — so it stays row-aligned even if bounded rejection yields a variable
    number of accepted draws per chunk."""
    import torch
    from .evaluation import collect_theta_prex
    from .inference_support import normalize_video

    pieces, ki, ce, ch = [], [], [], []
    for i, chunk in enumerate(chunks):
        omega = torch.tensor(normalize_video(chunk), dtype=torch.float32, device=device)
        cond = omega.unsqueeze(0)
        posterior.set_default_x(cond)
        flow = posterior.posterior_estimator
        drawn = collect_theta_prex(posterior, flow, vista_device, cond,
                                   n_per_chunk, theta_prex_batch_size, pool_mode)
        arr = drawn.detach().cpu().numpy()
        pieces.append(arr)
        if chunk_labels is not None:
            k, c, h = chunk_labels[i]
            ki.append(np.full(arr.shape[0], k, dtype=np.int64))
            ce.append(np.full(arr.shape[0], c, dtype=np.int64))
            ch.append(np.full(arr.shape[0], h, dtype=np.int64))
    if not pieces:
        raise ValueError("no experimental chunks were provided; the posterior_sample_pool "
                         "is empty (check the experimental recordings and windowing).")
    pool = np.concatenate(pieces, axis=0)
    labels = (None if chunk_labels is None
              else {"kind_index": np.concatenate(ki), "cell": np.concatenate(ce),
                    "chunk": np.concatenate(ch)})
    return pool, labels


def build_map_estimate_pool(posterior, chunks, device, vista_device, eval_cfg, *,
                            pool_mode, log_fn=None, chunk_labels=None):
    """MAP-estimate each chunk (seed-then-optimize) and pool the per-chunk MAP vectors,
    shape ``(n_chunks, D)``. Uses `evaluation.map_estimate` with the shared Evaluation
    hyperparameters, so the point estimates match the Detector Experiment stage.

    Returns ``(pool, labels)`` -- one MAP vector per chunk; ``labels`` (when ``chunk_labels``,
    one ``(kind_index, cell, chunk)`` per chunk, is given) is the per-window
    ``{"kind_index", "cell", "chunk"}`` dict, row-aligned to the ``(n_chunks, D)`` pool."""
    from .evaluation import map_estimate

    lr = (eval_cfg.learning_rate if eval_cfg.learning_rate
          else eval_cfg.learning_rate_minimum * eval_cfg.learning_rate_maximum_factor)
    tolerance = eval_cfg.learning_rate_minimum * eval_cfg.tolerance_factor
    pieces = []
    for chunk in chunks:
        _score, theta_log = map_estimate(
            posterior, chunk, device, vista_device,
            eval_cfg.theta_prex_size, eval_cfg.theta_prex_batch_size,
            eval_cfg.score_prex_batch_size, eval_cfg.elite_prex_size,
            eval_cfg.numb_steps, eval_cfg.optimizer_patience,
            eval_cfg.scheduler_patience, eval_cfg.show_progress_steps,
            eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
            lr, tolerance, pool_mode=pool_mode, log_fn=log_fn,
        )
        pieces.append(np.asarray(theta_log, dtype=float))
    if not pieces:
        raise ValueError("no experimental chunks were provided; the map_estimate_pool is "
                         "empty (check the experimental recordings and windowing).")
    pool = np.stack(pieces, axis=0)
    labels = (None if chunk_labels is None
              else {"kind_index": np.asarray([lbl[0] for lbl in chunk_labels], dtype=np.int64),
                    "cell": np.asarray([lbl[1] for lbl in chunk_labels], dtype=np.int64),
                    "chunk": np.asarray([lbl[2] for lbl in chunk_labels], dtype=np.int64)})
    return pool, labels


# =============================================================================
# Fit helpers (pool -> a representation)
# =============================================================================
def fit_gaussian(pool):
    """Full-covariance Gaussian fit: ``(mean (D,), cov (D, D))`` from the pool rows.

    ``cov`` is the joint sample covariance over whole vectors, so its off-diagonals carry
    the linear cross-parameter correlations (e.g. the ``kappa_g``/``kappa_c`` degeneracy).
    A tiny diagonal jitter keeps it positive-definite for sampling.
    """
    pool = np.asarray(pool, dtype=float)
    mean = pool.mean(axis=0)
    cov = np.atleast_2d(np.cov(pool, rowvar=False))
    cov = cov + 1e-9 * np.eye(cov.shape[0])
    return mean, cov


def quantile_box(pool, quantiles, prior_low, prior_high):
    """Per-parameter ``[low, high]`` from pool quantiles, clamped to the prior box.

    Returns clamped ``(low, high)`` and logs — never silently — how many parameter ranges
    the prior-box clamp reduced.
    """
    pool = np.asarray(pool, dtype=float)
    ql, qh = float(quantiles[0]), float(quantiles[1])
    if not 0.0 <= ql < qh <= 1.0:
        raise ValueError(f"box_quantiles must satisfy 0 <= low < high <= 1; got {quantiles}.")
    low = np.quantile(pool, ql, axis=0)
    high = np.quantile(pool, qh, axis=0)
    pl = np.asarray(prior_low, dtype=float)
    ph = np.asarray(prior_high, dtype=float)
    clamped_low = np.maximum(low, pl)
    clamped_high = np.minimum(high, ph)
    n_clamped = int(np.sum((clamped_low != low) | (clamped_high != high)))
    if n_clamped:
        logger.warning("Nuisance_DLI[box]: clamped %d/%d parameter range(s) to the imaging "
                       "prior box.", n_clamped, int(low.size))
    return clamped_low, clamped_high


# =============================================================================
# Pool cache — separate the expensive GPU pool from the cheap per-choice representation
# =============================================================================
# A pool is expensive (embedding + sampling, or 1 MAP optimization per chunk); the choice
# applied to it is cheap. So each pool is persisted once, stamped with the exact inputs that
# produced it, and reused across `--build` invocations (and across raw/gaussian/box, which
# share the posterior-sample pool). It is recomputed only when an input changes.

def pool_cache_path(posit_dir, project_alias, timing_label, pool_kind):
    """Path of a cached pool (``pool_kind`` in ``{"PosteriorSample", "MapEstimate"}``)."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}_Nuisance_DLI_{pool_kind}Pool.npz"


def pool_provenance(*, pool_mode, n_per_chunk, span_seconds, chunk_step_seconds, kinds,
                    max_cells, estimator_sha256):
    """The cache key: every input that determines the pool's contents. A cached pool is
    reused only when this matches exactly, so any change (including a different estimator,
    identified by its weights checksum) forces a recompute."""
    return {
        "pool_mode": pool_mode,
        "n_per_chunk": int(n_per_chunk),
        "span_seconds": int(span_seconds),
        "chunk_step_seconds": (None if chunk_step_seconds is None else int(chunk_step_seconds)),
        "kinds": list(kinds),
        "max_cells": int(max_cells),
        "estimator_sha256": estimator_sha256,
    }


def save_pool(path, pool, provenance, *, labels=None):
    """Persist a pool matrix + its provenance to ``path`` (compressed ``.npz``).

    When ``labels`` is given -- a ``{"kind_index", "cell", "chunk", "kinds"}`` dict of the
    per-row condition and time-window (cell, sliding-window chunk) fields row-aligned to
    ``pool`` (``kinds`` is the shared name array ``kind_index`` indexes into) -- the pool
    becomes self-describing and is stamped with ``pool_format_version``. Omitting ``labels``
    writes the legacy label-less layout, so an un-upgraded caller is unaffected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"pool": np.asarray(pool, dtype=float),
              "provenance": np.asarray(json.dumps(provenance, sort_keys=True))}
    if labels is not None:
        n = arrays["pool"].shape[0]
        for key in ("kind_index", "cell", "chunk"):
            col = np.asarray(labels[key])
            if col.shape[0] != n:
                raise ValueError(
                    f"save_pool: labels['{key}'] length {col.shape[0]} != pool rows {n}.")
            arrays[key] = col.astype(np.int64)
        arrays["kinds"] = np.asarray(labels["kinds"])
        arrays["pool_format_version"] = np.asarray(POOL_FORMAT_VERSION)
    np.savez_compressed(str(path), **arrays)


def load_pool_if_fresh(path, provenance):
    """Return the cached pool iff ``path`` exists and its stored provenance matches
    ``provenance`` exactly; otherwise ``None`` (missing or stale -> caller recomputes).

    Returns the pool matrix only; per-row labels (when present) are read separately via
    :func:`load_pool_labels`, since a representation fit does not need them."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as data:
        stored = json.loads(str(data["provenance"]))
        if stored == provenance:
            return data["pool"]
    return None


def load_pool_labels(path):
    """Return the pool cache's per-row labels dict (``kind_index``, ``cell``, ``chunk``,
    ``kinds``, ``pool_format_version``) for a self-describing (labeled) pool, else ``None``
    for a legacy label-less pool. Reads only the small label arrays, not the pool matrix."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as data:
        if "pool_format_version" not in data.files:
            return None
        return {"kind_index": data["kind_index"], "cell": data["cell"],
                "chunk": data["chunk"], "kinds": data["kinds"],
                "pool_format_version": int(data["pool_format_version"])}


# =============================================================================
# Emit the spec template (pre-filled with pool-derived suggestions)
# =============================================================================
def emit_spec_template(path, imaging_keys, suggestions, *, pool_mode="bounded", provenance=None):
    """Write a commented value-based Nuisance_DLI spec pre-filled with SUGGESTIONS.

    ``suggestions[key]`` may carry ``ci_low`` / ``ci_high`` (log10, the sampling space),
    used to pre-fill the per-parameter ranges that ``box_user`` reads. ``pool_mode`` is the
    mode the suggestion pool was drawn under and becomes the spec's default (so the finalized
    spec matches how emit sampled). Every number is a suggestion; the user edits the file to
    set the ultimate choice and values. Nothing here is authoritative until the user saves it.
    """
    def _num(v):
        return "null" if v is None else repr(round(float(v), 4))

    ql, qh = DEFAULT_BOX_QUANTILES
    out = [
        "# ============================================================================",
        "# Nuisance_DLI spec -- EDIT THIS FILE, then build with --build.",
        "# ============================================================================",
        "# The detector estimator is REQUIRED. --build reads ALL experimental recordings,",
        "# draws the posterior per model-window chunk on the fly, and pools the draws into",
        "# the posterior_sample_pool. `posterior_sample_pool_choice` decides how that pool",
        "# becomes the samplable Nuisance_DLI. All values are in LOG10 (the sampling space).",
        "#",
        "#   raw               -> resample the pool per whole vector (preserves correlations; DEFAULT).",
        "#   map_estimate_pool -> resample the pooled per-chunk MAP estimates (point estimates).",
        "#   gaussian          -> full-covariance Gaussian fit to the pool (linear correlations).",
        "#   box               -> per-parameter uniform over `box_quantiles` of the pool, clamped.",
        "#   box_user          -> per-parameter uniform over the [imaging.<KEY>] ranges below, clamped.",
        "#",
        "# pool_mode matches Evaluation/Experiment: bounded (rejection within the prior) | unrestricted.",
        "# raw / map_estimate_pool / gaussian are NOT clipped; box / box_user clamp to the prior box.",
        f"# provenance (auto-filled, do not edit): {json.dumps(provenance or {}, sort_keys=True)}",
        "",
        "[block]",
        'posterior_sample_pool_choice = "raw"   # raw | map_estimate_pool | gaussian | box | box_user',
        f'pool_mode = "{pool_mode}"                  # bounded | unrestricted',
        f"box_quantiles = [{ql}, {qh}]                # used only by \"box\"",
        "",
        "# Per-parameter ranges are read ONLY by \"box_user\"; pre-filled with the pool's",
        f"# {int(ql * 100)}th/{int(qh * 100)}th percentiles as suggestions. Ignored by every other choice.",
    ]
    for k in imaging_keys:
        s = suggestions.get(k, {}) if suggestions else {}
        out += [
            f"[imaging.{k}]",
            f"low  = {_num(s.get('ci_low'))}",
            f"high = {_num(s.get('ci_high'))}",
            "",
        ]
    Path(path).write_text("\n".join(out))
    return Path(path)


# =============================================================================
# Load + validate the (user-finalized) spec
# =============================================================================
def load_spec(path, imaging_keys, prior_low, prior_high):
    """Parse and FULLY validate the user-authored spec.

    Validates: ``posterior_sample_pool_choice`` in ``POOL_CHOICES``; ``pool_mode`` in
    ``POOL_MODES``; ``box_quantiles`` (when ``box``); and — for ``box_user`` only — that
    every imaging key has a ``[low, high]`` with ``low < high`` lying inside the imaging
    prior box (``prior_low``/``prior_high``, log10, in ``imaging_keys`` order). The other
    choices read no per-parameter ranges. Raises ``ValueError`` with a precise message.
    """
    with open(path, "rb") as fh:
        spec = tomllib.load(fh)
    block = spec.get("block", {})
    choice = block.get("posterior_sample_pool_choice")
    if choice not in POOL_CHOICES:
        raise ValueError(f"{path}: [block].posterior_sample_pool_choice={choice!r} must be "
                         f"one of {POOL_CHOICES}.")
    pool_mode = block.get("pool_mode", "bounded")
    if pool_mode not in POOL_MODES:
        raise ValueError(f"{path}: [block].pool_mode={pool_mode!r} must be one of {POOL_MODES}.")
    if choice == "box":
        q = block.get("box_quantiles", list(DEFAULT_BOX_QUANTILES))
        if (not isinstance(q, (list, tuple)) or len(q) != 2
                or not 0.0 <= float(q[0]) < float(q[1]) <= 1.0):
            raise ValueError(f"{path}: [block].box_quantiles must be [low, high] with "
                             f"0 <= low < high <= 1; got {q!r}.")
    if choice == "box_user":
        imaging = spec.get("imaging", {})
        missing = [k for k in imaging_keys if k not in imaging]
        if missing:
            raise ValueError(f"{path}: choice 'box_user' needs an [imaging.<KEY>] range for "
                             f"every parameter; missing {missing}.")
        bl = {k: float(prior_low[i]) for i, k in enumerate(imaging_keys)}
        bh = {k: float(prior_high[i]) for i, k in enumerate(imaging_keys)}
        tol = 1e-9
        for k in imaging_keys:
            row = imaging[k]
            lo, hi = row.get("low"), row.get("high")
            if lo is None or hi is None:
                raise ValueError(f"{path}: [imaging.{k}] needs `low` and `high` (log10).")
            lo, hi = float(lo), float(hi)
            if not lo < hi:
                raise ValueError(f"{path}: [imaging.{k}] low {lo} must be < high {hi}.")
            if lo < bl[k] - tol or hi > bh[k] + tol:
                raise ValueError(
                    f"{path}: [imaging.{k}] range [{lo}, {hi}] exceeds the imaging prior box "
                    f"[{bl[k]}, {bh[k]}] (log10); a Nuisance_DLI range must lie within it.")
    return spec


# =============================================================================
# Build the Nuisance_DLI from the finalized spec (+ pool, for the derived choices)
# =============================================================================
def build_nuisance_dli(spec, imaging_keys, prior_low, prior_high, *, pool=None):
    """Construct the `NuisanceDLI` for the spec's choice, from an already-computed pool.

    CPU-only: the expensive pool is built and cached by the caller (see the pool cache and
    `build_posterior_sample_pool` / `build_map_estimate_pool`), so switching among choices
    that share a pool costs nothing here. ``box_user`` needs no pool; every other choice
    requires the matching ``pool`` (the posterior-sample pool for raw/gaussian/box, the MAP
    pool for map_estimate_pool).
    """
    block = spec["block"]
    choice = block["posterior_sample_pool_choice"]
    pool_mode = block.get("pool_mode", "bounded")

    if choice == "box_user":
        low = [float(spec["imaging"][k]["low"]) for k in imaging_keys]
        high = [float(spec["imaging"][k]["high"]) for k in imaging_keys]
        return NuisanceDLI.from_box(imaging_keys, low, high, choice="box_user",
                                    pool_mode=pool_mode, prior_low=prior_low,
                                    prior_high=prior_high)

    if pool is None:
        raise ValueError(f"choice {choice!r} needs its pool (pass pool=...); only 'box_user' "
                         "builds without one.")

    if choice in ("raw", "map_estimate_pool"):
        return NuisanceDLI.from_samples(imaging_keys, pool, choice=choice, pool_mode=pool_mode,
                                        prior_low=prior_low, prior_high=prior_high)
    if choice == "gaussian":
        mean, cov = fit_gaussian(pool)
        return NuisanceDLI.from_gaussian(imaging_keys, mean, cov, pool_mode=pool_mode,
                                         prior_low=prior_low, prior_high=prior_high)
    # choice == "box"
    q = block.get("box_quantiles", list(DEFAULT_BOX_QUANTILES))
    low, high = quantile_box(pool, q, prior_low, prior_high)
    return NuisanceDLI.from_box(imaging_keys, low, high, choice="box", pool_mode=pool_mode,
                                prior_low=prior_low, prior_high=prior_high)


# =============================================================================
# The enforcement gate — downstream generation LOADS the built artifact
# =============================================================================
def require_nuisance_dli(posit_dir, project_alias, timing_label):
    """LOAD the built Nuisance_DLI artifact, or FAIL CLEARLY (naming the analysis).

    Generation consumes the persisted, self-contained artifact; it does not rebuild
    (rebuilding needs the estimator and a GPU — the analysis step's job). If the artifact is
    absent, the analysis has not been run: this raises a clear, actionable error rather than
    fabricating a nuisance. The `Nuisance_DLI` is a user decision, never an automatic output.
    """
    ap = artifact_path(posit_dir, project_alias, timing_label)
    if not ap.exists():
        sp = spec_path(posit_dir, project_alias, timing_label)
        raise FileNotFoundError(
            f"Nuisance_DLI artifact not found:\n    {ap}\n\n"
            f"The Nuisance_DLI is a user decision built by the analysis step, not an "
            f"automatic output. Run it to inspect the calibrated imaging, emit the spec, "
            f"edit it, and build:\n"
            f"    python {_ANALYSIS} --emit-template --total-time-seconds <T>\n"
            f"    (edit {sp.name}: set posterior_sample_pool_choice and, for box_user, the ranges)\n"
            f"    python {_ANALYSIS} --build --total-time-seconds <T>\n\n"
            f"See DETECTOR_WORKFLOW.md, \"Nuisance and artifact design\".")
    return NuisanceDLI.load(ap)
