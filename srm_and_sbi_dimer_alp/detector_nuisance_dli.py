"""Nuisance_DLI construction from a user-authored, analysis-emitted value-based spec.

Design: see "Constructing the Nuisance_DLI (the analysis step)" in DETECTOR_WORKFLOW.md.

The Detector calibrates the imaging model; a person then decides how that calibration
feeds production by editing a value-based spec — one table per imaging parameter, in the
same vocabulary as the Detector parameter table (roles: fixed / uniform / posterior /
samples) — that the Nuisance_DLI analysis EMITS pre-filled with posterior-derived
suggestions, reads back, and builds into a single pooled `Nuisance` (`domain="DLI"`).

Enforcement (the reason this module exists as a shared guard): the spec is a
PREREQUISITE. Downstream consumers call `require_nuisance_dli(...)`; if the finalized
spec is absent — because the analysis has not been run — it raises a clear error that
names the analysis to run first, rather than silently fabricating a nuisance.
"""
import json
import logging
from pathlib import Path

import numpy as np

try:
    import tomllib                       # stdlib on Python 3.11+ (this env is 3.13)
except ModuleNotFoundError:              # pragma: no cover
    import tomli as tomllib

from .nuisance import Nuisance

logger = logging.getLogger(__name__)

SPEC_SUFFIX = "_Nuisance_DLI_Spec.toml"      # the user-authored spec (analysis-emitted)
ARTIFACT_SUFFIX = "_Nuisance_DLI.npz"        # the built, samplable Nuisance artifact
PARAM_ROLES = ("fixed", "uniform")           # per-parameter roles (compose into a box)
BLOCK_FORMS = ("perparam", "posterior", "samples")

_ANALYSIS = "Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py"


def spec_path(posit_dir, project_alias, timing_label):
    """Path of the user-authored Nuisance_DLI spec (in the Posit subdir)."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}{SPEC_SUFFIX}"


def artifact_path(posit_dir, project_alias, timing_label):
    """Path of the built, samplable Nuisance_DLI artifact."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}{ARTIFACT_SUFFIX}"


# =============================================================================
# Emit the spec template (pre-filled with posterior-derived suggestions)
# =============================================================================

def emit_spec_template(path, imaging_keys, suggestions, *, timing_label="",
                       provenance=None):
    """Write a commented, value-based Nuisance_DLI spec pre-filled with SUGGESTIONS.

    ``suggestions[key]`` may carry ``map`` / ``ci_low`` / ``ci_high`` (all log10, the
    sampling space). Every number is a *suggestion*; the user edits the file to set the
    ultimate role and values. Nothing here is authoritative until the user saves it.
    """
    def _num(v):
        return "null" if v is None else repr(round(float(v), 4))

    out = [
        "# ============================================================================",
        "# Nuisance_DLI spec (value-based) -- EDIT THIS FILE, then build with --build.",
        "# ============================================================================",
        "# One [imaging.<KEY>] table per imaging parameter. All values are in LOG10 space",
        "# (the sampling space), matching the Detector parameter table.",
        "#",
        "# Per-parameter role (used when [block].form = \"perparam\"):",
        "#   role = \"fixed\"    -> held constant at `value`.",
        "#   role = \"uniform\"  -> BoxUniform over [low, high]  (the Nuisance_RDS analogue).",
        "# Whole-block forms (imaging block drawn jointly; set [block].form instead):",
        "#   \"posterior\"       -> drawn from the trained imaging estimator (--pool-mode honored).",
        "#   \"samples\"         -> drawn from a stored sample vector at [block].samples_path.",
        "#",
        "# The numbers below are SUGGESTIONS from the calibration posterior; you decide finals.",
        f"# provenance: {json.dumps(provenance or {}, sort_keys=True)}",
        "",
        "[block]",
        f'timing_label = "{timing_label}"',
        'form = "perparam"          # "perparam" | "posterior" | "samples"',
        'samples_path = ""           # required only when form = "samples"',
        "",
    ]
    for k in imaging_keys:
        s = suggestions.get(k, {}) if suggestions else {}
        out += [
            f"[imaging.{k}]",
            'role = "uniform"           # "fixed" | "uniform"',
            f"low   = {_num(s.get('ci_low'))}        # suggestion: posterior 5th percentile (log10)",
            f"high  = {_num(s.get('ci_high'))}        # suggestion: posterior 95th percentile (log10)",
            f"value = {_num(s.get('map'))}        # suggestion: MAP (log10); used if role = \"fixed\"",
            "",
        ]
    Path(path).write_text("\n".join(out))
    return Path(path)


# =============================================================================
# Load + validate the (user-finalized) spec
# =============================================================================

def load_spec(path, imaging_keys, prior_low, prior_high):
    """Parse and FULLY validate the user-authored spec: structure AND values within box.

    Validates (1) structure -- `[block].form` valid, every imaging key present, valid
    per-parameter role, required fields per role, `low < high`; and (2) that every
    user-authored value lands inside the Detector imaging prior box (`prior_low` /
    `prior_high`, log10, one per key in `imaging_keys` order): a `fixed` value must sit
    in the box, and a `uniform` `[low, high]` must lie within it. The calibration
    operates within that box, so a `Nuisance_DLI` cannot assert imaging outside it.
    (Drawn forms -- `posterior` / `samples` -- are clipped to the box with never-silent
    logging at build time, not rejected here.) Raises ``ValueError`` with a precise
    message on any violation.
    """
    with open(path, "rb") as fh:
        spec = tomllib.load(fh)
    block = spec.get("block", {})
    form = block.get("form", "perparam")
    if form not in BLOCK_FORMS:
        raise ValueError(f"{path}: [block].form={form!r} must be one of {BLOCK_FORMS}.")
    imaging = spec.get("imaging", {})
    missing = [k for k in imaging_keys if k not in imaging]
    if missing:
        raise ValueError(f"{path}: missing [imaging.<KEY>] table(s) for {missing}.")
    bl = {k: float(prior_low[i]) for i, k in enumerate(imaging_keys)}
    bh = {k: float(prior_high[i]) for i, k in enumerate(imaging_keys)}
    tol = 1e-9
    if form == "perparam":
        for k in imaging_keys:
            row = imaging[k]
            role = row.get("role")
            if role not in PARAM_ROLES:
                raise ValueError(f"{path}: [imaging.{k}].role={role!r} must be one of {PARAM_ROLES}.")
            if role == "fixed":
                v = row.get("value")
                if v is None:
                    raise ValueError(f"{path}: [imaging.{k}] role='fixed' needs a `value` (log10).")
                if not (bl[k] - tol <= float(v) <= bh[k] + tol):
                    raise ValueError(
                        f"{path}: [imaging.{k}] value {v} is outside the imaging prior box "
                        f"[{bl[k]}, {bh[k]}] (log10); the calibration cannot assert imaging outside it.")
            else:  # uniform
                lo, hi = row.get("low"), row.get("high")
                if lo is None or hi is None:
                    raise ValueError(f"{path}: [imaging.{k}] role='uniform' needs `low` and `high` (log10).")
                lo, hi = float(lo), float(hi)
                if not lo < hi:
                    raise ValueError(f"{path}: [imaging.{k}] low {lo} must be < high {hi}.")
                if lo < bl[k] - tol or hi > bh[k] + tol:
                    raise ValueError(
                        f"{path}: [imaging.{k}] range [{lo}, {hi}] exceeds the imaging prior box "
                        f"[{bl[k]}, {bh[k]}] (log10); a Nuisance_DLI range must lie within it.")
    elif form == "samples" and not block.get("samples_path"):
        raise ValueError(f"{path}: [block].form='samples' needs a non-empty `samples_path`.")
    return spec


# =============================================================================
# Build the pooled Nuisance_DLI from the finalized spec
# =============================================================================

def build_nuisance_dli(spec, imaging_keys, prior_low, prior_high, *, estimator=None,
                       x=None, pool_mode="bounded", n_materialize=10000):
    """Construct the pooled `Nuisance` (domain='DLI') from a validated spec.

    ``prior_low`` / ``prior_high`` are the Detector imaging prior box (log10, per key);
    they are the clip box for the drawn forms, so any draw outside the box is clipped
    with never-silent logging (§7's clipping knob).

    - ``form='perparam'``: a `BoxUniform` over the per-parameter [low, high]; a ``fixed``
      parameter contributes a degenerate ``[value, value]`` bin. (`from_spec`.)
    - ``form='posterior'``: draws come from the supplied trained imaging ``estimator``
      (conditioned on ``x``); ``pool_mode`` governs the draw; clipped to the prior box.
    - ``form='samples'``: draws come from the stored sample vector; clipped to the box.
    """
    block = spec.get("block", {})
    form = block.get("form", "perparam")
    imaging = spec["imaging"]

    if form == "perparam":
        low, high = [], []
        for k in imaging_keys:
            row = imaging[k]
            if row["role"] == "fixed":
                v = float(row["value"]); low.append(v); high.append(v)
            else:                                    # uniform
                low.append(float(row["low"])); high.append(float(row["high"]))
        return Nuisance.from_spec(imaging_keys, low, high, space="log10", domain="DLI")

    if form == "posterior":
        if estimator is None:
            raise ValueError("form='posterior' needs the trained imaging estimator "
                             "(pass estimator=...); it is drawn from the posterior.")
        if x is None:
            raise ValueError("form='posterior' is drawn from the posterior conditioned on the real "
                             "recordings; pass x=<observation embedding>. There is no unconditioned "
                             "default draw for this form.")
        if pool_mode not in ("bounded", "unrestricted"):
            raise ValueError(f"pool_mode={pool_mode!r} must be 'bounded' or 'unrestricted'.")
        import torch
        # Honor --pool-mode exactly as the Detector Experiment/Evaluation do
        # (evaluation.collect_theta_prex): 'bounded' = DirectPosterior rejection within the
        # prior box; 'unrestricted' = sample the raw flow (mass may lie outside, then clipped).
        flow = getattr(estimator, "posterior_estimator", estimator)
        with torch.no_grad():
            if pool_mode == "unrestricted":
                drawn = flow.sample((n_materialize,), condition=x).squeeze(1)
            else:                                    # bounded
                drawn = estimator.sample((n_materialize,), x=x)
        drawn = drawn.detach().cpu().numpy()
        return Nuisance.from_samples(imaging_keys, drawn, list(prior_low), list(prior_high),
                                     space="log10", domain="DLI")

    # form == "samples" -- clip box is the prior box, not the sample extent
    samples = np.load(block["samples_path"])
    samples = samples["samples"] if hasattr(samples, "files") else np.asarray(samples)
    return Nuisance.from_samples(imaging_keys, samples, list(prior_low), list(prior_high),
                                 space="log10", domain="DLI")


# =============================================================================
# The enforcement guard — downstream stages call THIS
# =============================================================================

def require_nuisance_dli(posit_dir, project_alias, timing_label, imaging_keys,
                         prior_low, prior_high, *, estimator=None, x=None,
                         pool_mode="bounded"):
    """VALIDATE + build the finalized Nuisance_DLI, or FAIL CLEARLY.

    This is a validating gate, not a mere existence check. Matched-synthetic generation
    and the production import call it. It fails clearly and actionably when:
      - the spec is absent (the analysis has not been run) -- names the analysis to run;
      - the spec is malformed or its user-authored values fall outside the imaging prior
        box -- `load_spec` raises a precise ``ValueError`` naming the offending parameter.
    The Nuisance_DLI is a user decision, never an automatic fabrication, and it must
    conform to the calibration's guidance before any downstream stage may use it.
    """
    sp = spec_path(posit_dir, project_alias, timing_label)
    if not sp.exists():
        raise FileNotFoundError(
            f"Nuisance_DLI spec not found:\n    {sp}\n\n"
            f"The Nuisance_DLI is a user decision, not an automatic output. Run the "
            f"Nuisance_DLI analysis FIRST to inspect the calibrated imaging posterior and "
            f"emit the spec template, then edit it and build:\n"
            f"    python {_ANALYSIS} --emit-template --total-time-seconds <T>\n"
            f"    (edit {sp.name}: set each parameter's role and its ultimate values)\n"
            f"    python {_ANALYSIS} --build --total-time-seconds <T>\n\n"
            f"See DETECTOR_WORKFLOW.md, \"Constructing the Nuisance_DLI (the analysis step)\".")
    spec = load_spec(sp, imaging_keys, prior_low, prior_high)   # structure + prior-box validation
    return build_nuisance_dli(spec, imaging_keys, prior_low, prior_high,
                              estimator=estimator, x=x, pool_mode=pool_mode)
