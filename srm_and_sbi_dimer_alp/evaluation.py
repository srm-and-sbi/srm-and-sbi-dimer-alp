"""MAP-recovery evaluation for the inference stage.

Estimates the maximum-a-posteriori (MAP) parameter vector for each held-out
EVAL video and compares it against the known ground-truth theta, quantifying
how well the trained posterior recovers the simulation parameters.

The estimator is a **seed-then-optimize** procedure (one MAP per video):

    1. collect_theta_prex  -- draw a pool of candidate theta from the posterior
                              conditioned on the video (respects prior bounds).
    2. collect_score_prex  -- score every candidate by the flow's log-probability.
    3. extract_elite_prex  -- keep the top-K candidates as optimization seeds.
    4. optimize_elite      -- gradient-ascent the log-probability from those seeds
                              (Adam + ReduceLROnPlateau + early stopping);
                              return the single best (score, theta).

All theta live in log10 space (the flow's space and the prior's space); the
ground-truth theta sets are stored linear, so the report compares
``log10(theta_true)`` against the inferred log10 theta. The pool sampler uses
the bounded ``DirectPosterior`` (rejection sampling between prior ranges); the
gradient steps use ``posterior.posterior_estimator`` (the flow, which exposes
``log_prob`` and tracks gradients but does not enforce bounds).

Module contents:
    MAPEstimate(...)          -- full seed-then-optimize MAP for one video.
    recovery_stats(...)       -- per-parameter recovery error statistics.
    recovery_table(...)       -- (headers, rows) recovery summary for the report.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from .inference_support import normalize_video


# =============================================================================
# Seed-then-optimize MAP estimate (one video)
# =============================================================================

def _empty_cache(device: torch.device) -> None:
    """Release cached CUDA memory between batches (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _theta_repr(theta_log10: np.ndarray, arc: int = 3) -> str:
    """Compact ``[v0, v1, ...]`` repr of a theta vector, rounded to ``arc`` places."""
    return str([round(float(v), arc) for v in np.asarray(theta_log10).ravel()])


def collect_theta_prex(posterior, flow, vista_device: torch.device,
                       cond: torch.Tensor, theta_prex_size: int,
                       theta_prex_batch_size: int, pool_mode: str = "bounded",
                       show: bool = False, verbose: bool = False) -> torch.Tensor:
    """Draw a pool of candidate theta conditioned on ``cond``, in one of two modes.

    ``pool_mode``:
        ``"bounded"``      -- ``DirectPosterior.sample`` rejection-samples within
                              the prior ranges, so every candidate is a valid
                              prior draw. Correct for a well-trained posterior;
                              can stall if the posterior's mass lies mostly
                              outside the prior box (rejection never accepts).
        ``"unrestricted"`` -- sample the flow (``posterior_estimator``) directly,
                              with no prior-range rejection. Never stalls, so it
                              suits smoke tests and landscape exploration on an
                              undertrained posterior (candidates may fall outside
                              the prior box; the gradient ascent explores freely).

    Sampling is batched and offloaded to ``vista_device`` (typically CPU) to
    spare GPU memory until ``theta_prex_size`` candidates are collected.

    Returns:
        Tensor of shape ``(theta_prex_size, D)`` on ``vista_device``.
    """
    if pool_mode not in ("bounded", "unrestricted"):
        raise ValueError(
            f"pool_mode={pool_mode!r}; must be 'bounded' or 'unrestricted'.")
    theta_set = []
    quota = 0
    t0 = time.time()
    while quota < theta_prex_size:
        batch_size = min(theta_prex_batch_size, theta_prex_size - quota)
        if pool_mode == "unrestricted":
            # flow.sample -> (batch_size, 1, D); drop the singleton condition dim.
            theta = flow.sample((batch_size,), condition=cond).squeeze(1)
        else:
            theta = posterior.sample(sample_shape=(batch_size,), x=cond,
                                     show_progress_bars=verbose)
        theta_set.append(theta.to(vista_device))
        quota += batch_size
        _empty_cache(cond.device)
    theta_prex = torch.cat(theta_set, dim=0)
    if show:
        print(f"  [pool ] ({pool_mode}) candidate theta predictive samples: "
              f"shape={tuple(theta_prex.shape)}  ({time.time() - t0:.3f}s)",
              flush=True)
    return theta_prex


def collect_score_prex(flow, train_device: torch.device, vista_device: torch.device,
                       cond: torch.Tensor, theta_prex: torch.Tensor,
                       score_prex_batch_size: int, show: bool = False) -> torch.Tensor:
    """Score each candidate theta by the flow's log-probability given ``cond``.

    Returns:
        1D tensor of shape ``(theta_prex_size,)`` on ``vista_device``: the
        log-probability of each candidate under the posterior flow.
    """
    score_set = []
    quota = theta_prex.size(0)
    t0 = time.time()
    for index in range(0, quota, score_prex_batch_size):
        theta = theta_prex[index:index + score_prex_batch_size].to(train_device)
        with torch.no_grad():
            theta_batch = theta.unsqueeze(0)                       # (1, B, D)
            cond_batch = cond.squeeze(0).expand(theta.size(0), *cond.shape[1:])
            score_batch = flow.log_prob(input=theta_batch, condition=cond_batch)
            score = score_batch.squeeze(0)                         # (B,)
        score_set.append(score.to(vista_device))
        del theta, theta_batch, cond_batch, score_batch, score
        _empty_cache(train_device)
    score_prex = torch.cat(score_set, dim=0)
    if show:
        print(f"  [score] flow log-probability of candidates: "
              f"shape={tuple(score_prex.shape)}  ({time.time() - t0:.3f}s)",
              flush=True)
    return score_prex


def extract_elite_prex(theta_prex: torch.Tensor, score_prex: torch.Tensor,
                       elite_prex_size: int, show: bool = False) -> torch.Tensor:
    """Keep the top-``elite_prex_size`` candidates by log-probability as seeds."""
    indices = torch.topk(score_prex, elite_prex_size).indices
    elite_prex = theta_prex[indices]                               # (K, D)
    if show:
        print(f"  [elite] optimization seeds (top-{elite_prex_size}): "
              f"shape={tuple(elite_prex.shape)}", flush=True)
    return elite_prex


def optimize_elite(flow, train_device: torch.device, vista_device: torch.device,
                   cond: torch.Tensor, elite_prex: torch.Tensor,
                   numb_steps: int, optimizer_patience: int,
                   scheduler_patience: int, show_progress_steps: int,
                   learning_rate_minimum: float, learning_rate_factor: float,
                   learning_rate: float, tolerance: float,
                   show: bool = False, verbose: bool = False, log_fn=None) -> tuple:
    """Gradient-ascent the flow log-probability from the elite seeds.

    Optimizes all ``K`` seeds in parallel with Adam, reduces the learning rate
    on plateau, and stops early after ``optimizer_patience`` steps without
    improvement (the early-stopping criterion) -- the loop can therefore halt
    before ``numb_steps``. Tracks the single best ``(score, theta)`` seen across
    all seeds and steps.

    ``log_fn``, if given, is called with each per-step progress line (at the
    ``show_progress_steps`` cadence) and the final stop line, so a caller can
    stream them to a file live (e.g. the debug progress log). Console printing is
    independent and gated by ``show``.

    Returns:
        ``(optimal_score, optimal_theta)`` where ``optimal_score`` is a float
        log-probability and ``optimal_theta`` is a 1D numpy array (log10 space)
        on ``vista_device``.
    """
    theta = elite_prex.to(train_device).clone().detach()
    theta.requires_grad_(True)
    optimizer = torch.optim.Adam(params=[theta], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=learning_rate_factor,
        patience=scheduler_patience, threshold=tolerance,
        threshold_mode="abs", min_lr=learning_rate_minimum,
    )

    optimal_score = float("-inf")
    optimal_theta = None
    steps_without_improve = 0
    stop = "full-run"   # "full-run" (ran all steps) | "early" (patience-stopped)

    if show:
        print(f"  [optim] gradient-ascent: up to {numb_steps} steps, "
              f"optimizer-patience {optimizer_patience}", flush=True)
    if verbose:
        print(f"  [optim] scheduler-patience={scheduler_patience}, "
              f"lr={learning_rate:.3e} (min {learning_rate_minimum:.3e}, "
              f"factor {learning_rate_factor}), tolerance={tolerance:.3e}", flush=True)
    t0 = time.time()
    for step in range(1, numb_steps + 1):
        optimizer.zero_grad()
        theta_batch = theta.unsqueeze(0)                           # (1, K, D)
        cond_batch = cond.squeeze(0).expand(theta.size(0), *cond.shape[1:])
        score_batch = flow.log_prob(input=theta_batch, condition=cond_batch)
        score = score_batch.squeeze(0)                             # (K,)
        loss = -score.mean()        # mean keeps the lr consistent across K seeds
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            value, idx = score.max(dim=0)
            if value.item() > optimal_score + tolerance:
                optimal_score = value.item()
                optimal_theta = theta[idx].detach().clone()
                steps_without_improve = 0
            else:
                steps_without_improve += 1
        scheduler.step(loss.detach())   # scheduler only reads the value; detach avoids the requires_grad->scalar warning
        del theta_batch, cond_batch, score_batch, score, loss
        _empty_cache(train_device)
        if (show or log_fn is not None) and (step % show_progress_steps == 0 or step == 1):
            dynamic_threshold = scheduler.best - scheduler.threshold
            progress_msg = (f"progress [{step}/{numb_steps}]  "
                            f"lr={scheduler.get_last_lr()[0]:.2e}  "
                            f"dynamic_threshold={dynamic_threshold:.3f}  "
                            f"optimal_log_prob={optimal_score:.3f}")
            if show:
                print(f"    {progress_msg}", flush=True)
            if log_fn is not None:
                log_fn(progress_msg)
        if steps_without_improve >= optimizer_patience:
            stop = "early"
            break

    optimal_np = optimal_theta.to(vista_device).numpy()
    stop_msg = (f"{stop} stop at step {step} ({time.time() - t0:.3f}s)  "
                f"optimal_log_prob={optimal_score:.3f}")
    if show:
        print(f"  [optim] {stop_msg}", flush=True)
        print(f"          optimal theta [LOG] {_theta_repr(optimal_np)}", flush=True)
        print(f"          optimal theta [ABS] {_theta_repr(np.power(10.0, optimal_np))}",
              flush=True)
    if log_fn is not None:
        log_fn(stop_msg)
    return optimal_score, optimal_np


def MAPEstimate(posterior, video_chunk: np.ndarray,
                train_device: torch.device, vista_device: torch.device,
                theta_prex_size: int, theta_prex_batch_size: int,
                score_prex_batch_size: int, elite_prex_size: int,
                numb_steps: int, optimizer_patience: int,
                scheduler_patience: int, show_progress_steps: int,
                learning_rate_minimum: float, learning_rate_factor: float,
                learning_rate: float, tolerance: float,
                pool_mode: str = "bounded", show: bool = False,
                verbose: bool = False, log_fn=None) -> tuple:
    """Estimate the MAP theta for one video via seed-then-optimize.

    Args:
        posterior: a trained ``DirectPosterior`` (its ``posterior_estimator`` is
            the gradient-enabled flow used for scoring and optimization).
        video_chunk: a single raw video array ``(n_frames, H, W)``; normalized to
            ``[0, 1]`` internally to match the training-time input.
        train_device / vista_device: compute device and CPU offload device.
        pool_mode: ``"bounded"`` (rejection-sample the candidate pool within the
            prior) or ``"unrestricted"`` (sample the flow directly); see
            ``collect_theta_prex``.
        show: when True, print the per-stage diagnostics (pool/score/
            elite shapes and timings, per-step optimization progress, the stopping
            reason, and the optimal theta in log10 and physical units).
        verbose: deeper level (implies show at the call site): enables sbi sampling
            progress bars (bounded pool) and the optimizer-configuration line.
        log_fn: optional sink called with each per-step optimization-progress line
            and the stop line, for live streaming to a file (e.g. the debug log).
        (remaining args): seed-then-optimize hyperparameters; see
            ``InferenceEvaluation`` for meanings and defaults.

    Returns:
        ``(optimal_score, optimal_theta)`` -- the log-probability and the inferred
        MAP theta (1D numpy array, log10 space).
    """
    omega = torch.tensor(normalize_video(video_chunk), dtype=torch.float32,
                         device=train_device)
    cond = omega.unsqueeze(0)                                      # mount batch dim
    posterior.set_default_x(cond)
    flow = posterior.posterior_estimator

    t0 = time.time()
    # Candidate pool: drawn via the bounded posterior, which embeds the video
    # itself; leave it on the raw conditioning.
    theta_prex = collect_theta_prex(posterior, flow, vista_device, cond,
                                    theta_prex_size, theta_prex_batch_size,
                                    pool_mode, show, verbose)
    # Embed the conditioning video ONCE, then swap the flow's embedding net for an
    # identity so scoring + optimization reuse the cached latent instead of re-running
    # the (expensive) Complex3DCNN + TemporalTransformer forward per candidate / per
    # gradient step. Behavior-preserving: the log-prob values are identical -- only the
    # embedding is hoisted out of the inner loops. The cached latent stands in for
    # `cond`; collect_score_prex / optimize_elite expand it exactly as before.
    with torch.no_grad():
        emb_x = flow.embedding_net(cond).detach()
    embed_holder = flow.net if (hasattr(flow, "net")
                                and hasattr(flow.net, "_embedding_net")) else flow
    original_embedding_net = embed_holder._embedding_net
    original_condition_shape = flow.condition_shape
    embed_holder._embedding_net = torch.nn.Identity()
    flow._condition_shape = tuple(emb_x.shape[1:])   # condition is now the cached latent (backing attr; the public one is read-only)
    try:
        score_prex = collect_score_prex(flow, train_device, vista_device, emb_x,
                                        theta_prex, score_prex_batch_size, show)
        elite_prex = extract_elite_prex(theta_prex, score_prex, elite_prex_size, show)
        optimal_score, optimal_theta = optimize_elite(
            flow, train_device, vista_device, emb_x, elite_prex,
            numb_steps, optimizer_patience, scheduler_patience, show_progress_steps,
            learning_rate_minimum, learning_rate_factor, learning_rate, tolerance,
            show, verbose, log_fn,
        )
    finally:
        embed_holder._embedding_net = original_embedding_net
        flow._condition_shape = original_condition_shape
    if show:
        print(f"  [MAP  ] process time {time.time() - t0:.3f}s", flush=True)
    return optimal_score, optimal_theta


def posterior_summary(posterior, video_chunk: np.ndarray,
                      train_device: torch.device, vista_device: torch.device,
                      n_samples: int, theta_prex_batch_size: int,
                      pool_mode: str = "bounded",
                      quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)) -> np.ndarray:
    """Per-parameter posterior quantile summary for one observation (View B).

    Complements the MAP point estimate: draws ``n_samples`` from the posterior
    conditioned on the video (via the same two-mode sampler as the candidate pool
    -- ``bounded`` rejection sampling or ``unrestricted`` direct flow sampling) and
    returns the requested quantiles of each parameter, i.e. the posterior credible
    summary (median + IQR, optionally outer quantiles). This captures the
    *within-observation* posterior uncertainty that the MAP mode discards.

    Returns:
        ``(D, len(quantiles))`` numpy array (log10 space): per parameter, the
        sampled quantiles in the order given by ``quantiles``.
    """
    omega = torch.tensor(normalize_video(video_chunk), dtype=torch.float32,
                         device=train_device)
    cond = omega.unsqueeze(0)
    posterior.set_default_x(cond)
    flow = posterior.posterior_estimator
    samples = collect_theta_prex(posterior, flow, vista_device, cond,
                                 n_samples, theta_prex_batch_size, pool_mode)
    arr = samples.detach().cpu().numpy()                       # (n_samples, D)
    return np.quantile(arr, list(quantiles), axis=0).T         # (D, len(quantiles))


# =============================================================================
# Recovery statistics (true vs. inferred, log10 space)
# =============================================================================

def recovery_stats(true_log10: np.ndarray, inferred_log10: np.ndarray,
                   guide: float = 0.3) -> list:
    """Per-parameter recovery error statistics (in log10 units).

    Args:
        true_log10: ground-truth theta in log10, shape ``(N, D)``.
        inferred_log10: inferred MAP theta in log10, shape ``(N, D)``.
        guide: half-width of the "within guide" band (log10 units).

    Returns:
        A list of ``D`` dicts, one per parameter, each with keys: ``n``,
        ``median_error``, ``mae``, ``rmse``, ``q95_abs_error``,
        ``frac_within_guide`` (computed over the finite (true, inferred) pairs).
    """
    true_log10 = np.asarray(true_log10, dtype=float)
    inferred_log10 = np.asarray(inferred_log10, dtype=float)
    stats = []
    for i in range(true_log10.shape[1]):
        x = true_log10[:, i]
        y = inferred_log10[:, i]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        error = y - x
        if error.size:
            abs_error = np.abs(error)
            stats.append({
                "n": int(error.size),
                "median_error": float(np.median(error)),
                "mae": float(np.mean(abs_error)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "q95_abs_error": float(np.quantile(abs_error, 0.95)),
                "frac_within_guide": float(np.mean(abs_error <= guide)),
            })
        else:
            stats.append({"n": 0, "median_error": float("nan"),
                          "mae": float("nan"), "rmse": float("nan"),
                          "q95_abs_error": float("nan"),
                          "frac_within_guide": float("nan")})
    return stats


def experiment_table(parameterization, inferred_by_kind: dict, kinds) -> tuple:
    """Build a ``(headers, rows)`` summary of inferred theta per condition.

    For real microscopy data there is no ground truth, so this reports, per
    learnable parameter and per condition (kind), the distribution of the
    inferred MAP theta: count, median and IQR in log10, and the median in
    physical units. ``inferred_by_kind`` maps each kind to an ``(N, D)`` array
    of inferred log10 theta (N = cells x chunks for that kind).
    """
    headers = ["parameter", "label", "kind", "n", "median log10",
               "IQR log10", "median value"]
    rows = []
    for i, para in enumerate(parameterization):
        for kind in kinds:
            arr = np.asarray(inferred_by_kind.get(kind, []), dtype=float)
            col = arr[:, i] if arr.ndim == 2 and arr.size else np.array([])
            col = col[np.isfinite(col)]
            if col.size:
                med = float(np.median(col))
                q1, q3 = np.quantile(col, [0.25, 0.75])
                rows.append([para["KEY"], para.get("LABEL") or "-", kind,
                             str(col.size), f"{med:+.3f}", f"{q3 - q1:.3f}",
                             f"{10 ** med:.4g}"])
            else:
                rows.append([para["KEY"], para.get("LABEL") or "-", kind,
                             "0", "nan", "nan", "nan"])
    return headers, rows


def posterior_coverage_table(parameterization, true_log10: np.ndarray,
                             post_q: np.ndarray) -> tuple:
    """Posterior calibration summary per parameter (View B, recovery).

    ``post_q`` is an ``(N, D, 5)`` array of per-observation posterior quantiles
    ``[Q05, Q25, Q50, Q75, Q95]`` (log10). Reports, per parameter, the fraction
    of ground-truths falling inside the 50% (IQR) and 90% (Q05-Q95) posterior
    credible intervals -- a well-calibrated posterior covers ~50% and ~90%
    respectively -- plus the median absolute bias of the posterior median.
    """
    headers = ["parameter", "label", "n", "cover@50% (IQR)", "cover@90%",
               "median |median-true|"]
    true_log10 = np.asarray(true_log10, dtype=float)
    rows = []
    for i, para in enumerate(parameterization):
        t = true_log10[:, i]
        q = post_q[:, i, :]
        mask = np.isfinite(t) & np.all(np.isfinite(q), axis=1)
        t, q = t[mask], q[mask]
        if t.size:
            c50 = float(np.mean((t >= q[:, 1]) & (t <= q[:, 3])))
            c90 = float(np.mean((t >= q[:, 0]) & (t <= q[:, 4])))
            bias = float(np.median(np.abs(q[:, 2] - t)))
            rows.append([para["KEY"], para.get("LABEL") or "-", str(t.size),
                         f"{c50 * 100:.0f}%", f"{c90 * 100:.0f}%", f"{bias:.3f}"])
        else:
            rows.append([para["KEY"], para.get("LABEL") or "-", "0", "-", "-", "-"])
    return headers, rows


def recovery_table(parameterization, true_log10: np.ndarray,
                   inferred_log10: np.ndarray, guide: float = 0.3) -> tuple:
    """Build a ``(headers, rows)`` recovery summary for the diagnostic report.

    One row per learnable parameter, with the error statistics from
    ``recovery_stats``. ``parameterization`` is the learnable-only
    ``PARAMETERIZATION`` list (its order matches the theta columns).
    """
    headers = ["parameter", "label", "n", "median err", "MAE", "RMSE",
               "q95|err|", f"within +/-{guide:g}"]
    stats = recovery_stats(true_log10, inferred_log10, guide)
    rows = []
    for para, st in zip(parameterization, stats):
        rows.append([
            para["KEY"],
            para.get("LABEL") or "-",
            str(st["n"]),
            f"{st['median_error']:+.3f}",
            f"{st['mae']:.3f}",
            f"{st['rmse']:.3f}",
            f"{st['q95_abs_error']:.3f}",
            f"{st['frac_within_guide'] * 100:.0f}%",
        ])
    return headers, rows
