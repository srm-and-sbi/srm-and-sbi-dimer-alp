"""Per-example test-loss distribution artifact for the Inference stage.

The reported per-epoch test loss is a *mean*; two estimators with the same mean
can differ in the spread and tails of the per-example loss, which is where
generalization behavior lives. This module holds, for the best epoch of a run,
the per-example loss over the held-out TEST set, keyed by the stable identifier
``(task_index, sim_index)``, alongside the parameter vectors and a fully
self-describing manifest (parameter table + run provenance). It is one
best-epoch snapshot — treated identically to the posterior / checkpoint (the
canonical is overwritten when a better epoch is found; the run only keeps its
best) — written to a compressed ``.npz`` that post-hoc analysis reads to compare
estimators beyond the mean: paired comparison of two runs' best snapshots on a
shared ``(task_index, sim_index)`` subset, loss-distribution / tail statistics,
and the central-limit-assumption checks.

The class is deliberately pure (numpy + json only, no torch and no distributed
primitives): the caller collects the per-example losses from the validation
pass, gathers and de-duplicates them across ranks, and hands plain arrays to
:meth:`TestLossDistribution.from_epoch`. That keeps this container unit-testable
in isolation and reusable regardless of the training backend.

See ``ESTIMATOR_GENERALIZATION_METHODS.md`` for the design and the naming rule.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

# Bumped only on a breaking change to the on-disk layout; readers check it.
ARTIFACT_FORMAT_VERSION = 1

# Fixed quantiles reported in the extended stats card.
_CARD_QUANTILES = (50.0, 90.0, 99.0)


def _skewness(values: np.ndarray) -> float:
    """Fisher-Pearson skewness of a 1-D array (0.0 for fewer than two points or
    a degenerate spread). A right (positive) skew flags the heavy upper tail of
    catastrophic per-example losses that the mean conceals."""
    if values.size < 2:
        return 0.0
    centered = values - values.mean()
    variance = np.mean(centered ** 2)
    if variance <= 0.0:
        return 0.0
    return float(np.mean(centered ** 3) / variance ** 1.5)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, level: float,
                       seed: int) -> tuple:
    """Percentile bootstrap confidence interval for the mean.

    Uses a fixed local seed so the interval is reproducible from the stored
    losses; this randomness is a post-hoc summary of already-recorded data and
    is unrelated to the (seedless) generative pipeline. Returns ``(lo, hi)``, or
    ``(nan, nan)`` when there are too few points or ``n_boot <= 0``.
    """
    n = values.size
    if n < 2 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, size=n)].mean()
    tail = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return (float(lo), float(hi))


def loss_summary(loss: np.ndarray) -> dict:
    """Cheap per-epoch stats (mean, std, min, max, n) over finite losses.

    Standalone so the training loop can log a spread each epoch without building
    a snapshot; :meth:`TestLossDistribution.summary` reuses it for the best epoch.
    """
    values = np.asarray(loss, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite losses to summarize.")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


class TestLossDistribution:
    """One best-epoch snapshot of the per-example test loss.

    Holds parallel arrays over the fixed held-out TEST set — the stable
    ``(task_index, sim_index)`` keys, the parameter vectors ``theta``, and the
    per-example ``loss`` — plus a self-describing ``manifest``. Constructed from
    an epoch's evaluated losses via :meth:`from_epoch`; written / reloaded as a
    ``.npz`` via :meth:`flush` / :meth:`load`.
    """

    def __init__(self, keys: np.ndarray, theta: Optional[np.ndarray],
                 loss: np.ndarray, manifest: dict):
        self.keys = np.asarray(keys)                 # (n_test, 2) int64
        self.theta = theta                           # (n_test, n_params) or None
        self.loss = np.asarray(loss, dtype=np.float64)   # (n_test,)
        self.manifest = dict(manifest)

    @classmethod
    def from_epoch(cls, epoch: int, task_index, sim_index, loss,
                   theta=None, manifest: Optional[dict] = None,
                   best_test_loss: Optional[float] = None) -> "TestLossDistribution":
        """Build a snapshot from one epoch's per-example losses.

        ``task_index``, ``sim_index``, ``loss`` (and ``theta``) are parallel
        arrays over the evaluated TEST examples, in any order. Keys are sorted
        and **de-duplicated** (a repeated key from a distributed sampler's
        boundary padding collapses to one row, with its first-seen loss/theta),
        so the snapshot always has one row per distinct example. ``epoch`` and
        ``best_test_loss`` are stamped into the manifest.
        """
        task_index = np.asarray(task_index).astype(np.int64).ravel()
        sim_index = np.asarray(sim_index).astype(np.int64).ravel()
        loss = np.asarray(loss, dtype=np.float64).ravel()
        if not (task_index.size == sim_index.size == loss.size):
            raise ValueError(
                f"task_index ({task_index.size}), sim_index ({sim_index.size}) "
                f"and loss ({loss.size}) must have equal length.")
        if theta is not None:
            theta = np.asarray(theta, dtype=np.float64)
            if theta.shape[0] != task_index.size:
                raise ValueError(
                    f"theta has {theta.shape[0]} rows; expected "
                    f"{task_index.size} (one per example).")

        # np.unique(axis=0) returns rows sorted lexicographically (task, then
        # sim) with the first-occurrence index of each unique key.
        pairs = np.stack([task_index, sim_index], axis=1)
        keys, first_idx = np.unique(pairs, axis=0, return_index=True)
        loss = loss[first_idx]
        theta = theta[first_idx] if theta is not None else None

        manifest = dict(manifest or {})
        manifest["epoch"] = int(epoch)
        if best_test_loss is not None:
            manifest["best_test_loss"] = float(best_test_loss)
        return cls(keys=keys, theta=theta, loss=loss, manifest=manifest)

    # -- statistics --------------------------------------------------------
    def _finite(self) -> np.ndarray:
        values = self.loss[np.isfinite(self.loss)]
        if values.size == 0:
            raise ValueError("snapshot has no finite losses.")
        return values

    def summary(self) -> dict:
        """Cheap stats (mean, std, min, max, n) for the live training log."""
        return loss_summary(self.loss)

    def extended_card(self, train_mean: Optional[float] = None,
                      tail_threshold: Optional[float] = None,
                      n_boot: int = 1000, ci_level: float = 0.95,
                      boot_seed: int = 0) -> dict:
        """Fuller distribution summary, shown when a new best is reached:
        quantiles, skewness, tail mass, the train-test mean gap (if given), and
        a bootstrap confidence interval on the mean.

        tail_threshold: a FIXED loss value; the card reports ``tail_mass``, the
            fraction of per-example losses above it (the heavy-upper-tail
            'catastrophic miss' rate). Keep it fixed across epochs and runs so
            the tail mass stays comparable; ``None`` leaves ``tail_mass`` null.
        """
        values = self._finite()
        p50, p90, p99 = np.percentile(values, _CARD_QUANTILES)
        ci_lo, ci_hi = _bootstrap_mean_ci(values, n_boot, ci_level, boot_seed)
        tail_mass = (float(np.mean(values > tail_threshold))
                     if tail_threshold is not None else None)
        card = {
            "epoch": self.manifest.get("epoch"),
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(p50),
            "p90": float(p90),
            "p99": float(p99),
            "max": float(values.max()),
            "skew": _skewness(values),
            "tail_threshold": tail_threshold,
            "tail_mass": tail_mass,
            "mean_ci95": [ci_lo, ci_hi],
        }
        if train_mean is not None:
            # Positive => the held-out (test) loss exceeds the train loss.
            card["train_test_gap"] = float(values.mean() - train_mean)
        return card

    # -- persistence -------------------------------------------------------
    def flush(self, path, backup_path=None) -> None:
        """Write the snapshot to ``path`` as a compressed ``.npz`` and, if given,
        copy it to ``backup_path`` (cheaper than serializing twice)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.manifest)
        manifest["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
        theta = (self.theta if self.theta is not None
                 else np.empty((self.keys.shape[0], 0), dtype=np.float64))
        np.savez_compressed(
            str(path),
            keys=self.keys,
            theta=theta,
            loss=self.loss,
            manifest=np.asarray(json.dumps(manifest)),
        )
        if backup_path is not None:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(backup_path))

    @classmethod
    def load(cls, path) -> "TestLossDistribution":
        """Reload a flushed snapshot for post-hoc analysis."""
        with np.load(str(path), allow_pickle=False) as data:
            theta = data["theta"]
            return cls(
                keys=data["keys"],
                theta=theta if theta.shape[1] > 0 else None,
                loss=data["loss"],
                manifest=json.loads(str(data["manifest"])),
            )
