"""Nuisance artifact: a samplable, self-describing marginalized parameter block.

Part of the Detector calibration workflow (DETECTOR_WORKFLOW.md §7, A4). A
`Nuisance` supplies draws for a parameter block that is marginalized rather than
inferred, and carries its own `parameter_keys` manifest so a draw is
self-labeling. It serves two roles: `Nuisance_RDS` (the reaction-diffusion
biology marginalized during Detector calibration) and `Nuisance_DLI` (the imaging
block marginalized in the production workflow — one of the three ways the
Detector's imaging output feeds production).

Two knobs:
  1. Never-silent clipping. With `clip_to_prior=True`, draws are clipped to the
     stored box; every `sample` call returns a per-draw `clipped` mask, keeps a
     running `clipped_count`, and logs when clipping occurs — so it is always
     possible to tell which draws (and which training simulations) used a clamped
     value.
  2. Stored distribution numerics. The numeric parameters of the underlying
     distribution are stored in the artifact — the box `low`/`high` for a
     from-spec `BoxUniform`, or the sample matrix for a stored sample-set — so the
     artifact is reconstructible and inspectable.

Draws are returned in the declared `space` ("log10" or "physical"); a from-spec
nuisance over log10 ranges yields log10 draws the consumer exponentiates. The
artifact is pooled across conditions (it carries no per-condition state).
"""

import json
import logging

import numpy as np
import torch
from sbi.utils import BoxUniform

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1


class Nuisance:
    """Samplable, self-describing marginalized parameter block (see module docstring).

    Underlying forms (`kind`):
      - ``"box"``     — a `BoxUniform` over ``[low, high]`` (from ``from_spec``);
      - ``"samples"`` — a stored sample-set, resampled with replacement (from
        ``from_samples`` or, materialized, from ``from_posterior``).
    ``low``/``high`` always define the clip box (and, for ``"box"``, the support).
    """

    def __init__(self, parameter_keys, low, high, space, domain, kind,
                 samples=None, clip_to_prior=True):
        self.parameter_keys = list(parameter_keys)
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.space = space
        self.domain = domain
        self.kind = kind
        self.samples = None if samples is None else np.asarray(samples, dtype=float)
        self.clip_to_prior = clip_to_prior
        self.clipped_count = 0

        d = len(self.parameter_keys)
        if self.low.shape != (d,) or self.high.shape != (d,):
            raise ValueError(
                f"low/high must have shape ({d},) to match parameter_keys; "
                f"got low {self.low.shape}, high {self.high.shape}."
            )
        if kind not in ("box", "samples"):
            raise ValueError(f"kind must be 'box' or 'samples'; got {kind!r}.")
        if kind == "samples":
            if self.samples is None or self.samples.ndim != 2 or self.samples.shape[1] != d:
                raise ValueError(
                    f"a 'samples' nuisance needs a (n, {d}) sample matrix; "
                    f"got {None if self.samples is None else self.samples.shape}."
                )
        self._prior = (BoxUniform(low=torch.tensor(self.low), high=torch.tensor(self.high))
                       if kind == "box" else None)

    # ---- builders --------------------------------------------------------
    @classmethod
    def from_spec(cls, parameter_keys, low, high, space="log10", domain="RDS",
                  clip_to_prior=True):
        """Nuisance backed by a `BoxUniform` over ``[low, high]`` (inline spec)."""
        return cls(parameter_keys, low, high, space, domain, kind="box",
                   clip_to_prior=clip_to_prior)

    @classmethod
    def from_samples(cls, parameter_keys, samples, low, high, space="log10",
                     domain="DLI", clip_to_prior=True):
        """Nuisance backed by a stored sample-set (resampled with replacement)."""
        return cls(parameter_keys, low, high, space, domain, kind="samples",
                   samples=samples, clip_to_prior=clip_to_prior)

    @classmethod
    def from_posterior(cls, parameter_keys, posterior, low, high, n_materialize=10000,
                       x=None, space="log10", domain="DLI", clip_to_prior=True):
        """Materialize a sample-set from a (conditioned) posterior, then store it.

        Draws ``n_materialize`` samples from ``posterior`` (conditioned on ``x`` if
        given), so the persisted artifact is a reconstructible sample-set rather
        than a torch/sbi object. Clipping to the box is applied at sample time.
        """
        with torch.no_grad():
            drawn = (posterior.sample((n_materialize,), x=x) if x is not None
                     else posterior.sample((n_materialize,)))
        return cls.from_samples(parameter_keys, drawn.cpu().numpy(), low, high,
                                space=space, domain=domain, clip_to_prior=clip_to_prior)

    # ---- sampling --------------------------------------------------------
    def _raw(self, n):
        if self.kind == "box":
            return self._prior.sample((n,)).cpu().numpy()
        idx = np.random.default_rng().integers(0, self.samples.shape[0], size=n)
        return self.samples[idx]

    def sample(self, n=None):
        """Draw from the nuisance, returning ``(draws, clipped_mask)``.

        For ``n=None`` returns ``(draw (d,), clipped (bool))``; for an integer
        ``n`` returns ``(draws (n, d), clipped_mask (n,))``. When
        ``clip_to_prior`` is set, draws are clipped to the box, the per-draw mask
        flags which were clamped, ``clipped_count`` accumulates, and a warning is
        logged so clipping is never silent.
        """
        single = n is None
        m = 1 if single else int(n)
        draws = np.asarray(self._raw(m), dtype=float)
        clipped_mask = np.zeros(m, dtype=bool)
        if self.clip_to_prior:
            clamped = np.clip(draws, self.low, self.high)
            clipped_mask = np.any(clamped != draws, axis=1)
            n_clipped = int(clipped_mask.sum())
            if n_clipped:
                self.clipped_count += n_clipped
                logger.warning(
                    "Nuisance[%s]: clipped %d/%d draw(s) to the prior box "
                    "(running total %d).", self.domain, n_clipped, m, self.clipped_count)
            draws = clamped
        if single:
            return draws[0], bool(clipped_mask[0])
        return draws, clipped_mask

    # ---- persistence -----------------------------------------------------
    def manifest(self):
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "parameter_keys": self.parameter_keys,
            "domain": self.domain,
            "space": self.space,
            "kind": self.kind,
            "clip_to_prior": self.clip_to_prior,
            "clipped_count": self.clipped_count,
            "n_samples": (int(self.samples.shape[0]) if self.kind == "samples" else None),
        }

    def flush(self, path):
        """Serialize to ``.npz`` (box low/high + sample matrix if any + manifest)."""
        arrays = {
            "low": self.low,
            "high": self.high,
            "manifest": np.array(json.dumps(self.manifest())),
        }
        if self.kind == "samples":
            arrays["samples"] = self.samples
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        manifest = json.loads(str(data["manifest"]))
        samples = data["samples"] if "samples" in data.files else None
        obj = cls(
            manifest["parameter_keys"], data["low"], data["high"],
            manifest["space"], manifest["domain"], manifest["kind"],
            samples=samples, clip_to_prior=manifest["clip_to_prior"],
        )
        obj.clipped_count = int(manifest.get("clipped_count", 0))
        return obj
