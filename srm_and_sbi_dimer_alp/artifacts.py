"""Self-describing, version-portable estimator artifact (DETECTOR_WORKFLOW.md §7, A5).

The canonical `save_posterior` pickles the live `DirectPosterior`; because the
embedding is `torch.compile`d before training, that pickle bakes in
`torch._dynamo` internals whose private layouts change between torch releases —
the artifact is torch-version-locked. This module persists an estimator as three
**separable** components instead, so it reconstructs under whatever torch version
is loading it. The canonical `save_posterior`/`load_posterior` path is untouched.

Three components (one `.npz`):
  (a) a compile-stripped `state_dict` — tensor weights only, the `_orig_mod.`
      compile prefix removed, serialized `weights_only`-loadably;
  (b) a rebuild spec — the `Complex3DCNN` args, the `build_maf` args, and the
      `theta_dim`/`video_shape` hints `build_maf` needs for shape inference,
      captured exactly as the caller constructed the estimator (Approach 1: the
      caller supplies them, so there is no fragile introspection and no canonical
      edit);
  (c) a metadata block — the ordered `parameter_keys` mapped to prior bounds,
      plus provenance (torch version, weights checksum, and caller metadata such
      as the timing label, best test loss, and source conditions/accessions).

`load_estimator` rebuilds the `Complex3DCNN` **uncompiled** under the current
torch, replays `build_maf` on zero-cost dummy batches of the stored shapes, loads
the stripped weights (`weights_only=True`), verifies the checksum, and attaches a
fresh device-aware prior — never deserializing any torch-internal or compiled
code. It returns a `DirectPosterior`, the same type every canonical consumer
accepts. This mirrors the existing Construction rebuild pattern (Complex3DCNN +
build_maf + load_state_dict), minus its re-`torch.compile` requirement.
"""

import hashlib
import io
import json

import numpy as np
import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_maf
from sbi.utils import BoxUniform

from .inference_network import Complex3DCNN

ARTIFACT_FORMAT_VERSION = 1
_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(state_dict: dict) -> dict:
    """Return a copy of ``state_dict`` with the `torch.compile` ``_orig_mod.``
    prefix removed from every key (so the weights match an uncompiled rebuild)."""
    return {key.replace(_COMPILE_PREFIX, ""): value for key, value in state_dict.items()}


def _rebuild_estimator(embedding_args, maf_args, theta_dim, video_shape, device):
    """Reconstruct the (uncompiled) MAF estimator from the rebuild spec."""
    embedding = Complex3DCNN(**embedding_args)
    # Dummy batches size the flow (build_maf infers dims from them); their values
    # are irrelevant — the trained standardizing stats are restored by
    # load_state_dict. A fixed seed keeps the reconstruction deterministic and
    # gives the standardizing net non-degenerate variance during construction.
    generator = torch.Generator().manual_seed(0)
    batch_x = torch.randn((2, theta_dim), generator=generator)
    batch_y = torch.randn((2, *video_shape), generator=generator)
    estimator = build_maf(batch_x=batch_x, batch_y=batch_y,
                          embedding_net=embedding, **maf_args)
    return estimator.to(device)


def save_estimator(estimator, *, embedding_args, maf_args, theta_dim, video_shape,
                   parameter_keys, prior_low, prior_high, path, metadata=None):
    """Persist a trained estimator as a self-describing, version-portable artifact.

    Args:
        estimator: the trained MAF estimator (the `build_maf` output, whose
            embedding may be `torch.compile`d).
        embedding_args: the exact kwargs used to build the `Complex3DCNN`.
        maf_args: the exact `build_maf` kwargs (excluding `batch_x`/`batch_y`/
            `embedding_net`), e.g. `z_score_x`, `z_score_y`, `dropout_probability`,
            `use_batch_norm`.
        theta_dim: parameter-vector dimension (columns of `batch_x`).
        video_shape: per-example video shape (e.g. `(n_frames, H, W)`).
        parameter_keys: ordered learnable-parameter keys.
        prior_low, prior_high: prior bounds aligned with `parameter_keys`.
        path: output `.npz` path.
        metadata: optional provenance dict (timing label, best test loss,
            source conditions/accessions, ...).
    """
    stripped = {key: value.detach().cpu()
                for key, value in strip_compile_prefix(estimator.state_dict()).items()}
    buffer = io.BytesIO()
    torch.save(stripped, buffer)
    weights = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    checksum = hashlib.sha256(weights.tobytes()).hexdigest()

    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "rebuild_spec": {
            "embedding_class": "Complex3DCNN",
            "embedding_args": embedding_args,
            "maf_builder": "build_maf",
            "maf_args": maf_args,
            "theta_dim": int(theta_dim),
            "video_shape": list(video_shape),
        },
        "parameter_keys": list(parameter_keys),
        "state_dict_keys": list(stripped.keys()),
        "torch_version": torch.__version__,
        "weights_sha256": checksum,
        "metadata": dict(metadata or {}),
    }
    np.savez_compressed(
        path,
        weights=weights,
        prior_low=np.asarray(prior_low, dtype=float),
        prior_high=np.asarray(prior_high, dtype=float),
        manifest=np.array(json.dumps(manifest)),
    )


def load_estimator(path, device: str = "cpu") -> DirectPosterior:
    """Rebuild a `DirectPosterior` from a `save_estimator` artifact, version-portably.

    Reconstructs the uncompiled estimator from the rebuild spec under the current
    torch, loads the compile-stripped weights (`weights_only=True`), verifies the
    checksum, and attaches a fresh device-aware `BoxUniform` prior. No
    torch-internal or compiled code is deserialized.
    """
    data = np.load(path, allow_pickle=False)
    manifest = json.loads(str(data["manifest"]))
    spec = manifest["rebuild_spec"]

    weights = data["weights"]
    if hashlib.sha256(weights.tobytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"weights checksum mismatch loading estimator artifact {path}")
    state_dict = torch.load(io.BytesIO(weights.tobytes()), weights_only=True)

    estimator = _rebuild_estimator(
        spec["embedding_args"], spec["maf_args"],
        spec["theta_dim"], tuple(spec["video_shape"]), device,
    )
    estimator.load_state_dict(state_dict)
    estimator.eval()

    prior = BoxUniform(
        low=torch.tensor(data["prior_low"], dtype=torch.float32),
        high=torch.tensor(data["prior_high"], dtype=torch.float32),
        device=device,
    )
    return DirectPosterior(estimator, prior)
