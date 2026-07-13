"""Real-versus-synthetic domain distance in the learned embedding space.

Part of the Detector calibration workflow (see the quantitative real-versus-synthetic
distance in DETECTOR_WORKFLOW.md). Quantifies how far real microscopy recordings sit
from imaging-parameter-matched synthetic videos, measured in the trained
``Complex3DCNN`` embedding space, via two complementary two-sample statistics:

  - Maximum Mean Discrepancy (MMD; Gretton et al. 2012) -- a kernel two-sample
    distance with an RBF kernel (median-heuristic bandwidth) and a permutation
    p-value. Larger means a bigger gap.
  - Classifier Two-Sample Test (C2ST; Lopez-Paz & Oquab 2017) -- the accuracy of a
    classifier trained to tell real from synthetic. ~0.5 means indistinguishable (a
    small gap); ->1.0 means easily separable (a large gap).

Both resample at the CELL level: video chunks within one recording are correlated,
not independent, so the significance test permutes / splits whole cells rather than
individual chunks. A sample with no cell label is treated as its own block (i.e.
independent), which is the correct treatment for synthetic videos.

Pure analysis: numpy + torch (for the embedding forward pass); scikit-learn and scipy
are imported lazily inside ``c2st`` so importing this module is cheap and its MMD path
carries no hard sklearn dependency. It imports nothing from ``parameterization`` /
``artifacts``, so it is unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Embedding
# =============================================================================

def embed_videos(posterior, videos, device=None, *, batch_size=16,
                 expected_frames=None, to_numpy=True):
    """Embed videos through the trained ``Complex3DCNN`` into CLS-token vectors.

    Args:
        posterior: a trained ``DirectPosterior`` (its ``posterior_estimator`` is the
            flow whose ``embedding_net`` is the ``Complex3DCNN``), or the flow itself.
        videos: sequence/array of ``(n_frames, H, W)`` videos (raw integer or already
            normalized); each is passed through ``normalize_video`` before embedding,
            the same preprocessing the training/evaluation path uses.
        device: torch device to embed on; defaults to the flow's own device.
        batch_size: videos embedded per forward pass (bounds GPU memory).
        expected_frames: if given, assert every video's frame axis equals it (guards a
            train/eval window-length mismatch).
        to_numpy: return a numpy array (default) or a CPU tensor.

    Returns:
        ``[N, D]`` embeddings (``D`` read from the network output, never hardcoded).
    """
    from .inference_support import normalize_video   # lazy: keeps the MMD/C2ST math import-light

    flow = getattr(posterior, "posterior_estimator", posterior)
    flow.eval()
    dev = torch.device(device) if device is not None else next(flow.parameters()).device
    out = []
    n = len(videos)
    for i in range(0, n, batch_size):
        batch = videos[i:i + batch_size]
        arr = np.stack([normalize_video(np.asarray(v)) for v in batch])   # (b, T, H, W) float32 [0,1]
        if expected_frames is not None and arr.shape[1] != expected_frames:
            raise ValueError(
                f"video frame axis {arr.shape[1]} != expected_frames {expected_frames}; "
                f"real and synthetic videos must be diced to the estimator's window.")
        x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        with torch.no_grad():
            emb = flow.embedding_net(x)                                   # [b, D] CLS embedding
        out.append(emb.detach().to("cpu"))
    emb_all = torch.cat(out, dim=0)
    return emb_all.numpy() if to_numpy else emb_all


# =============================================================================
# Cell-level block bookkeeping
# =============================================================================

def _blocks(n_x, n_y, cells_x, cells_y):
    """Return a per-sample integer block id for the pooled [X; Y] stack.

    A sample with a cell label shares a block with the other samples of that cell
    (real chunks within a recording); a sample with no label becomes its own block
    (independent -- the correct treatment for synthetic videos). Real and synthetic
    cell-id namespaces are kept disjoint so a real cell never merges with a synthetic
    one.
    """
    blocks = np.empty(n_x + n_y, dtype=object)
    for off, cnt, cells, tag in ((0, n_x, cells_x, "x"), (n_x, n_y, cells_y, "y")):
        if cells is None:
            for j in range(cnt):
                blocks[off + j] = (tag, "iid", j)
        else:
            cells = np.asarray(cells)
            for j in range(cnt):
                blocks[off + j] = (tag, "cell", cells[j])
    # map to contiguous integer ids
    uniq = {b: i for i, b in enumerate(dict.fromkeys(blocks.tolist()))}
    return np.array([uniq[b] for b in blocks.tolist()], dtype=int)


# =============================================================================
# MMD
# =============================================================================

def _sq_dists(Z):
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.maximum(d2, 0.0)


def _mmd2_unbiased(K, n):
    """Unbiased squared-MMD from a pooled Gram matrix ``K`` whose first ``n`` rows are X."""
    m = K.shape[0] - n
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
    sxy = Kxy.mean()
    return float(sxx + syy - 2.0 * sxy)


def mmd_rbf(X, Y, *, bandwidth="median", n_permutations=1000, seed=None,
            cells_x=None, cells_y=None):
    """Unbiased squared-MMD (RBF kernel) with a cell-level permutation p-value.

    The bandwidth defaults to the median-heuristic (median of pooled squared pairwise
    distances). The permutation test reuses the single precomputed Gram matrix and
    permutes at the BLOCK level (whole cells; see ``_blocks``), so correlated chunks
    move together and the null respects the correlation structure.

    Returns a dict: ``mmd2``, ``mmd`` (=sqrt(max(mmd2,0))), ``sigma``, ``gamma``,
    ``p_value``, ``n``, ``m``, ``n_permutations``.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y]); L = n + m
    D2 = _sq_dists(Z)
    if bandwidth == "median":
        med = np.median(D2[np.triu_indices(L, k=1)])
        sigma2 = float(med) if med > 0 else 1.0
    else:
        sigma2 = float(bandwidth) ** 2
    gamma = 1.0 / (2.0 * sigma2)
    K = np.exp(-gamma * D2)
    observed = _mmd2_unbiased(K, n)

    block = _blocks(n, m, cells_x, cells_y)
    uniq_blocks = np.unique(block)
    # each block belongs wholly to X or Y in the observed split (X = indices < n)
    block_is_x = np.array([np.all(np.where(block == b)[0] < n) for b in uniq_blocks])
    n_x_blocks = int(block_is_x.sum())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(uniq_blocks))
        x_blocks = set(uniq_blocks[perm[:n_x_blocks]])
        mask_x = np.array([block[i] in x_blocks for i in range(L)])
        order = np.concatenate([np.where(mask_x)[0], np.where(~mask_x)[0]])
        Kp = K[np.ix_(order, order)]
        if _mmd2_unbiased(Kp, int(mask_x.sum())) >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return {"mmd2": observed, "mmd": float(np.sqrt(max(observed, 0.0))),
            "sigma": float(np.sqrt(sigma2)), "gamma": float(gamma),
            "p_value": float(p), "n": n, "m": m, "n_permutations": int(n_permutations)}


# =============================================================================
# C2ST
# =============================================================================

def c2st(X, Y, *, classifier="mlp", n_folds=5, standardize=True, seed=None,
         cells_x=None, cells_y=None):
    """Classifier two-sample test with cell-aware (grouped) cross-validation.

    Trains a classifier to separate X (label 0) from Y (label 1) under k-fold CV; if
    cell labels are given, uses ``GroupKFold`` so a cell's chunks never split across
    train/test. Reports mean out-of-fold accuracy (null = 0.5), across-fold spread and
    a 95% CI, and a **cell-aware** p-value: a one-sided t-test of the per-fold
    accuracies against 0.5. Because the folds are grouped by cell, each fold-accuracy is
    an out-of-cell estimate, so the test respects the cell correlation (it does NOT pool
    per-chunk predictions as independent Bernoulli trials, which would be
    anti-conservative). Conservative with few cells (few folds).

    ``classifier``: ``"mlp"`` (sbi-style two-layer MLP), ``"rf"`` (random forest), or
    ``"logreg"``.
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import ttest_1samp

    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y]); y = np.concatenate([np.zeros(n), np.ones(m)]).astype(int)
    D = Z.shape[1]
    groups = None
    if cells_x is not None or cells_y is not None:
        gx = [("x", c) for c in (cells_x if cells_x is not None else range(n))]
        gy = [("y", c) for c in (cells_y if cells_y is not None else range(m))]
        gmap = {g: i for i, g in enumerate(dict.fromkeys(gx + gy))}
        groups = np.array([gmap[g] for g in (gx + gy)])

    def _make():
        if classifier == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(10 * D, 10 * D),
                                 max_iter=1000, random_state=seed)
        if classifier == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=seed)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)

    if groups is not None:
        n_splits = min(n_folds, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits).split(Z, y, groups)
    else:
        n_splits = n_folds
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(Z, y)

    accs = []
    for tr, te in splitter:
        Ztr, Zte = Z[tr], Z[te]
        if standardize:
            sc = StandardScaler().fit(Ztr)
            Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        clf = _make().fit(Ztr, y[tr])
        accs.append(float(np.mean(clf.predict(Zte) == y[te])))
    accs = np.array(accs)
    acc = float(accs.mean())
    sd = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(accs)) if len(accs) > 1 else 0.0
    # Cell-aware significance: one-sided t-test on the per-fold (cell-grouped) accuracies
    # vs 0.5 -- respects cell correlation (each fold-accuracy is out-of-cell), unlike a
    # per-chunk binomial that would treat correlated chunks as independent trials.
    p = (float(ttest_1samp(accs, 0.5, alternative="greater").pvalue)
         if len(accs) >= 2 else float("nan"))
    return {"accuracy": acc, "accuracy_std": sd, "per_fold": accs.tolist(),
            "ci95": (acc - half, acc + half), "null": 0.5, "p_value": p,
            "classifier": classifier, "n_folds": int(n_splits), "n": n, "m": m}


# =============================================================================
# Orchestrator + report table
# =============================================================================

def compute_distance(posterior, real_videos, synth_videos, *, device=None,
                     batch_size=16, expected_frames=None, real_cells=None,
                     synth_cells=None, mmd_kwargs=None, c2st_kwargs=None, seed=None):
    """Embed both sets and run MMD + C2ST; returns a combined dict incl. the embeddings.

    ``real_cells`` is REQUIRED: the real recordings are cell-correlated (chunks within a
    recording are not independent), so the cell-level resampling needs per-video cell
    labels; omitting them would silently degrade to an iid (anti-conservative) test.
    ``synth_cells`` may be None — synthetic videos are independent draws (each its own
    block).
    """
    if real_cells is None:
        raise ValueError(
            "compute_distance requires real_cells (one cell/recording label per real video): "
            "real chunks within a recording are correlated, so the cell-level resampling that "
            "§8 specifies needs them; omitting them would silently treat correlated chunks as "
            "independent. Pass real_cells=<per-video cell id>. (synth_cells may be None.)")
    er = embed_videos(posterior, real_videos, device=device, batch_size=batch_size,
                      expected_frames=expected_frames)
    es = embed_videos(posterior, synth_videos, device=device, batch_size=batch_size,
                      expected_frames=expected_frames)
    mmd = mmd_rbf(er, es, cells_x=real_cells, cells_y=synth_cells, seed=seed,
                  **(mmd_kwargs or {}))
    c = c2st(er, es, cells_x=real_cells, cells_y=synth_cells, seed=seed,
             **(c2st_kwargs or {}))
    return {"mmd": mmd, "c2st": c, "n_real": len(er), "n_synth": len(es),
            "embed_dim": int(er.shape[1]), "emb_real": er, "emb_synth": es}


def distance_table(result):
    """Render a ``compute_distance`` result as ``(headers, rows)`` for the reporter."""
    mmd, c = result["mmd"], result["c2st"]
    headers = ["metric", "value", "null / ref", "p-value", "reading"]
    rows = [
        ["MMD (unbiased sq.)", f"{mmd['mmd2']:.4g}", "0 (identical)",
         f"{mmd['p_value']:.3g}", "larger => bigger gap"],
        ["MMD (sqrt)", f"{mmd['mmd']:.4g}", f"RBF sigma={mmd['sigma']:.3g}", "", ""],
        ["C2ST accuracy", f"{c['accuracy']:.3f}",
         f"0.5; CI[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]", f"{c['p_value']:.3g}",
         "closer to 0.5 => smaller gap"],
    ]
    return headers, rows
