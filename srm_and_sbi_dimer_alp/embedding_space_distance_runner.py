"""Shared engine for the embedding-space distance analysis (biology and detector workflows).

Both workflows ask the same question of a different embedding: *does the trained network place the
REAL recordings where it places its own synthetic training distribution, or somewhere else?* The
embedding is the only representation the posterior ever sees, so a gap there is a gap the inference
cannot recover from -- every parameter estimate on real data is then an extrapolation. Measured two
ways that fail differently: a kernel two-sample test (MMD, with a permutation null) and a
classifier two-sample test (C2ST, out-of-fold accuracy), both blocked by recording so that
within-recording correlation cannot masquerade as a real difference.

What the result MEANS is where the two workflows part company, and the difference is worth stating
because it inverts:

* **detector** -- the imaging parameters are what the network infers, so the synthetic videos are
  supposed to reproduce the real ones' *appearance*. A gap is a realism failure of the imaging
  forward model, and closing it is the goal of the detector calibration.
* **biology** -- the imaging is held fixed and the reaction-diffusion parameters are inferred, so a
  gap is NOT primarily an imaging defect. It says the real recordings carry structure the synthetic
  prior never generated -- biological variation outside the prior's reach, or an imaging mismatch
  the frozen imaging vector cannot express. The remedy is a wider prior or a recalibrated imaging
  vector, not a longer training run.

Everything measurable is shared; only the spec (:func:`_esd_spec`) differs -- the parameterization,
the alias-qualified paths, and the report-facing parameter descriptions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure

from . import artifacts
from . import embedding_space_distance as esd
from .diagnostics import DiagnosticReporter
from .inference_support import resolve_topology
from .io import convert_video_dtype
from .parameterization import PARAMETERS, RunTiming
from .workflow import parameter_keys, parameter_table

# Conditions are named scientifically wherever a reader sees them. The tokens below survive only as
# the recording FILENAMES on disk (Experiment_<KEY>_Cell_...), which is why they persist at all.
KIND_OF_CONDITION = {"MET-FAB": "ALP", "MET-INLB": "BET"}

_FIGURE_SUBSAMPLE = 1000

# ALP/BET are internal keys only — the recording filenames (Experiment_<KEY>_Cell_...) and the
# --kinds CLI. Every figure and table shows the experimental condition names instead. Same
# convention as the sibling analysis scripts.
CONDITION_DISPLAY = {"ALP": "MET-FAB", "BET": "MET-INLB"}


def _disp(key):
    """Display label for an internal condition key (identity for non-conditions, e.g. 'synthetic')."""
    return CONDITION_DISPLAY.get(key, key)


# Concise report-facing descriptions of the DETECTOR parameters -- the six learnable targets plus the
# five SCOPE camera-nuisance parameters — distilled from the NOTE fields in
# detector_parameterization (the authoritative long-form source for that workflow).
_PARAM_MEANING = {
    "gamma": "camera gain: ADU per photoelectron (gamma = g/C)",
    "kappa_o": "optical background offset: pre-gain photons/pixel/frame; amplified by the gain, sets the ADU floor",
    "kappa_b": "camera baseline: constant ADU added after gain",
    "kappa_s": "read noise: post-register Gaussian sigma (ADU)",
    "kappa_q": "quantum efficiency: photon->photoelectron probability (marginalized as SCOPE camera nuisance; only gamma*kappa_q identifiable)",
    "mu_r": "PSF width: median of the per-emitter PSF-width distribution (pixels)",
    "sigma_r": "PSF-width spread: log-spread of that distribution",
    "mu_pc": "emitter brightness: median photon count per emitter",
    "sigma_pc": "emitter-brightness spread: log-spread of that distribution",
    "prob_photo_bleach": "photobleaching probability over a 100-frame reference window",
    "lambda_rate": "flicker rate: base rate of emitter brightness-state transitions",
}


def _chunk_geometry(timing, chunk_step_seconds):
    """(n_frames, step_frames) for the model-length sliding window over a recording."""
    n_frames = timing.frame_count
    window_seconds = timing.total_time_seconds
    step_seconds = chunk_step_seconds if chunk_step_seconds else int(window_seconds)
    if (window_seconds != int(window_seconds) or step_seconds < 1
            or step_seconds > int(window_seconds)
            or int(window_seconds) % int(step_seconds) != 0):
        raise SystemExit(
            f"--chunk-step-seconds={step_seconds} is invalid: a positive integer dividing the model "
            f"window ({window_seconds:g} s) and not exceeding it.")
    return n_frames, int(round(step_seconds / timing.frame_time_seconds))


def _discover_cells(experiment_dir, kind, span):
    """Cell indices with an ``Experiment_{kind}_Cell_{n}_{span}S_RAW.tif`` recording, sorted."""
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        try:
            cells.append(int(path.stem.split("_Cell_")[1].split("_")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(cells)


def _load_posterior(R):
    """Load the required detector estimator onto this machine's device (fail loud if absent)."""
    if not R["estimator_path"].exists():
        raise FileNotFoundError(
            f"Detector estimator not found:\n    {R['estimator_path']}\n"
            f"Train the Detector Inference stage first so the estimator exists.")
    device = resolve_topology().device
    posterior = artifacts.load_estimator(str(R["estimator_path"]), device=str(device),
                                          expected_parameter_keys=R["parameter_keys"])
    posterior.posterior_estimator.to(device)
    return posterior, device


def _embed_experimental(R, posterior, device, kinds, span, n_frames, step_frames, max_cells):
    """Embed each experimental recording's model-length windows, one recording at a time.

    Returns ``{kind: {"emb": [n, D], "cells": [n], "n_cells": int}}``. Embedding per recording keeps
    only one recording's frames in memory at a time; the ``cells`` label (``"{kind}_{cell}"``, unique
    across kinds) carries the recording identity the cell-level significance needs.
    """
    groups = {}
    for kind in kinds:
        cells = _discover_cells(R["experiment_dir"], kind, span)
        if max_cells > 0:
            cells = cells[:max_cells]
        emb_rows, cell_ids = [], []
        for cell in cells:
            tif = R["paths"].experiment_video_path(kind, cell, span, R["data_bank_root"])
            if not tif.exists():
                continue
            v8 = convert_video_dtype(tifffile.imread(str(tif)), bits_from=16, bits_to=8)
            chunks = [v8[s:s + n_frames]
                      for s in range(0, v8.shape[0] - n_frames + 1, step_frames)]
            if not chunks:
                continue
            emb_rows.append(esd.embed_videos(posterior, chunks, device=str(device),
                                             expected_frames=n_frames))
            cell_ids += [f"{kind}_{cell}"] * len(chunks)
        emb = np.vstack(emb_rows) if emb_rows else np.empty((0, 0))
        groups[kind] = {"emb": emb, "cells": np.array(cell_ids), "n_cells": len(cells)}
    return groups


def _embed_eval(R, posterior, device, eval_tasks, n_frames):
    """Embed the synthetic EVAL video sets task-by-task (lazy zarr); return ``([N, D], n_tasks)``."""
    from srm_and_sbi_dimer_alp.visualization_dli import load_video_set
    embs, n_tasks, task = [], 0, 0
    while eval_tasks is None or n_tasks < eval_tasks:
        path = R["paths"].video_set_path(task, R["data_bank_root"], R["timing_label"],
                                         compress=True, split="EVAL")
        if not path.exists():
            break
        arr = load_video_set(str(path))                          # lazy (n_sims, n_frames, H, W)
        embs.append(esd.embed_videos(posterior, arr, device=str(device), expected_frames=n_frames))
        n_tasks += 1
        task += 1
    if not embs:
        raise FileNotFoundError(
            f"No synthetic EVAL video sets found under "
            f"{R['data_bank_root'] / R['paths'].video_subdir} for timing {R['timing_label']}. "
            f"Generate the EVAL split first.")
    return np.vstack(embs), n_tasks


def _load_eval_theta(R, n_tasks):
    """Imaging parameters for the embedded EVAL tasks, in the same task/video order as ``_embed_eval``.

    Returns ``([N, P] physical-unit theta, keys, labels)`` with columns 1:1 to
    the workflow's parameter keys, or ``(None, None, None)`` if any task's theta set is absent -- in
    which case the PC1 parameter tracking is skipped (it is an optional add-on to the distance
    measure, needing the theta that generated the exact videos embedded).
    """
    from srm_and_sbi_dimer_alp.io import load_data
    rows = []
    for task in range(n_tasks):
        path = R["paths"].theta_set_path(task, R["data_bank_root"], R["timing_label"],
                                         compress=True, split="EVAL")
        if not path.exists():
            return None, None, None
        rows.append(np.asarray(load_data(str(path))))
    keys = R["parameter_keys"]
    labmap = {e["KEY"]: (e.get("LABEL") or e["KEY"]) for e in R["parameterization"]}
    return np.vstack(rows), keys, [labmap.get(k, k) for k in keys]


# --- distances (local helpers for the faithful figures; MMD uses the same squared distances) ------

def _sqdist(A, B):
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    return np.maximum(aa[:, None] + bb[None, :] - 2.0 * (A @ B.T), 0.0)


def _pair_dists(A, B=None):
    """Flat vector of Euclidean distances: within ``A`` (upper triangle) if ``B`` is None, else A×B."""
    if B is None:
        d = np.sqrt(_sqdist(A, A))
        return d[np.triu_indices(len(A), k=1)]
    return np.sqrt(_sqdist(A, B)).ravel()


def _l2norm(E):
    """Row-wise projection to the unit sphere; the Euclidean distance then lies in [0, 2] (cosine)."""
    E = np.asarray(E, float)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def _subsample(emb, cap, rng):
    return emb if len(emb) <= cap else emb[rng.choice(len(emb), cap, replace=False)]


def _compare(A, cells_a, B, cells_b, *, alpha, n_perm, seed):
    """One comparison: MMD + C2ST + a combined verdict (overlap iff both are non-significant)."""
    mmd = esd.mmd_rbf(A, B, cells_x=cells_a, cells_y=cells_b, n_permutations=n_perm, seed=seed)
    c = esd.c2st(A, B, cells_x=cells_a, cells_y=cells_b, seed=seed)
    c2st_ok = np.isnan(c["p_value"]) or c["p_value"] >= alpha        # NaN (too few folds) -> can't reject
    verdict = "overlap" if (mmd["p_value"] >= alpha and c2st_ok) else "gap"
    if np.isnan(c["p_value"]):
        verdict += " (C2ST inconclusive)"
    return {"mmd": mmd, "c2st": c, "verdict": verdict}


# --- figures (deterministic; all groups on shared axes) -------------------------------------------

def _figure_distance_distributions(groups_emb, present, rng):
    """Four panels: rows are the distance geometry (top raw, bottom L2-normalized), columns are
    (a) within-group spread and (b) cross-group separation. In either row, overlap reads as panel
    (b)'s mass falling in panel (a)'s range; a rightward shift in (b) is a gap. The raw row is the
    primary geometry — its scale is in the network's activation units, so arbitrary; the L2 row
    projects each embedding to the unit sphere, bounding the distance in [0, 2] (cosine), a check on
    whether the raw picture is merely a magnitude effect.
    """
    subs = {k: _subsample(v, _FIGURE_SUBSAMPLE, rng) for k, v in groups_emb.items() if len(v)}
    subs_l2 = {k: _l2norm(v) for k, v in subs.items()}
    cross = [(present[0], "synthetic")]
    if len(present) == 2:
        cross += [(present[1], "synthetic"), (present[0], present[1])]

    def _draw(ax_w, ax_c, emb):
        for k, e in emb.items():
            if len(e) > 1:
                ax_w.hist(_pair_dists(e), bins=60, density=True, histtype="step", linewidth=1.6,
                          label=f"within {_disp(k)}")
        for a, b in cross:
            if a in emb and b in emb:
                ax_c.hist(_pair_dists(emb[a], emb[b]), bins=60, density=True, histtype="step",
                          linewidth=1.6, linestyle="--", label=f"{_disp(a)} ↔ {_disp(b)}")
        for ax in (ax_w, ax_c):
            ax.legend(fontsize=7)

    fig = Figure(figsize=(10, 8), layout="constrained")
    (ax_rw, ax_rc), (ax_lw, ax_lc) = fig.subplots(2, 2, sharex="row")
    _draw(ax_rw, ax_rc, subs)
    _draw(ax_lw, ax_lc, subs_l2)
    ax_rw.set_title("(a) within-group — spread of each cloud")
    ax_rc.set_title("(b) cross-group — separation between clouds")
    ax_rw.set_ylabel("raw\ndensity")
    ax_lw.set_ylabel("L2-normalized\ndensity")
    for ax in (ax_rw, ax_rc):
        ax.set_xlabel("embedding distance (arbitrary units)")
    for ax in (ax_lw, ax_lc):
        ax.set_xlabel("L2 distance (cosine, [0, 2])")
    fig.suptitle("Embedding-distance distributions — raw (top, primary) vs L2-normalized (bottom); "
                 "overlap ⇔ (b) lies over (a)")
    return fig


def _figure_c2st_scores(oof_score, n_exp, exp_kind_labels):
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    oof = np.asarray(oof_score, dtype=float)
    exp_scores, synth_scores = oof[:n_exp], oof[n_exp:]
    kinds = np.asarray(exp_kind_labels)
    for k in sorted(set(kinds.tolist())):
        ax.hist(exp_scores[kinds == k], bins=30, range=(0, 1), density=True, histtype="step",
                linewidth=1.6, label=f"experimental {_disp(k)}")
    ax.hist(synth_scores, bins=30, range=(0, 1), density=True, histtype="step", linewidth=1.6,
            label="synthetic")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("classifier P(synthetic)")
    ax.set_ylabel("density")
    ax.set_title("C2ST out-of-fold scores (all near 0.5 ⇒ overlap; split to 0/1 ⇒ gap)")
    ax.legend(fontsize=8)
    return fig


def _figure_distance_matrix(groups_emb, present, rng):
    labels = [k for k in present + ["synthetic"] if k in groups_emb and len(groups_emb[k])]
    subs = {k: _subsample(groups_emb[k], _FIGURE_SUBSAMPLE, rng) for k in labels}
    n = len(labels)
    M = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            M[i, j] = _pair_dists(subs[a]).mean() if i == j else _pair_dists(subs[a], subs[b]).mean()
    fig = Figure(figsize=(5.6, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(M, cmap="viridis")
    fig.colorbar(im, ax=ax, label="mean embedding distance")
    disp = [_disp(l) for l in labels]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(disp, rotation=30, ha="right"); ax.set_yticklabels(disp)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Inter-group mean embedding distance")
    return fig


def _pca_pooled(groups_emb, present):
    """Top-2 PCA of the pooled raw embeddings (= classical MDS). Returns the 2-D coordinates, the
    per-point group labels, the ordered explained-variance ratios, the components (for the loading
    diagnostics), and the centered embeddings (for the Shepard high-dimensional distances)."""
    labels = [k for k in present + ["synthetic"] if k in groups_emb and len(groups_emb[k])]
    X = np.vstack([groups_emb[k] for k in labels])
    grp = np.concatenate([[k] * len(groups_emb[k]) for k in labels])
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / np.sum(S ** 2)
    return {"coords": Xc @ Vt[:2].T, "grp": grp, "labels": labels, "ev": ev,
            "components": Vt, "Xc": Xc}


def _figure_pca_scatter(pca, rng):
    """Three panels on the raw-embedding PCA (= classical MDS, the optimal linear distance-preserving
    projection). (a) The pooled embeddings on their top-2 principal components, colored by group,
    each axis labeled with its explained-variance fraction — when PC1 dominates, the points fall on a
    single curved arc (the horseshoe signature of one-dimensional data), not separated blobs. (b) The
    distribution of each group's PC1 coordinate — the honest one-dimensional reading of where each
    group sits along the single real axis and how far it spreads. (c) A Shepard diagram: projected
    2-D pairwise distance against the true high-dimensional distance; a linear projection can only
    shrink distances, so points lie on or below the identity line, and the annotated correlation
    states how faithful the view is.
    """
    coords, grp, labels, ev, Xc = pca["coords"], pca["grp"], pca["labels"], pca["ev"], pca["Xc"]
    fig = Figure(figsize=(15, 4.6), layout="constrained")
    ax_s, ax_d, ax_h = fig.subplots(1, 3)

    # Fixed per-group colors (canonical order) so panels (a) and (b) agree on which color is which.
    color = {k: f"C{i}" for i, k in enumerate(labels)}
    # Draw the large synthetic set first (underneath, faint); the tight experimental clusters go on
    # top -- larger, more opaque, higher zorder -- so MET-FAB / MET-INLB stay visible, not buried.
    draw_order = [k for k in labels if k == "synthetic"] + [k for k in labels if k != "synthetic"]
    for k in draw_order:
        pts = coords[grp == k]
        if len(pts) > _FIGURE_SUBSAMPLE:
            pts = pts[rng.choice(len(pts), _FIGURE_SUBSAMPLE, replace=False)]
        syn = k == "synthetic"
        ax_s.scatter(pts[:, 0], pts[:, 1], s=6 if syn else 12, alpha=0.2 if syn else 0.7,
                     edgecolors="none", color=color[k], label=_disp(k), zorder=1 if syn else 3)
    ax_s.set_xlabel(f"PC1 ({ev[0] * 100:.2f}% var)")
    ax_s.set_ylabel(f"PC2 ({ev[1] * 100:.2f}% var)")
    ax_s.set_title(f"(a) PCA projection — top 2 capture {(ev[0] + ev[1]) * 100:.2f}% of variance")
    ax_s.legend(fontsize=8, markerscale=2)

    for k in labels:
        ax_d.hist(coords[grp == k][:, 0], bins=60, density=True, histtype="step", linewidth=1.6,
                  color=color[k], label=_disp(k))
    ax_d.set_xlabel(f"PC1 ({ev[0] * 100:.2f}% var) — the single real axis")
    ax_d.set_ylabel("density")
    ax_d.set_title("(b) position along PC1 — spread of each group")
    ax_d.legend(fontsize=8)

    cap = min(len(Xc), 600)
    idx = rng.choice(len(Xc), cap, replace=False) if len(Xc) > cap else np.arange(len(Xc))
    hd, ld = _pair_dists(Xc[idx]), _pair_dists(coords[idx])
    r = float(np.corrcoef(hd, ld)[0, 1]) if len(hd) > 1 else float("nan")
    sel = rng.choice(len(hd), 5000, replace=False) if len(hd) > 5000 else np.arange(len(hd))
    ax_h.scatter(hd[sel], ld[sel], s=4, alpha=0.2, edgecolors="none")
    lim = float(hd.max()) if len(hd) else 1.0
    ax_h.plot([0, lim], [0, lim], color="grey", linestyle=":", linewidth=1.0, label="identity")
    ax_h.set_xlabel("high-dimensional distance")
    ax_h.set_ylabel("projected 2-D distance")
    ax_h.set_title(f"(c) Shepard diagram — distance correlation r = {r:.2f}")
    ax_h.legend(fontsize=8)
    return fig


def _track_collapse(pc1, theta, keys, labels, param_meaning):
    """Rank each target parameter by |Spearman correlation| with PC1 (the dominant embedding axis).

    Spearman (rank) so it is invariant to the log/linear storage of the parameters and to a monotone
    nonlinearity between a parameter and the embedding coordinate. Returns rows sorted by ``|rho|``.
    """
    from scipy.stats import spearmanr
    rows = []
    for j, (k, lab) in enumerate(zip(keys, labels)):
        rho, p = spearmanr(pc1, theta[:, j])
        rows.append({"key": k, "label": lab, "meaning": param_meaning.get(k, ""),
                     "rho": float(rho), "p": float(p)})
    rows.sort(key=lambda r: -abs(r["rho"]))
    return rows


def _figure_collapse_tracking(pc1, theta, keys, ranked):
    """(a) |Spearman rho| of PC1 with each target parameter, sorted; (b) PC1 against the top one."""
    fig = Figure(figsize=(11, 4.6), layout="constrained")
    ax_b, ax_s = fig.subplots(1, 2)
    ax_b.bar(range(len(ranked)), [abs(r["rho"]) for r in ranked], color="steelblue")
    ax_b.set_xticks(range(len(ranked)))
    ax_b.set_xticklabels([r["label"] for r in ranked], fontsize=9)
    ax_b.set_ylim(0, 1)
    ax_b.set_ylabel("|Spearman ρ| with PC1")
    ax_b.set_title("(a) target parameter vs the dominant axis")
    top = ranked[0]
    j = keys.index(top["key"])
    ax_s.scatter(theta[:, j], pc1, s=6, alpha=0.3, edgecolors="none")
    ax_s.set_xlabel(f"{top['label']}  ({top['key']}, physical units)")
    ax_s.set_ylabel("PC1 coordinate")
    ax_s.set_title(f"(b) PC1 vs {top['label']} — ρ = {top['rho']:+.2f}")
    return fig


def _write_report(R, args, comps, exp, present, groups_emb, n_ev_tasks, pooled_kind, n_pooled_exp,
                  rng, l2_mmd, pca, synth_theta, theta_keys, theta_labels):
    report_dir = R["posit_dir"] / f"{R['paths'].project_alias}_{R['timing_label']}_Embedding_Space_Distance"
    reporter = DiagnosticReporter(
        stage="Embedding_Space_Distance", enabled=True, dump=True, dump_dir=report_dir,
        run_label=f"{R['paths'].project_alias}_{R['timing_label']}")

    # Report checks: the embedded sample counts (both groups non-empty) and finiteness.
    n_synth = int(groups_emb["synthetic"].shape[0])
    finite = bool(np.isfinite(groups_emb["synthetic"]).all()
                  and all(np.isfinite(exp[k]["emb"]).all() for k in present))
    reporter.check("experimental_windows_nonempty", n_pooled_exp > 0,
                   f"{n_pooled_exp} experimental windows embedded across {len(present)} condition(s)")
    reporter.check("synthetic_windows_nonempty", n_synth > 0,
                   f"{n_synth} synthetic EVAL windows embedded ({n_ev_tasks} task(s))")
    reporter.check("no_nan_inf(embeddings)", finite, "clean" if finite else "NaN/Inf present")

    label = {"pooled": "experimental (pooled) vs synthetic"}
    for k in present:
        label[k] = f"experimental {_disp(k)} vs synthetic"
    order = ["pooled"] + present
    if len(present) == 2:
        key = f"{present[0]}_vs_{present[1]}"
        label[key] = f"experimental {_disp(present[0])} vs {_disp(present[1])}"
        order.append(key)
    rows = []
    for key in order:
        c = comps[key]; m, cc = c["mmd"], c["c2st"]
        cp = f"{cc['p_value']:.3g}" if not np.isnan(cc["p_value"]) else "n/a"
        rows.append([label[key], f"{m['mmd']:.4g}", f"{m['p_value']:.3g}", f"{cc['accuracy']:.3f}",
                     f"[{cc['ci95'][0]:.3f}, {cc['ci95'][1]:.3f}]", cp, c["verdict"]])
    reporter.table(
        "Experimental-versus-synthetic embedding distance",
        ["comparison", "MMD", "MMD p", "C2ST acc", "C2ST 95% CI", "C2ST p", "verdict"], rows,
        note=f"Overlap (realistic imaging) is concluded when both agree: the C2ST CI covers 0.5 "
             f"(p >= {args.alpha}) and the MMD permutation p >= {args.alpha}. The pooled row is the "
             f"primary imaging-realism result (imaging is a microscope property, not a condition); "
             f"the per-condition and between-condition rows are diagnostics.")

    reporter.table(
        "Group composition",
        ["group", "recordings", "windows (embeddings)"],
        [[_disp(k), str(exp[k]["n_cells"]), str(exp[k]["emb"].shape[0])] for k in present]
        + [["synthetic EVAL", f"{n_ev_tasks} task(s)", str(groups_emb["synthetic"].shape[0])]],
        note=f"Pooled experimental group composed by mix='{args.mix}'.")

    ev = pca["ev"]
    kk = min(6, len(ev))
    cum = np.cumsum(ev)
    reporter.table(
        "PCA variance spectrum (pooled raw embeddings)",
        ["component", "variance", "cumulative"],
        [[f"PC{i + 1}", f"{ev[i] * 100:.2f}%", f"{cum[i] * 100:.2f}%"] for i in range(kk)],
        note="Intrinsic dimensionality of the embedding — how fast the variance concentrates; a "
             "single dominant component means a near-one-dimensional embedding.")
    v1 = pca["components"][0]
    pr = float(1.0 / np.sum(v1 ** 4))       # participation ratio: effective # of dims building PC1
    reporter.stat("PC1 max |loading|", f"{float(np.max(np.abs(v1))):.3f}")
    reporter.stat("PC1 participation ratio",
                  f"{pr:.1f} of {len(v1)} dims "
                  f"({'one coordinate dominates' if pr < 2 else 'a multi-dimensional combination'})")

    reporter.stat("estimator", str(R["estimator_path"]))
    reporter.stat("mix", args.mix)
    reporter.stat("alpha", args.alpha)
    reporter.stat("n_permutations", args.n_permutations)
    reporter.stat("pooled MMD (L2/cosine, secondary)",
                  f"{l2_mmd['mmd']:.4g}  (permutation p {l2_mmd['p_value']:.3g})")

    reporter.save_figure(
        "embedding_distance_distributions", _figure_distance_distributions(groups_emb, present, rng),
        caption="Distributions of pairwise distances between video windows in the trained detector "
                "embedding, across two geometries (rows) and two panel types (columns). Columns: "
                "(a) within-group distances (each cloud to itself) show internal spread; (b) "
                "cross-group distances show how far apart groups sit. Rows: the top row is the raw "
                "embedding — the primary geometry, in the network's activation units, so its scale "
                "is arbitrary; the bottom row projects each embedding to the unit sphere, bounding "
                "the distance in [0, 2] (cosine), a check on whether the raw picture is a magnitude "
                "artifact. In either row, cross-group mass in (b) falling within the within-group "
                "range of (a) means overlap (realistic imaging); a shift to larger distances means a "
                "gap. The raw MMD is the statistic of record; the pooled MMD in the L2 geometry is a "
                "secondary diagnostic (reported below the table).")
    reporter.save_figure(
        "c2st_scores", _figure_c2st_scores(comps["pooled"]["c2st"]["oof_score"],
                                           n_pooled_exp, pooled_kind),
        caption="Out-of-fold probability that each window is synthetic, from the C2ST classifier "
                "(every window scored by a fold that did not train on it, so the scores are honest). "
                "If experimental and synthetic are indistinguishable the classifier can only guess "
                "and all groups pile on the dotted 0.5 line; the more the synthetic scores push "
                "toward 1 and the experimental toward 0, the more separable the groups — a larger "
                "imaging gap. This is the graphical form of the C2ST accuracy (0.5 = overlap, "
                "→1 = gap).")
    reporter.save_figure(
        "inter_group_distance_matrix", _figure_distance_matrix(groups_emb, present, rng),
        caption="Mean pairwise embedding distance between and within groups: the diagonal is each "
                "group's own internal spread, the off-diagonal the distance between two groups. Read "
                "each off-diagonal cell against its row's diagonal — comparable means the groups "
                "overlap, much larger means a gap. Same arbitrary activation units as the distance "
                "distributions, so compare cells to one another, not to an absolute scale.")
    reporter.save_figure(
        "pca_scatter", _figure_pca_scatter(pca, rng),
        caption="PCA (equivalently classical MDS) of the raw embeddings — the optimal linear "
                "distance-preserving projection, deterministic unlike UMAP or t-SNE. (a) The pooled "
                "embeddings on their top two principal components, colored by group, with each axis's "
                "explained-variance fraction; when PC1 dominates, the points fall on a single curved "
                "arc (the horseshoe artifact of one-dimensional data) rather than separated blobs. "
                "(b) The distribution of each group's PC1 coordinate — the honest one-dimensional "
                "reading of where each group sits along the single real axis and how far it spreads. "
                "(c) The Shepard diagram plots projected 2-D distance against true high-dimensional "
                "distance; a linear projection can only shrink distances, so points lie on or below "
                "the identity line, and the annotated correlation states how faithfully the view "
                "preserves them. The variance-spectrum table quantifies the dimensionality; the "
                "distance-distribution and matrix figures remain the exact high-dimensional summary.")

    if synth_theta is not None:
        pc1_synth = pca["coords"][pca["grp"] == "synthetic", 0]
        if len(pc1_synth) == len(synth_theta):
            ranked = _track_collapse(pc1_synth, synth_theta, theta_keys, theta_labels, R["param_meaning"])
            reporter.table(
                "PC1 vs target parameters (Spearman) - collapse driver",
                ["parameter", "key", "meaning", "Spearman rho", "abs(rho)", "p"],
                [[r["label"], r["key"], r["meaning"], f"{r['rho']:+.3f}", f"{abs(r['rho']):.3f}",
                  f"{r['p']:.1e}"] for r in ranked],
                note="Correlation of the dominant embedding axis (PC1) with each known imaging "
                     "parameter of the synthetic EVAL set (meanings distilled from the parameter "
                     "schema); the largest |rho| is the parameter that drives the embedding's single "
                     "dominant direction.")
            reporter.save_figure(
                "collapse_tracking",
                _figure_collapse_tracking(pc1_synth, synth_theta, theta_keys, ranked),
                caption="Which imaging parameter the dominant embedding axis (PC1) encodes. (a) The "
                        "absolute Spearman correlation of PC1 with each imaging parameter of the "
                        "synthetic EVAL set, sorted; the tallest bar is the parameter driving the "
                        "near-one-dimensional collapse. (b) PC1 against that top parameter, labeled "
                        "with the signed Spearman rho. Rank correlation, so it is unaffected by the "
                        "log or linear storage of the parameters.")
    reporter.write_report()
    return report_dir



def run_embedding_space_distance(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    R = _esd_spec(cfg, args)
    # The CLI names conditions scientifically; the recordings on disk are named with the stored
    # tokens, so translate once, here, at the boundary.
    kinds = [KIND_OF_CONDITION.get(k.strip(), k.strip()) for k in args.kinds.split(",") if k.strip()]
    n_frames, step_frames = _chunk_geometry(R["timing"], args.chunk_step_seconds)

    if args.dry_run:
        ev0 = R["paths"].video_set_path(0, R["data_bank_root"], R["timing_label"], True, "EVAL")
        print(f"[DRY RUN] Embedding-space distance ({cfg.tag}) - would read/write:")
        print(f"    estimator     : {R['estimator_path']}  "
              f"[{'OK' if R['estimator_path'].exists() else 'MISSING'}]")
        print(f"    experimental  : {R['experiment_dir']}/Experiment_<KIND>_Cell_<n>_"
              f"{args.experiment_span_seconds}S_RAW.tif")
        for k in kinds:
            n_cells = len(_discover_cells(R["experiment_dir"], k, args.experiment_span_seconds))
            print(f"        {_disp(k):<9}: {n_cells} recording(s)")
        print(f"    synthetic EVAL: {ev0.parent}/...Video_Set_TASK_<k>_EVAL.zarr  "
              f"[task 0 {'OK' if ev0.exists() else 'MISSING'}]  (--eval-tasks {args.eval_tasks or 'all'})")
        print(f"    window        : {n_frames} frames, step {step_frames}; mix={args.mix}, "
              f"alpha={args.alpha}")
        print(f"    writes        : {R['posit_dir']}/{R['paths'].project_alias}_"
              f"{R['timing_label']}_Embedding_Space_Distance/")
        print("[DRY RUN] no compute performed.")
        return 0

    posterior, device = _load_posterior(R)
    print("Embedding experimental recordings (one recording at a time) ...")
    exp = _embed_experimental(R, posterior, device, kinds, args.experiment_span_seconds,
                              n_frames, step_frames, args.max_cells)
    present = [k for k in kinds if exp[k]["emb"].shape[0] > 0]
    if not present:
        raise FileNotFoundError(
            f"No experimental recordings found under {R['experiment_dir']} for "
            f"{[_disp(k) for k in kinds]} at span {args.experiment_span_seconds}s. "
            f"Stage the recordings first.")
    print("Embedding synthetic EVAL videos (task by task) ...")
    synth_emb, n_ev_tasks = _embed_eval(R, posterior, device, args.eval_tasks, n_frames)

    groups_emb = {k: exp[k]["emb"] for k in present}
    groups_emb["synthetic"] = synth_emb

    # Pooled experimental group (mix=natural keeps the dataset's proportions; balanced draws an
    # equal number of windows per condition). Each recording stays its own cell either way.
    rng = np.random.default_rng(0)
    if args.mix == "balanced" and len(present) > 1:
        m = min(exp[k]["emb"].shape[0] for k in present)
        picks = {k: rng.choice(exp[k]["emb"].shape[0], m, replace=False) for k in present}
        pooled_emb = np.vstack([exp[k]["emb"][picks[k]] for k in present])
        pooled_cells = np.concatenate([exp[k]["cells"][picks[k]] for k in present])
        pooled_kind = np.concatenate([[k] * m for k in present])
    else:
        pooled_emb = np.vstack([exp[k]["emb"] for k in present])
        pooled_cells = np.concatenate([exp[k]["cells"] for k in present])
        pooled_kind = np.concatenate([[k] * exp[k]["emb"].shape[0] for k in present])

    comps = {"pooled": _compare(pooled_emb, pooled_cells, synth_emb, None,
                                alpha=args.alpha, n_perm=args.n_permutations, seed=0)}
    for k in present:
        comps[k] = _compare(exp[k]["emb"], exp[k]["cells"], synth_emb, None,
                            alpha=args.alpha, n_perm=args.n_permutations, seed=0)
    if len(present) == 2:
        a, b = present
        comps[f"{a}_vs_{b}"] = _compare(exp[a]["emb"], exp[a]["cells"],
                                        exp[b]["emb"], exp[b]["cells"],
                                        alpha=args.alpha, n_perm=args.n_permutations, seed=0)

    # Secondary diagnostic matching the figure's L2 row: the pooled MMD in the L2-normalized
    # (cosine) geometry. Raw stays the geometry of record (the verdict table); this is a scale
    # check only.
    l2_mmd = esd.mmd_rbf(_l2norm(pooled_emb), _l2norm(synth_emb), cells_x=pooled_cells,
                         cells_y=None, n_permutations=args.n_permutations, seed=0)
    pca = _pca_pooled(groups_emb, present)
    synth_theta, theta_keys, theta_labels = _load_eval_theta(R, n_ev_tasks)
    if synth_theta is None:
        print("  (EVAL theta not found - skipping PC1 parameter tracking)")

    report_dir = _write_report(R, args, comps, exp, present, groups_emb, n_ev_tasks, pooled_kind,
                               pooled_emb.shape[0], rng, l2_mmd, pca,
                               synth_theta, theta_keys, theta_labels)
    print(f"\nprimary (pooled) verdict: {comps['pooled']['verdict']}  "
          f"(C2ST acc {comps['pooled']['c2st']['accuracy']:.3f}, "
          f"MMD p {comps['pooled']['mmd']['p_value']:.3g})")
    print(f"report + figures:\n    {report_dir}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration; sets the timing label locating the "
                        "estimator and EVAL set and naming the outputs.")
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="duration (s) of the experimental recordings to read (default 20).")
    p.add_argument("--kinds", type=str, default="MET-FAB,MET-INLB",
                   help="comma-separated experimental conditions (default 'MET-FAB,MET-INLB'); each "
                        "is a diagnostic row, all are pooled for the primary row.")
    p.add_argument("--eval-tasks", type=int, default=None,
                   help="number of synthetic EVAL tasks to embed as the reference "
                        "(default: all present).")
    p.add_argument("--chunk-step-seconds", type=int, default=None,
                   help="sliding-window step (s) for dicing recordings; default the model window "
                        "(non-overlapping).")
    p.add_argument("--mix", choices=("natural", "balanced"), default="natural",
                   help="how the pooled experimental group is composed: 'natural' keeps the dataset "
                        "proportions (default); 'balanced' draws equal windows per condition.")
    p.add_argument("--n-permutations", type=int, default=1000,
                   help="MMD permutation count (default 1000).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="significance level for the overlap/gap verdict (default 0.05).")
    p.add_argument("--max-cells", type=int, default=0,
                   help="cap the recordings per condition (0 = all; default 0).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, "
                        "compute nothing.")
    return p


# ---- the one place the two workflows differ ----------------------------------------------------

# Concise report-facing descriptions. The detector's are distilled from the long-form NOTE fields of
# detector_parameterization; the biology entries state each parameter's role in the A/B/C reaction
# scheme documented in PROJECT_CONTEXT.md (A monomer, B mobile dimer, C immobile dimer).
_PARAM_MEANING_DETECTOR = _PARAM_MEANING

_PARAM_MEANING_BIOLOGY = {
    "count_alp": "abundance: initial monomer (A) count in the simulated region",
    "count_bet": "abundance: initial mobile-dimer (B) count",
    "count_chi": "abundance: initial immobile-dimer (C) count",
    "diffusivity_alp": "mobility: monomer diffusion coefficient D_A (um^2/s)",
    "relative_diffusivity_bet": "mobility ratio: mobile-dimer diffusivity as a fraction of D_A",
    "relative_diffusivity_chi": "mobility ratio: immobile-dimer diffusivity as a fraction of D_A",
    "relative_rate_dimerization": "association: monomer-monomer dimerization rate (A + A -> B)",
    "rate_dissociation": "dissociation: mobile dimer breaks into two monomers (B -> A + A), per second",
    "rate_immobility": "immobilization: mobile dimer enters the immobile state (B -> C), per second",
    "rate_mobility": "remobilization: immobile dimer returns to the mobile state (C -> B), per second",
}


def _esd_spec(cfg, args):
    """Resolve the workflow-specific half of the analysis."""
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths
    posit_dir = data_bank_root / paths.posit_subdir
    return dict(
        timing=timing, timing_label=timing.label, data_bank_root=data_bank_root,
        paths=paths, posit_dir=posit_dir,
        experiment_dir=data_bank_root / paths.experiment_subdir,
        estimator_path=posit_dir / f"{paths.project_alias}_{timing.label}_Estimator.npz",
        parameter_keys=parameter_keys(cfg),
        parameterization=parameter_table(cfg),
        param_meaning=(_PARAM_MEANING_DETECTOR if cfg.tag == "detector"
                       else _PARAM_MEANING_BIOLOGY),
        workflow=cfg.tag,
    )
