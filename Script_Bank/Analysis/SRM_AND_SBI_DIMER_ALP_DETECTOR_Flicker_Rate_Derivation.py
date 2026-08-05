"""SRM_AND_SBI_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.py

Special-situation utility (NOT a canonical pipeline stage; not wired into the HPC dispatcher).
Derives the emitter brightness-flicker rate ``lambda_rate`` (DETECTOR_WORKFLOW.md sec. 6.3) from
the public MET single-molecule localization tables, so the prior on ``lambda_rate`` is reproducible
from data rather than assumed. Reads only; writes nothing to the data root.

Method (DETECTOR_WORKFLOW.md sec. 6.3):
  1. Measure. For each track, ``log(intensity[photon])`` vs ``frame`` is linearly detrended --
     this removes bleaching and the per-emitter mean, both multiplicative and hence additive in
     the log. A gap-aware temporal autocorrelation is formed and pooled over tracks. The lag-0 ->
     lag-1 drop is per-localization white fit noise (confined to lag 0); the decay of ``rho(k>=1)``
     is the physical flicker correlation time ``tau_corr``.
  2. Model. The autocorrelation of the model brightness chain (``compute_matrices`` ->
     ``generate_state_trajectories``) decays on the generator's spectral-gap timescale
     (Norris 1997, Markov Chains, CUP). It is measured the SAME way over a grid of ``lambda_rate``
     -> ``tau_model(lambda_rate)``. To null the finite-track-length detrend bias, the model
     trajectories are cut to the EMPIRICAL track-length distribution and detrended identically.
  3. Map. Match the early autocorrelation SHAPE (window-independent) of data and model to convert
     ``tau_corr -> lambda_rate``; the 1/e crossing is reported as a cross-check. The mapping is
     evaluated across the ``sigma_pc`` prior and per condition (Fab, InlB) -- the rate is a
     photophysical quantity and should be condition-independent.

Result (MET S-BSST712): ``tau_corr ~ 0.13 s`` -> ``lambda_rate ~ 5``, Fab and InlB consistent;
prior locked to log-uniform (0.0, 1.0) == [1, 10].

Provenance: MET S-BSST712 (BioImage Archive / EBI BioStudies), ThunderSTORM ``.tracked.csv``
``intensity[photon]`` column. Run under the SRM_AND_SBI_ENVY_V0 environment.

Usage:
  MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.py \
      --data-root /path/to/SPT_Movies_Complete_Data_Bank/MET
"""
import argparse
import csv
import glob

import numpy as np

from srm_and_sbi_dimer_alp.simulation_dli_support import (
    compute_brightness, compute_matrices, generate_state_trajectories)

# Fixed CTMC structure (identical to generation): brightness-state quantile grid, the bleaching
# reference-frame window, and the 50 FPS frame cadence. mu_pc cancels in the flicker dynamics, so
# its value is immaterial; sigma_pc (the flicker length scale) is swept over its prior below.
BQ = np.array([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
MU_PC, DT, PB, NPB = 386.0, 0.02, 1e-4, 100
KMAX, MIN_LEN, MIN_PAIRS, KFIT, NF, NE = 40, 40, 2000, 12, 300, 4000
SIGMA_PC_GRID = (0.42, 0.60, 0.80)   # spans the sigma_pc prior; 0.60 is the Fab reference


def read_tracked(path):
    """Per-track (frames, intensity) from a ThunderSTORM .tracked.csv (>= MIN_LEN localizations)."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        fi, ii, ti = h.index("frame"), h.index("intensity [photon]"), h.index("track.id")
        mx = max(fi, ii, ti)
        rows = []
        for row in r:
            if len(row) > mx:
                try:
                    rows.append((int(float(row[ti])), int(float(row[fi])), float(row[ii])))
                except ValueError:
                    pass
    rows.sort()
    out, cur, cf, ci = [], None, [], []
    for tid, fr, it in rows:
        if tid != cur:
            if cur is not None and len(cf) >= MIN_LEN:
                out.append((np.array(cf), np.array(ci)))
            cur, cf, ci = tid, [], []
        cf.append(fr)
        ci.append(it)
    if cur is not None and len(cf) >= MIN_LEN:
        out.append((np.array(cf), np.array(ci)))
    return out


def _acf(x_arr, length, csum, cnt):
    """Accumulate a gap-aware autocorrelation (NaN = gap) into (csum, cnt) up to KMAX."""
    for k in range(min(KMAX, length - 1) + 1):
        a, b = (x_arr, x_arr) if k == 0 else (x_arr[:length - k], x_arr[k:])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 5:
            csum[k] += np.sum(a[m] * b[m])
            cnt[k] += m.sum()


def _decay(csum, cnt):
    """Normalized flicker decay d(k) = rho(k)/rho(1), k = 1..KMAX (rho(1) divides out the white
    lag-0 spike, leaving the pure-flicker shape)."""
    rho = csum / np.maximum(cnt, 1)
    rho = rho / rho[0]
    return rho[1:] / rho[1]


def _one_over_e(d):
    """Interpolated lag (frames) at which the flicker decay crosses 1/e -- a window-independent
    correlation-time summary."""
    thr = 1 / np.e
    below = np.where(d < thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    return j + (d[j - 1] - thr) / (d[j - 1] - d[j]) + 1


def data_decay(tracks):
    """Pooled data flicker decay + its 1/e correlation time, and the empirical track-span list."""
    csum, cnt, spans = np.zeros(KMAX + 1), np.zeros(KMAX + 1), []
    for fr, it in tracks:
        if len(fr) < MIN_LEN or np.any(it <= 0):
            continue
        x = np.log(it)
        x = x - np.polyval(np.polyfit(fr, x, 1), fr)
        length = int(fr.max() - fr.min() + 1)
        spans.append(length)
        arr = np.full(length, np.nan)
        arr[fr - fr.min()] = x
        _acf(arr, length, csum, cnt)
    if cnt[1] < MIN_PAIRS:
        return None, np.nan, np.array(spans)
    d = _decay(csum, cnt)
    return d, _one_over_e(d) * DT, np.array(spans)


def model_decay(lam, sigma_pc, spans, nfmax):
    """Matched-length model flicker decay at (lambda_rate, sigma_pc): the trajectories are cut to
    the empirical span distribution and detrended identically, nulling the finite-length bias."""
    bright = compute_brightness(BQ, MU_PC, sigma_pc, 0)
    _q, p = compute_matrices(MU_PC, sigma_pc, BQ, PB, NPB, DT, float(lam))
    st = generate_state_trajectories(nfmax, NE, p, BQ, MU_PC, sigma_pc, 0, seed=0)
    b = bright[st]
    good = np.all(b > 0, axis=0)
    x = np.log(b[:, good])
    ne = x.shape[1]
    rng = np.random.default_rng(123)
    lens = np.clip(rng.choice(spans, ne), 10, nfmax)
    csum, cnt = np.zeros(KMAX + 1), np.zeros(KMAX + 1)
    for e in range(ne):
        length = int(lens[e])
        s0 = int(rng.integers(0, nfmax - length + 1))
        seg = x[s0:s0 + length, e]
        t = np.arange(length, dtype=float)
        seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
        _acf(seg, length, csum, cnt)
    return _decay(csum, cnt)


def derive(condition, tracks):
    """Shape-match (primary) + 1/e (cross-check) lambda_rate for one condition, across sigma_pc."""
    d_data, tau_data, spans = data_decay(tracks)
    if d_data is None:
        print(f"  {condition}: insufficient data")
        return
    nfmax = int(min(np.percentile(spans, 99), 400)) + 5
    lams = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    ll = np.log(lams)
    print(f"  {condition}: {len(spans)} tracks, tau_corr(1/e) = {tau_data:.3f} s")
    for spc in SIGMA_PC_GRID:
        dm = {lam: model_decay(lam, spc, spans, nfmax) for lam in lams}
        l2 = np.array([np.sum((dm[lam][:KFIT] - d_data[:KFIT]) ** 2) for lam in lams])
        i = int(np.argmin(l2))
        if 0 < i < len(lams) - 1:
            a, b, c = l2[i - 1], l2[i], l2[i + 1]
            denom = a - 2 * b + c
            off = 0.5 * (a - c) / denom if denom != 0 else 0.0
            lam_shape = float(np.exp(ll[i] + off * (ll[i + 1] - ll[i])))
        else:
            lam_shape = float(lams[i])
        oes = np.array([_one_over_e(dm[lam]) for lam in lams])
        order = np.argsort(oes)
        lam_oe = float(np.exp(np.interp(_one_over_e(d_data), oes[order], ll[order])))
        tag = "  <- Fab reference" if abs(spc - 0.60) < 1e-9 else ""
        print(f"    sigma_pc={spc:.2f}: shape-match lambda_rate={lam_shape:5.2f} "
              f"(log10 {np.log10(lam_shape):+.2f}) ; 1/e cross-check={lam_oe:5.2f}{tag}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True,
                   help="MET localization root holding <condition>/<cell>/tracks/*.tracked.csv "
                        "(ThunderSTORM tables from the public accession S-BSST712).")
    p.add_argument("--conditions", nargs="+", default=["Fab", "InlB"],
                   help="condition subdirectories to derive (default: Fab InlB).")
    args = p.parse_args(argv)
    print("Flicker-rate derivation (DETECTOR_WORKFLOW.md sec. 6.3); reads only.")
    for cond in args.conditions:
        paths = sorted(glob.glob(f"{args.data_root}/{cond}/*/tracks/*.tracked.csv"))
        tracks = [t for path in paths for t in read_tracked(path)]
        derive(cond, tracks)


if __name__ == "__main__":
    main()
