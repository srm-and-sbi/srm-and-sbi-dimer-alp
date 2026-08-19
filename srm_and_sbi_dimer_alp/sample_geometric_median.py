"""Sample Geometric Median: the correlation-preserving summary of a cloud of parameter vectors.

Workflow-agnostic kernel, shared by the biology and detector analyses. Nothing here knows which
parameters it is summarizing: every function takes vectors, a prior box, and returns numbers.

THE METHOD. Given a collection of parameter vectors, the honest single-vector summary is the median
VECTOR, not the vector of per-dimension medians. Real parameter clouds are correlated -- in imaging a
spot's peak brightness scales with its width; in the reaction-diffusion model the species counts and
the rates that interconvert them move together -- so taking each dimension's median independently
composes a vector whose coordinates never co-occurred. That composite can land in a region of
parameter space the collection never visited, and for a multimodal cloud it drifts into the
low-density valley BETWEEN the modes, which is the worst possible representative.

The Sample Geometric Median (SGM) avoids this by construction: it is the actual collection member
minimizing the summed normalized Euclidean distance to every other member. Being a real member, its
joint correlations are intact and it is guaranteed realizable -- some acquisition (or some draw)
actually had that configuration. Reference: Ramirez Sierra & Sokolowski, Mach. Learn.: Sci. Technol.
6, 015004 (2025).

SPACE. All computation is in ABSOLUTE (physical, ``10**theta``) values normalized by the absolute
prior range. The geometric median is not invariant under the log-to-linear transform, so centrality
must be defined in the space where the vector is actually used -- the simulator consumes physical
values. Normalizing by the prior range makes the dimensions commensurable, so no parameter dominates
the distance merely because its units are larger.
"""
from __future__ import annotations

import numpy as np

# Above this many members the exact medoid (an O(N^2) pairwise-distance matrix) stops being
# tractable and Weiszfeld's iteration is used instead, snapped back to the nearest real member so
# the result is still an actual collection member rather than a synthetic point.
EXACT_MEDOID_CAPACITY = 20000
WEISZFELD_ITERS = 2000

# Tractability / rendering knobs (not scientific parameters).
KDE_SUBSAMPLE = 50000            # cap on points used to estimate local density for the typicality read
FIGURE_SUBSAMPLE = 5000          # cap on scatter points drawn per figure


def sample_geometric_median(vecs_abs, range_abs):
    """Index (and method) of the collection member closest to the geometric median, in prior-range-
    normalized absolute space -- the correlation-preserving median VECTOR (an actual member, so its
    joint correlations are intact). Exact medoid when tractable, else Weiszfeld's iteration snapped
    to the nearest member. (Ramirez Sierra & Sokolowski 2025.)"""
    from scipy.spatial.distance import pdist, squareform     # lazy: keep the sampling path numpy-only
    m = np.asarray(vecs_abs, dtype=float) / range_abs
    if m.shape[0] <= EXACT_MEDOID_CAPACITY:
        return int(np.argmin(squareform(pdist(m)).sum(0))), "exact_medoid"
    y = m.mean(0)
    for _ in range(WEISZFELD_ITERS):
        dist = np.maximum(np.linalg.norm(m - y, axis=1), 1e-12)
        weight = 1.0 / dist
        y_next = (weight[:, None] * m).sum(0) / weight.sum()
        if np.linalg.norm(y_next - y) < 1e-10:
            y = y_next
            break
        y = y_next
    return int(np.argmin(np.linalg.norm(m - y, axis=1))), "weiszfeld_snap"


def typicality(vecs_abs, range_abs, point_abs, rng):
    """Local density and Mahalanobis distance of a single point relative to the cloud, in prior-range-
    normalized absolute space. Density is a subsampled Gaussian-kernel estimate; both are relative
    reads -- they compare two candidate summary points against each other, not against an absolute
    scale."""
    m = np.asarray(vecs_abs, dtype=float) / range_abs
    x = np.asarray(point_abs, dtype=float) / range_abs
    mean = m.mean(0)
    cov_inv = np.linalg.pinv(np.cov(m, rowvar=False))
    maha = float(np.sqrt((x - mean) @ cov_inv @ (x - mean)))
    # A kernel density estimate needs more members than dimensions -- below that the covariance is
    # singular by construction and no smoothing bandwidth exists. This is a real situation, not an
    # edge case to be papered over: a condition whose estimates mostly fall outside the prior box
    # leaves an in-box subcollection of only a handful of vectors. The honest result is that the
    # density is not estimable, reported as NaN, while the Mahalanobis distance (which uses a
    # pseudo-inverse) still carries information. Crashing here would lose the whole report over a
    # secondary diagnostic.
    from scipy.stats import gaussian_kde
    if m.shape[0] <= m.shape[1]:
        return maha, float("nan")
    sub = m if m.shape[0] <= KDE_SUBSAMPLE else m[rng.choice(m.shape[0], KDE_SUBSAMPLE, replace=False)]
    try:
        density = float(gaussian_kde(sub.T)(x.reshape(-1, 1))[0])
    except (np.linalg.LinAlgError, ValueError):
        density = float("nan")
    return maha, density


def summary_vectors(pool_log, low, high, rng):
    """SGM and vector-of-medians (absolute + log10) for the full collection and its in-box
    subcollection, plus the typicality of each summary point. Returns ``(variants, in_box_mask)``.

    Both variants are reported because they answer different questions. ``unrestricted`` summarizes
    everything the collection contains, including estimates that fell outside the prior box -- the
    honest description of where the inference actually landed. ``bounded_in_box`` summarizes only the
    members the prior admits, which is what a downstream stage constrained to that box can use. A
    large gap between the two is itself the finding: it says the collection's mass sits substantially
    outside the box it was meant to occupy.
    """
    pool_log = np.asarray(pool_log, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    a_range = 10.0 ** high - 10.0 ** low
    in_box = np.all((pool_log >= low) & (pool_log <= high), axis=1)
    variants = [("unrestricted", np.ones(pool_log.shape[0], dtype=bool)),
                ("bounded_in_box", in_box)]
    out = []
    for name, mask in variants:
        subset_log = pool_log[mask]
        if subset_log.shape[0] == 0:
            out.append(dict(variant=name, n=0))
            continue
        subset_abs = 10.0 ** subset_log
        idx, method = sample_geometric_median(subset_abs, a_range)
        sgm_abs = subset_abs[idx]
        vom_abs = np.median(subset_abs, axis=0)
        maha_s, dens_s = typicality(subset_abs, a_range, sgm_abs, rng)
        maha_v, dens_v = typicality(subset_abs, a_range, vom_abs, rng)
        out.append(dict(
            variant=name, n=int(subset_log.shape[0]), method=method,
            sgm_abs=sgm_abs, sgm_log=np.log10(sgm_abs),
            # Computed, not asserted: the SGM of the UNRESTRICTED collection is a member of the
            # full pool and can lie outside the prior box (the bounded_in_box variant is inside
            # by construction). A hardcoded True here once printed a false "in-box" line for
            # SGM vectors that were genuinely out of support.
            sgm_in_box=bool(np.all((np.log10(sgm_abs) >= low) & (np.log10(sgm_abs) <= high))),
            vom_abs=vom_abs, vom_log=np.log10(vom_abs),
            vom_in_box=bool(np.all((np.log10(vom_abs) >= low) & (np.log10(vom_abs) <= high))),
            box_dist=float(np.linalg.norm((sgm_abs - vom_abs) / a_range)),
            maha_sgm=maha_s, maha_vom=maha_v,
            density_ratio=(dens_v / dens_s if dens_s and np.isfinite(dens_s) else float("nan"))))
    return out, in_box
