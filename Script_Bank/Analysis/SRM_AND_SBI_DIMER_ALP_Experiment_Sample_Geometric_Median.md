# Experiment Sample Geometric Median

Companion to `SRM_AND_SBI_DIMER_ALP_Experiment_Sample_Geometric_Median.py`. It reduces the
Experiment stage's cloud of MAP estimates on real MET single-particle-tracking recordings to a
single representative parameter vector, and reports how that vector differs from the naive
per-dimension summary.

## Why not the per-dimension median

Given many estimated parameter vectors, the obvious summary is to take the median of each
dimension separately. That composite is not a member of the collection, and nothing guarantees it
is even a configuration the system can occupy. The ten reaction-diffusion parameters are
correlated — the species counts constrain one another through the reactions that interconvert
them, and abundance trades off against the rates producing it — so a coordinate taken from one
recording and another taken from a different recording need never have co-occurred. For a
multimodal cloud the composite is actively misleading: it lands in the low-density valley
*between* the modes, the one configuration the data most clearly rules out.

The **Sample Geometric Median (SGM)** is the actual collection member minimizing the summed
normalized Euclidean distance to every other member. Because it is a real member, its joint
correlations are intact and it is guaranteed realizable: some window of some recording actually
produced it. Reference: Ramirez Sierra & Sokolowski, *Mach. Learn.: Sci. Technol.* **6**, 015004
(2025).

This is not a stylistic preference. On the 2 s MET data the two summaries disagree by a factor of
2.1 on the dimer count in the monomer control (see *What it found*), which is larger than most
effects the analysis is used to detect.

## The space the median is taken in

All computation happens in **absolute (physical, `10**theta`) values normalized by the absolute
prior range**. Two deliberate choices:

- **Absolute, not log.** The geometric median is not invariant under the log-to-linear transform,
  so centrality has to be defined in the space where the vector is actually used — the simulator
  consumes physical values.
- **Normalized by the prior range.** This makes the dimensions commensurable, so no parameter
  dominates the distance merely because its units are larger.

## What it reports

- **SGM vs vector-of-medians**, per parameter, in both absolute and log10 units, for the full
  collection (`unrestricted`) and for the members inside the prior box (`bounded_in_box`). Both
  variants appear because they answer different questions: `unrestricted` describes where the
  inference actually landed, `bounded_in_box` describes what a downstream stage constrained to
  that box could use. A large gap between them is itself the finding.
- **Typicality** — the Mahalanobis distance and local KDE density of each summary point. A density
  ratio near 1 means the composite did not land in a low-density region *for this collection*,
  which happens when the cloud is unimodal and weakly correlated. The SGM's guarantee does not
  depend on that: it is realizable whatever the local density.
- **Joint correlation matrix** — the structure the SGM preserves and the composite discards.
  Pooled correlations can be inflated by a between-condition shift (Simpson's paradox), so the
  report says which case it is computing.
- **Out-of-prior mass**, per parameter. On real recordings this is a genuine finding, not a
  defect: the estimates are unconstrained by the box, so mass outside it means the recordings pull
  that parameter beyond the range the prior anticipated.

Figures: `sgm_plane` (dimer abundance versus dimerization rate, with both summary points),
`sgm_corner` (all pairwise structure), `out_of_prior_mass`.

## How to run

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Sample_Geometric_Median.py \
    --total-time-seconds 2.0 [--condition pooled|MET-FAB|MET-INLB] [--max-samples N] [--dry-run]
```

CPU only, seconds to run — it reads the completed Experiment output and neither loads the
estimator nor renders videos. It is a post-hoc analysis, never wired into the stage dispatcher.

`--condition` selects the experimental condition by its scientific name: `MET-FAB` (the monomer
control), `MET-INLB` (the dimer condition), or `pooled` for both. A named condition writes to its
own `..._MET-FAB/` or `..._MET-INLB/` directory, because comparing the conditions is the point and
a shared path would let each run destroy the one before it.

The tokens `ALP` and `BET` appear nowhere in the interface, the report, or the output paths. They
survive only inside the Experiment output's stored `kinds` field — a data-schema artifact of how
those recordings were namespaced — and are translated at the boundary.

## Shared engine

The analysis runs on one shared engine, `srm_and_sbi_dimer_alp.sample_geometric_median_runner`,
over the workflow-agnostic kernel `srm_and_sbi_dimer_alp.sample_geometric_median`. The kernel knows
nothing about which parameters it summarizes: it takes vectors and a prior box. The detector
workflow's Nuisance_DLI analysis calls the same kernel, so "the median vector" means one thing
across every report this repository produces.

Per-workflow differences are carried by the spec resolver `_sgm_spec` — the parameterization
module, the alias-qualified paths, the available collection sources, and the two parameters the
plane figure shows. Biology exposes one collection (`experiment-map`, the real optimized MAPs);
the detector additionally has Nuisance_DLI sources with no biology counterpart.

## What it found on the 2 s MET data

Run on the 500-window Experiment output (250 windows per condition):

| | SGM (median vector) | vector of medians | disagreement |
|---|---|---|---|
| `count_chi`, MET-FAB (monomer control) | **22.2** | 47.6 | composite **2.1× higher** |
| `count_chi`, MET-INLB (dimer) | **329.5** | 287.9 | composite 1.14× lower |

The dimer-induction contrast between the two conditions is therefore **14.9×** measured by the
median vector, against 6.0× measured per dimension — the correct summary more than doubles the
effect. The composite's error is largest exactly where it matters most, in the control condition
whose dimer count is the baseline everything else is compared against.

Two further readings:

- **The MET-INLB dimer count is prior-limited.** Its SGM, 329.5, sits above the prior's upper bound
  (10^2.5 = 316), and **41.6 %** of MET-INLB windows fall above that bound on `count_chi` (against
  1.2 % for MET-FAB). The model cannot express as much dimer as the MET-INLB recordings ask for, so
  that number is a floor on the true value, not an estimate of it.
- **Only 8 of 250 MET-INLB windows lie fully inside the prior box** (against 107 of 250 for
  MET-FAB). The `bounded_in_box` variant for MET-INLB therefore rests on 8 vectors and its density estimate is
  reported as not estimable — with 10 dimensions there are too few members to support one. Read
  MET-INLB's in-box row as indicative only.
