# Sample Geometric Median of a Nuisance_DLI pool — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI_Sample_Geometric_Median.py`. The Detector calibrates
the imaging model against experimental recordings and the calibration is stored as the `Nuisance_DLI`
artifact — a cloud of imaging parameter vectors (the six photophysics parameters). This analysis reduces
that cloud to a single representative vector while keeping its joint structure intact, and contrasts
that vector with the naive per-dimension summary. This note explains what it computes, how to run it,
and how to read the result, so it can be used and understood without reading the code.

This is a post-hoc, user-driven analysis, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and is complementary to the
`Nuisance_DLI` construction step, which produces the artifact this consumes. It reads the built artifact
only, so it needs neither the estimator nor a GPU — it runs on any machine. The design of the nuisance
it summarizes is in `DETECTOR_WORKFLOW.md`, section "Nuisance and artifact design"; the construction of
the artifact is in the companion `SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.md` beside this note.

## What it computes

The imaging parameters are correlated: a spot's peak brightness scales with its width, and a
summed-dimer signal shifts brightness and flicker together, so the pool occupies a tilted, sometimes
multimodal region of the six-dimensional space rather than an axis-aligned box. The right one-point
summary of such a cloud is therefore the median **vector**, not the vector of per-dimension medians —
the latter takes each coordinate independently and can return a combination that lies off the joint
manifold, a configuration no acquisition ever produced.

The correlation-preserving summary is the **Sample Geometric Median (SGM)**: the actual pool member that
minimizes the sum of normalized distances to every other member (Ramirez Sierra & Sokolowski 2025; see
Reference). Because it is an actual member, the returned vector is a real, co-occurring configuration
with all of its cross-parameter correlations intact — a robust center-of-mass estimate of location,
distinct from the maximum-a-posteriori point (the mode) and from the marginal medians. The analysis
computes it two ways for comparison against the per-dimension vector of medians, and reports which
coordinates the marginal summary would have misrepresented.

Four conventions matter and are deliberate:

- **Posterior samples versus MAP estimates — and how each weights the vectors.** The geometric median
  can summarize either one MAP estimate per acquisition or the full posterior-sample pool, and that
  choice sets the weighting; no per-vector weight is applied beyond it. The MAP collection gives each
  acquisition exactly one vector, so every recording carries equal weight and none dominates. The
  posterior collection weights every draw equally, which amounts to weighting by sample density: an
  acquisition with a sharp posterior concentrates its draws, so the denser cluster is pulled toward and
  the summary is biased to the majority. Both are legitimate reads of the pool; they coincide when the
  per-acquisition posteriors are sharp and diverge when they are broad. The MAP collection's one vector
  per acquisition is the real optimized MAP — the Detector Experiment output, or a MapEstimate pool —
  or, as an explicit named alternative, the per-window Sample Geometric Median of that window's draws
  (`--map-source`); it is never a silent stand-in.

- **One condition or both.** The calibration pool mixes the two experimental conditions (MET-FAB, the
  monomer control, and MET-INLB, the dimer), which shift the apparent PSF width and brightness. Summarizing
  the pool as a whole averages across them, and the joint correlations can then carry a between-condition
  (Simpson) component; restricting to a single condition (`--condition`) removes that, at the cost of a
  smaller collection. The restriction reads the collection's own per-row condition labels — so it operates
  on the data as delivered, not a re-derived split — and a condition requested on an unlabeled collection
  is refused rather than silently answered from the full pool.
- **Absolute space.** The geometric median is not invariant under the log-to-linear transform, so the
  space in which distances are measured is a real modeling choice, not a formatting detail. The pool is
  inferred and stored in log10, but the forward renderer consumes physical values, so centrality is
  defined where the vector is actually used: the analysis works on the physical values (ten to the power
  of the stored log10) and normalizes each dimension by its **absolute** prior range. The per-dimension
  median, by contrast, is transform-equivariant — identical whether taken in log10 or physical space —
  so only the SGM depends on this choice.
- **Full pool versus in-box subcollection.** The geometric median is reported for the whole pool and,
  separately, for the subcollection whose vectors lie entirely inside the imaging prior box. The in-box
  restriction is the analog, for a posterior sample pool, of selecting a subcollection by a quality
  threshold: it isolates the central configuration that respects the prior, and its size relative to the
  full pool measures how much of the calibration mass the prior box excludes.

Alongside these two vectors the analysis reports the out-of-prior mass per parameter (the genuine
out-of-bounds set), the joint Pearson correlation matrix the marginal summary discards, and a relative
typicality read (local density and Mahalanobis distance) for each vector of medians.

## How to run it

Preview first with `--dry-run`, which resolves the artifact and output paths and reports what would be
read or written without loading or computing anything.

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI_Sample_Geometric_Median.py \
      --total-time-seconds 2.0 [--collection map|posterior] [--map-source experiment|window-sgm] \
      [--condition pooled|ALP|BET] [--pool-source artifact|cache] [--dry-run]

Arguments:

- `--total-time-seconds` (required) — the model window / recording duration that sets the `timing_label`
  (for example `2.0` gives `2S_50FPS`), used to locate the `Nuisance_DLI` artifact and to name the
  outputs.
- `--pool-source` (`artifact` default, or `cache`) — `artifact` reads the built `Nuisance_DLI`: its
  stored sample matrix for a `raw` or `map_estimate_pool` representation, or draws from it for a
  `gaussian`/`box` representation. `cache` reads the raw posterior-sample pool matrix beside the
  artifact — the richest cloud, before any representation choice was applied.
- `--collection` (`map` default, or `posterior`) — the collection the geometric median summarizes:
  `map` one estimate per acquisition (see `--map-source`), `posterior` the posterior-sample pool (all
  draws, density-weighted). Both stay on the CPU.
- `--map-source` (`experiment` default, or `window-sgm`) — for `--collection map`, where the
  one-estimate-per-acquisition comes from. `experiment` the REAL optimized MAPs — a `MapEstimate` pool
  cache if present, otherwise the Detector Experiment stage's MAP output — failing loudly if neither
  exists, never substituting a stand-in silently. `window-sgm` the per-window Sample Geometric Median
  (the medoid of each window's posterior draws), an explicit samples-derived estimate computed CPU-only
  from the posterior-sample pool. (This `window-sgm` is the SGM applied per window; it was previously a
  silent fallback and is now an explicit, named choice.)
- `--condition` (`pooled` default, or `ALP`/`BET`) — restrict the collection to one experimental
  condition before summarizing: `pooled` both, `ALP` = MET-FAB (monomer control), `BET` = MET-INLB
  (dimer). The restriction reads the collection's own per-row condition labels, so it needs a labeled
  pool (a fresh build writes them; migrate a legacy pool with the Nuisance_DLI build's
  `--migrate-pool-labels`) or the labeled experiment MAP; a real condition on an unlabeled collection
  is a loud error, not a silent full-pool summary.
- `--n-samples` — number of vectors to draw when the artifact is a `gaussian`/`box` representation
  (default 200000); ignored when a sample matrix is already stored.
- `--max-samples` — cap the pool by uniform subsampling before the geometric median, for tractability
  (0 uses all). A numerical detail, not a scientific knob; the result is a real pool member either way.
- `--seed` — seed for the subsampling used in the density read and the figures. The geometric median
  itself is deterministic.
- `--dry-run` — resolve paths and report what would be read and written; load nothing, compute nothing.

If the artifact is absent the run stops with a message naming the `Nuisance_DLI` construction step to
run first, rather than fabricating a summary.

## Reading the result

The report gives, for the full pool and the in-box subcollection, the SGM and the vector of medians
side by side in both physical and log10 units, and states whether each lies inside the prior box. The
load-bearing distinction is not which one sits at higher density but which one is realizable: the SGM is
always an actual acquisition's configuration, so it is guaranteed to be a coherent parameter set; the
vector of medians is assembled coordinate by coordinate and carries no such guarantee — it can leave the
prior box on a single dimension even when every coordinate is individually reasonable.

The typicality table reports, per variant, the normalized distance between the two vectors and each
one's Mahalanobis distance and local density. These are relative diagnostics: when the two vectors sit
close and at comparable density, the marginal summary is not badly wrong for that pool, and the SGM's
value is its realizability rather than a density advantage; when they diverge, the correlation matrix
names the joint structure the marginal summary broke. Pooled correlations can be inflated by a shift
between subpopulations (Simpson's paradox), so a strong pooled correlation that weakens within a
subpopulation is a sign the pool mixes distinct regimes — read the correlation matrix together with the
out-of-prior mass, which flags whether the calibration is pressing against a prior bound.

## Visualization

The figures are deterministic. Every figure draws the pool together with both summary vectors so the
contrast reads directly:

- **PSF width versus brightness** — the pool in the (`mu_r`, `mu_pc`) plane with the SGM (a real sample)
  and the vector of medians marked, and the prior box drawn. When the pool is multimodal, the marginal
  summary drifts toward the low-density region between modes while the SGM stays on a real configuration.
- **Pairwise corner** — the pool across all six imaging parameters with the two vectors overplotted.
  Off-diagonal tilt is the joint correlation the SGM keeps and the vector of medians, taken per
  dimension, ignores.
- **Out-of-prior mass** — the fraction of pool draws outside the prior box per parameter, the genuine
  out-of-bounds set, nonzero only when the pool was built unrestricted.

## Outputs

Written to the Detector-namespaced `Posit/` subdirectory under the data bank (`<alias>` is
`SRM_AND_SBI_DIMER_ALP_DETECTOR`), in a `<alias>_<timing_label>_Nuisance_DLI_Sample_Geometric_Median/` directory:

- `report.md` — the checks, the out-of-prior mass table, the SGM-versus-vector-of-medians table per
  variant (physical and log10), the typicality table, the joint correlation matrix, and the run
  provenance (the artifact path, the representation choice and pool mode, the pool size and in-box
  count, and the space convention).
- `figures/` — the figures above.

## Caveats

- **These are calibrated estimates on real recordings, not ground-truth recovery.** The recordings have
  no ground truth; the pool describes where the calibrated imaging posterior places its mass, and the
  Sample Geometric Median summarizes that. Recovery against known values is quantified only on held-out synthetic
  data in the Detector Evaluation stage.
- **The absolute-space choice is deliberate and material.** Because the geometric median is not
  transform-invariant, the SGM computed here differs from one computed in log10 space; the physical
  space is chosen because the renderer consumes physical values. The vector of medians is unaffected by
  the choice, so a divergence between the two spaces is a property of the SGM alone.
- **This summarizes the pool as delivered; it does not attribute the spread.** When a pool mixes
  distinct acquisition regimes (for example a monomer condition and a dimer condition, which shift the
  apparent width and brightness), a single representative vector averages across them. Splitting the pool by
  regime requires the artifact's construction order and condition labels, which this general summary does
  not reconstruct; the out-of-prior mass and the correlation matrix are the signals that such mixing is
  present.
- **The tractability path does not change the answer's meaning.** For a large pool the geometric median
  is found by Weiszfeld's iteration and snapped to the nearest actual member rather than by an exhaustive
  pairwise search; both return a real pool member, and the report records which path was taken.
- **The brightness-flicker probability is weakly identified per window.** Its coordinate in any single
  vector is the least constrained of the six and should be read as broad regardless of the summary used.

## Reference

Sample Geometric Median: Ramirez Sierra and Sokolowski, "Comparing AI versus optimization workflows for
simulation-based inference of spatial-stochastic systems," *Machine Learning: Science and Technology*
(2025), doi:10.1088/2632-2153/ada0a3 — the geometric-median selection over normalized parameter vectors,
realized as an actual sample, that this analysis applies to the imaging nuisance. Geometric median
algorithm: Weiszfeld, "Sur le point pour lequel la somme des distances de n points donnés est minimum,"
*Tôhoku Mathematical Journal* 43, 355 (1937). The nuisance artifact and its construction are described
in `DETECTOR_WORKFLOW.md`, section "Nuisance and artifact design," and in the companion note
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.md`. Experimental recordings: MET single-particle-tracking
data, BioImage Archive accession S-BSST712.
