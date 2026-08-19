# Experimental-versus-synthetic embedding distance — usage and interpretation

Authoritative companion for **both** workflows' embedding-distance analyses:
`SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.py` (biology) and
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py` (detector). Both are thin shims over
one shared engine, `srm_and_sbi_dimer_alp.embedding_space_distance_runner`.

The analysis measures, as an objective scalar with a significance test, how far the real recordings
sit from synthetic videos in the trained network's embedding — the only representation the posterior
ever sees. If the two occupy different regions of it, every estimate produced from real recordings is
an extrapolation: confidently reported, but outside the region the network learned. This note
explains what it measures, how to run it, and how to read the result, so it can be used and
understood without reading the code.

## The measurement is shared; the reading inverts

The number is computed identically for both workflows. What it *means* is opposite, because the two
workflows infer opposite halves of the model:

- **Detector** — the imaging parameters are the inference target, so the synthetic videos are
  supposed to reproduce the real ones' *appearance*. A gap is a **realism failure of the imaging
  forward model**, and closing it is the goal of the detector calibration.
- **Biology** — the imaging is held **fixed** and the reaction-diffusion parameters are inferred, so
  a gap is **not** primarily an imaging defect. It says the real recordings carry structure the
  synthetic prior never generated: biological variation outside the prior's reach, or an imaging
  mismatch the frozen imaging vector cannot express. Neither is fixed by training longer — the
  remedies are a wider prior or a recalibrated imaging vector.

Reading a biology gap as an imaging-realism failure (or a detector gap as biological variation) would
send the work in exactly the wrong direction, which is why the two shims exist under separate names
rather than one script with a flag.

This is a post-hoc, user-driven analysis, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and is complementary to the
Experiment stage. Because it embeds videos through the trained estimator, it is a GPU step. Its design
and justification are in `DETECTOR_WORKFLOW.md` (section "Quantitative experimental-versus-synthetic
distance").

## What it measures

The detector workflow marginalizes the biology as a nuisance, so the estimator's embedding is a
representation of the imaging parameters; a distance measured in that embedding is a statement about
imaging realism, not biology. The analysis embeds two sets of videos through the trained
`Complex3DCNN` — the experimental recordings and the held-out synthetic EVAL set — and computes two
complementary two-sample statistics on the raw embeddings:

- **Maximum Mean Discrepancy (MMD)** — a kernel discrepancy (RBF kernel, median-heuristic bandwidth)
  with a permutation p-value. Zero means identical distributions; larger means a bigger gap.
- **Classifier Two-Sample Test (C2ST)** — the cross-validated accuracy of a classifier trained to
  separate the two embedding sets. An accuracy near 0.5 means indistinguishable (overlap); approaching
  1.0 means separable (a gap).

Both resample at the **recording (cell) level**. Each experimental recording is cut into model-length
windows, and windows from one recording share its acquisition and are statistically dependent.
Treating them as independent would inflate the effective sample size and make the significance
anticonservative. The MMD permutation therefore resamples whole recordings, and the C2ST uses
recording-grouped cross-validation (`GroupKFold`), reading significance from the across-fold
accuracies rather than pooled per-window predictions. Synthetic videos are independent draws, each its
own block.

## Comparisons

The imaging is a property of the microscope, not the biological condition, so the **primary**
comparison pools all experimental recordings across conditions against the synthetic set — the
aggregate imaging-realism number. The by-condition comparisons are **diagnostics**: whether each
condition individually overlaps the synthetic set, and whether the two conditions separate from each
other in the detector embedding — which they should not, since the biology is marginalized, so a
separation would indicate residual biological signal the embedding has not fully integrated out.

| comparison | role |
|---|---|
| experimental (MET-FAB + MET-INLB pooled) vs. synthetic | **primary** — aggregate imaging realism |
| experimental MET-FAB vs. synthetic | diagnostic — per-condition realism |
| experimental MET-INLB vs. synthetic | diagnostic — per-condition realism |
| experimental MET-FAB vs. MET-INLB | diagnostic — condition separation in the detector embedding |

The conditions are named `MET-FAB` (monomer control) and `MET-INLB` (dimer) on the command line, in
the figures, and in the tables. The tokens `ALP` and `BET` survive only in the on-disk recording
filenames and the stored schemas; the analysis maps each condition name to its filename token
internally (`MET-FAB` → `ALP`: Fab, the non-activating monomer control; `MET-INLB` → `BET`: InlB,
the activating dimer).

The pooled group is formed by **natural composition** by default: every window from both conditions,
in the proportions the dataset has, each recording kept as its own cell so the resampling is
unchanged. A **balanced** alternative (`--mix balanced`) instead draws an equal number of windows from
each condition, so neither dominates the aggregate; it downsamples the larger condition and is the
deliberate choice when equal weighting is wanted. The per-condition counts are always reported, so the
composition is transparent under either setting.

## How to run it

Preview first with `--dry-run`, which resolves the input and output paths and reports what would be
read or written without loading the estimator or computing.

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.py \        # or ..._DETECTOR_... \
      --total-time-seconds 2.0 --experiment-span-seconds 20 [--mix balanced] [--dry-run]

Arguments:

- `--total-time-seconds` (required) — the model window / recording duration that sets the
  `timing_label` (for example `2.0` → `2S_50FPS`), used to locate the estimator and the EVAL set and
  to name the outputs.
- `--experiment-span-seconds` — the duration of the experimental recordings to read (files named
  `Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif`, where `<KIND>` is the on-disk filename token `ALP` or
  `BET`); default `20`.
- `--kinds` — comma-separated experimental condition names (default `MET-FAB,MET-INLB`); each
  contributes a diagnostic row and all are pooled for the primary row.
- `--eval-tasks` — number of synthetic EVAL tasks to embed as the synthetic reference (default: all).
- `--chunk-step-seconds` — the sliding-window step used to dice the recordings; default the model
  window (non-overlapping chunks).
- `--mix` (`natural` default, or `balanced`) — how the pooled experimental group is composed (above).
- `--n-permutations` — MMD permutation count (default 1000).
- `--alpha` — significance level for the pass rule (default 0.05).
- `--max-cells` — cap the cells per kind (0 = all).
- `--dry-run` — resolve the paths and report what would be read and written; load nothing, compute
  nothing.

## Reading the result

For each comparison the report gives the MMD (with its permutation p-value), the C2ST accuracy (with
its 95% confidence interval and cell-grouped p-value), and a verdict. **Overlap (pass)** is concluded
when both statistics concur — the C2ST confidence interval includes 0.5 (equivalently its p-value
≥ α) *and* the MMD permutation p-value ≥ α; a **gap** is flagged when either is significant. The
verdict is reported per comparison: the primary (pooled) row is the headline imaging-realism result,
and the diagnostics qualify it.

## Visualization

The figures are deterministic and tied to the statistics; stochastic embeddings (UMAP, t-SNE) are not
used, because their layouts are seed-dependent and need not track the computed distance. Every figure
shows all groups together — experimental MET-FAB, experimental MET-INLB, and synthetic — so overlap
and clustering read directly:

- **Embedding-distance distributions** — a four-panel grid. Columns are within-group distances
  (within MET-FAB, within MET-INLB, within synthetic) and cross-group distances (each condition to
  synthetic, and MET-FAB to MET-INLB); rows are the distance geometry — the raw embedding (the
  primary geometry, arbitrary scale) on top, the L2-normalized embedding (unit sphere, cosine
  distance bounded in [0, 2]) below. In either row, a cross-group curve sitting on the within-group
  curves is overlap; a shift to larger distances is separation. The raw row holds the distances the
  MMD is built from, so its picture and the headline number agree; the L2 row is a scale check on
  whether the raw picture is a magnitude artifact, with the pooled MMD in that geometry reported as a
  secondary diagnostic.
- **C2ST score distributions** — the classifier's out-of-fold predicted probabilities for each group
  on one axis; overlapping score distributions are the graphical form of an accuracy near 0.5.
- **Inter-group mean-distance matrix** — a compact MET-FAB/MET-INLB/synthetic square panel of mean
  embedding distances, the at-a-glance summary of which groups sit close and which apart.
- **PCA / classical-MDS panels (projection, PC1 density, Shepard)** — the raw embeddings projected
  onto their top two principal components (the optimal linear distance-preserving projection), in
  three panels: (a) the projection colored by group, each axis labeled with its explained-variance
  fraction — when one component dominates, the points fall on a single curved arc (the horseshoe
  signature of one-dimensional data), not separate blobs; (b) the distribution of each group's PC1
  coordinate, the honest one-dimensional reading of where each group sits along the single real axis
  and how far it spreads; (c) a Shepard diagram of projected 2-D distance against true
  high-dimensional distance (points on or below the identity line, with an annotated correlation)
  stating how faithful the projection is. Deterministic, unlike UMAP/t-SNE, and self-diagnosing:
  trust the projection to the degree the explained variance and correlation are high, with the
  distance-distribution and matrix figures as the exact high-dimensional summary. The accompanying
  variance-spectrum table quantifies the embedding's intrinsic dimensionality.
- **PC1 parameter tracking** — when the synthetic EVAL imaging parameters are available: (a) a bar
  chart of the absolute Spearman correlation between the dominant embedding axis (PC1) and each
  imaging parameter, sorted, so the parameter driving the single dominant direction stands out; (b)
  PC1 against that top parameter. This names the imaging parameter the embedding varies along when it
  collapses toward one dimension. Rank correlation, so it is unaffected by the log/linear storage of
  the parameters.

## Outputs

Written to the workflow's `Posit/` subdirectory under the data bank, in a
`<alias>_<timing_label>_Embedding_Space_Distance/` directory. `<alias>` is
`SRM_AND_SBI_DIMER_ALP` for biology and `SRM_AND_SBI_DIMER_ALP_DETECTOR` for detector, so the two
workflows' results never collide:

- `report.md` — the per-comparison table (MMD, C2ST, p-values, verdict), the pooled-group composition,
  the PCA variance spectrum (intrinsic dimensionality), the PC1-versus-imaging-parameter ranking with
  each parameter's meaning (when the EVAL imaging parameters are present), the pooled MMD in the L2 (cosine) geometry as a secondary
  diagnostic, and the run provenance (estimator checksum, the EVAL and experimental sources, the chunk
  count, the permutation count, α).
- `figures/` — the figures above (the PC1 parameter-tracking figure is included when the EVAL imaging
  parameters are present).

## Caveats

- **These are calibrated comparisons on real recordings, not ground-truth recovery.** The recordings
  have no ground truth; the analysis measures distributional overlap in the embedding, not a
  demonstrated recovery of true parameters. Recovery is quantified only on held-out synthetic data in
  the workflow's Evaluation stage.
- **The measure diagnoses; it does not close a gap.** A measured gap is reduced only by changing the
  model (recalibrating the imaging, or widening the imaging prior) or the training (an
  embedding-alignment term), neither performed here (see `DETECTOR_WORKFLOW.md`, section "Quantitative
  experimental-versus-synthetic distance").
- **Few recordings weaken significance.** With few cells there are few permutation blocks and few
  cross-validation folds, so the tests are conservative; the point statistics remain informative.
- **The interpretation is the detector question only.** A distance here is imaging realism; the
  analogous distance under the biology workflow (imaging marginalized) is a
  biological question and a separate companion analysis.

## Known limitations of the significance machinery

The two effect sizes (the MMD value and the C2ST accuracy) are the quantities to read; the
significance machinery around them has two known construction defects, so treat verdicts
qualitatively until the corrected design lands:

- **C2ST null.** The score is plain accuracy on the pooled, imbalanced set (hundreds of
  experimental windows against thousands of synthetic videos), so the no-information baseline is
  the majority fraction (about 0.95 at production sizes), not the 0.5 the test statistic assumes;
  the accompanying p-value (a t-test of fold scores against 0.5) inherits that wrong center and is
  not valid. The observed accuracies (about 0.999) far exceed the imbalance baseline, and the
  balanced condition-versus-condition row has a true 0.5 null, so the qualitative gap conclusions
  stand.
- **MMD permutation null.** The permutation reassigns whole recording blocks while preserving the
  block COUNT, not the window count, so permuted group sizes differ from the observed ones
  (a permuted "experimental" group holds roughly a tenth of the observed windows). The dominant
  effect is a wider null — a conservative bias toward missing gaps — so a reported gap survives,
  but the printed p-values are not exact and should not rank candidate models.

The corrected design (balanced or AUC-based C2ST with a unit-level permutation null, MMD
permutations over matched unit counts, and a matched synthetic reference) replaces both; until
then, compare runs by effect size only.

## Reference

Maximum Mean Discrepancy: Gretton, Borgwardt, Rasch, Schölkopf, Smola, "A Kernel Two-Sample Test,"
*Journal of Machine Learning Research*, 2012. Classifier Two-Sample Test: Lopez-Paz and Oquab,
"Revisiting Classifier Two-Sample Tests," *ICLR*, 2017. Cell-grouped cross-validation controls the
within-recording correlation of the video windows. Experimental recordings: MET single-particle-tracking
data, BioImage Archive accession S-BSST712. The embedding, the imaging prior box, and the
detector-versus-biology framing are in `DETECTOR_WORKFLOW.md`.
