# Nuisance_DLI construction — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py`. The Detector calibrates the imaging
model on real recordings; this analysis turns that calibration into the `Nuisance_DLI` — the
samplable imaging distribution the production run draws from when it marginalizes the imaging block.
This note explains what the step builds, how to run it, and how to choose among its options, so the
construction can be used and understood without reading the code.

This is a post-hoc, user-driven analysis step, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and is **self-contained**: it loads
the trained detector estimator, reads the real recordings, and runs the estimator itself. It does not
depend on the Detector Experiment stage's output — running the estimator here lets the step own the
chunk windowing. Because it runs the estimator, both of its modes are a GPU step.

## What it builds

The trained detector estimator is a **required** prerequisite: there is no way to declare a
`Nuisance_DLI` a priori, because its content *is* the calibration result. The step reads every
experimental recording, cuts each into model-length windows, draws the posterior conditioned on each
window, and pools those draws across every window of every recording into the **posterior sample
pool**.

The pool is a *mixture*, and that is the key modeling point. The real recordings genuinely differ in
their imaging conditions, so the quantity we want is the *distribution of imaging parameters across
the recordings* — not a single posterior conditioned on all of them at once. Conditioning on multiple
observations jointly (an amortized posterior's iid mode) would assume they share one parameter vector,
which would collapse the across-recording spread that is precisely the nuisance we are after. Pooling
the per-window posteriors is therefore not a workaround; it is the correct representation. Being a
mixture, the pool is *empirical* — a cloud of sample vectors, not a closed-form density.

One user choice, `posterior_sample_pool_choice`, decides how that pool becomes the samplable artifact:

| choice | what it is | cross-parameter correlations |
|---|---|---|
| **`raw`** (default) | resample the pool per whole vector | **preserved exactly** |
| **`map_estimate_pool`** | resample the pooled per-window MAP (best-fit) estimates per whole vector | preserved (point estimates only) |
| **`gaussian`** | a full-covariance multivariate Gaussian fit to the pool | linear only (via the covariance) |
| **`box`** | a per-parameter uniform over quantiles of the pool | none (independent per dimension) |
| **`box_user`** | a per-parameter uniform over user-set ranges | none |

**Why `raw` is the faithful default.** Each entry in the pool is a complete parameter vector whose
components were drawn jointly, so resampling *whole vectors* preserves the joint structure exactly —
including the degeneracies the calibration constrains. The dominant one in this model is concrete: the
videos identify essentially only the ratio `γ = g/C` of the EM gain `g` (`kappa_g`) to the
electrons-per-ADU conversion `C` (`kappa_c`), so `g` and `C` ride a tight ridge in the posterior.
Because θ is in log₁₀, that ridge is a straight line (`log g − log C ≈ const`). Sampling the two
parameters independently — as any per-dimension representation does — would pair a gain from one draw
with a conversion from another and break the very `γ` the data pins down, injecting physically
impossible detectors into the training set. `raw` never does this; it only ever emits vectors that
actually occurred.

**Why the other forms are lossy, in order.** `gaussian` fits a single tilted ellipsoid to the pool: it
keeps the *linear* correlations (the `g/C` ridge survives in the covariance's off-diagonal) but cannot
represent multimodality or a curved ridge, and it smooths the cloud into one unimodal blob. `box` and
`box_user` are per-parameter uniforms, so they discard all correlation — they sample box corners the
joint posterior never visits. The fidelity order is thus `raw` (exact) → `gaussian` (linear only) →
`box`/`box_user` (none). The compensating benefits are compactness (a Gaussian is `mean` + covariance;
a box is `[low, high]`), smoothness (a genuine continuous density rather than a resampled cloud), and,
for the boxes, hard bounds.

**`pool_mode`** (`bounded` by default, or `unrestricted`) is identical to the convention used by the
Detector Evaluation and Experiment stages: `bounded` rejection-samples the pool within the imaging
prior box; `unrestricted` takes the raw normalizing-flow draws, whose mass may lie outside it. It
governs how the pool is drawn, so it shapes `raw`, `gaussian`, and `box` (and the MAP pool). The
calibration-faithful forms (`raw`, `map_estimate_pool`, `gaussian`) are then **not clipped** — under
`unrestricted` they can legitimately place mass outside the prior box, matching Evaluation and
Experiment. Only `box`/`box_user` are constrained to the prior box, clamped at build.

**The range rule.** A user supplies per-parameter ranges only for `box_user`. `box` derives its ranges
from pool quantiles (the 5th/95th percentiles by default) clamped to the prior box, and
`raw`/`map_estimate_pool`/`gaussian` take no range at all. So the spec asks for per-parameter ranges if
and only if the choice is `box_user`.

## How to run it

Preview either mode first with `--dry-run`, which resolves the input and output paths and reports what
would be read or written without loading the estimator or computing.

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
      --total-time-seconds 2.0 --experiment-span-seconds 20 --emit-template [--pool-mode unrestricted] [--dry-run]

    (edit the emitted *_Nuisance_DLI_Spec.toml: set posterior_sample_pool_choice (and pool_mode,
     pre-filled from --pool-mode); for box_user, also set the [imaging.<KEY>] ranges)

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
      --total-time-seconds 2.0 --experiment-span-seconds 20 --build [--dry-run]

Arguments:

- `--total-time-seconds` (required) — the model window / recording duration that sets the `timing_label`
  (for example `2.0` → `2S_50FPS`), used to locate the estimator and name the outputs. It is the same
  timing knob every pipeline stage takes; the label is always derived from it, never set by hand.
- `--experiment-span-seconds` — the duration of the real recordings to read (the files are named
  `Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif`); default `20`.
- `--kinds` — comma-separated recording kinds to pool (default `ALP,BET`). The construction always pools
  across kinds; a by-kind split is only ever a diagnostic, never the constructed artifact, because the
  imaging is a property of the microscope, not the biological condition.
- `--chunk-step-seconds` — the sliding-window step; default is the model window (non-overlapping
  chunks). A smaller step yields more, overlapping chunks per recording.
- `--pool-mode` (`bounded` default, or `unrestricted`) — an emit-only knob: the mode the posterior pool
  is drawn under, written into the emitted spec as its `pool_mode` default so the finalized spec records
  how emit sampled. `--build` reads `pool_mode` from the spec, not this flag. It matches the `pool_mode`
  vocabulary of the Evaluation and Experiment stages. Use `unrestricted` when the posterior sits largely
  outside the prior box, where bounded rejection barely accepts.
- `--n-per-chunk` — posterior draws per chunk, both modes (default: the evaluation config's
  `posterior_samples`).
- `--repool` — force recomputing the pool on the GPU even when a fresh cache exists (see Caching).
- `--max-cells` — cap the cells per kind (0 = all).
- `--dry-run` — resolve the paths and report what would be read and written; load nothing, compute
  nothing.

## Caching

The pool — the posterior draws over every chunk — is the GPU cost, and it is cached so exploring
choices is cheap. `--emit-template` computes the pool once and caches it as
`<alias>_<timing_label>_Nuisance_DLI_<Kind>Pool.npz`; `--build` then reuses that cache (no GPU) to apply
the chosen representation. The cache is keyed on the build inputs (kinds, span, step, draws per chunk,
`pool_mode`, cell cap) and the estimator's weight checksum, so it is reused only when those are unchanged
and recomputed automatically when a different estimator is swapped in.

`raw`, `gaussian`, and `box` all draw from the one posterior-sample pool, so switching among them costs
nothing beyond re-applying the representation; `map_estimate_pool` uses a separate MAP pool (its own GPU
pass); `box_user` needs no pool at all. Pass `--repool` to force a recompute even when a fresh cache
exists.

## Outputs

Written to the Detector-namespaced `Posit/` subdirectory under the data bank (here `<alias>` is the
Detector-namespaced project alias `SRM_AND_SBI_DIMER_ALP_DETECTOR`):

- **Emit-template** writes `<alias>_<timing_label>_Nuisance_DLI_Spec.toml` — the value-based
  specification (all values in log₁₀). Its `[block]` sets `posterior_sample_pool_choice`, `pool_mode`,
  and `box_quantiles`; its `[imaging.<KEY>]` ranges are pre-filled with the pool's 5th/95th percentiles
  as suggestions and are read only by `box_user`. A provenance comment records the estimator, the pool
  source, the chunk count, and the draws per chunk. The file is authoritative only once the user saves
  their edits.
- **Build** writes `<alias>_<timing_label>_Nuisance_DLI.npz` — the built, self-contained artifact,
  carrying its `parameter_keys` manifest, the chosen representation's numeric parameters (a box's
  `[low, high]`, a resampled sample matrix, or a Gaussian's `mean`/`covariance`), the prior box, and the
  `posterior_sample_pool_choice` and `pool_mode` it was built with. Being self-contained, it is sampled
  at generation with neither the estimator nor the recordings.
- **Build** also writes a `<alias>_<timing_label>_Nuisance_DLI_Analysis/` directory beside the artifact:
  a `report.md` (provenance and a per-parameter marginal summary — median, 5th/95th percentiles, prior
  box, and the fraction of each parameter's mass outside the prior box) and
  `figures/nuisance_marginals.png` (the 1-D marginal of each imaging parameter, with the prior bounds
  marked). This is the record a person reads to judge, and choose between, nuisance constructions.

Any downstream stage obtains the `Nuisance_DLI` through the module's gate (`require_nuisance_dli`),
which **loads** the built artifact and fails loud — naming this analysis — if it is absent. The gate
does not rebuild: the build is this step's job.

## Choosing a representation

- **`raw`** — the recommended default. Use it unless you have a specific reason not to: it is the only
  form that reproduces the calibrated imaging distribution faithfully, correlations and all.
- **`gaussian`** — when a compact, smooth, continuous density is wanted and the pool is plausibly
  unimodal. It preserves the linear correlations but averages away any second mode.
- **`box`** — when a bounded, correlation-free uniform is wanted, sized automatically from the pool's
  central mass (the quantiles). Conservative and simple.
- **`box_user`** — the manual escape hatch: a uniform over ranges you set (for example, to reproduce a
  fixed-imaging production by setting each range to a single value). Clamped to the prior box.
- **`map_estimate_pool`** — the point-estimate sibling of `raw`: it pools the per-window MAP (best-fit)
  vectors instead of full posterior draws, so it captures the across-recording spread of the point
  estimates but discards the within-window posterior width.

## Caveats

- **Only `raw` preserves the joint distribution.** Every other choice trades correlation fidelity for
  compactness, smoothness, or bounds; the report and this note make that trade explicit so it is a
  deliberate choice, not an accident. The `g/C` degeneracy above is the concrete stake.
- **The pool is a mixture, and may be multimodal.** It is pooled across recordings with different
  imaging, so a single Gaussian (`gaussian`) can misrepresent it by placing mass between modes. Prefer
  `raw` when in doubt.
- **`unrestricted` can produce out-of-prior draws.** This is intentional and matches the Evaluation and
  Experiment stages; it is not clipped. Use `bounded` (the default) to keep the pool within the prior
  box by rejection.
- **These are calibrated estimates on real data, not ground-truth recovery.** The recordings have no
  ground truth; the pool describes where the calibrated imaging posterior places its mass, not a
  demonstrated recovery of true imaging parameters. Recovery is quantified only on held-out synthetic
  data with known values, in the Detector Evaluation stage.
- **Prior-box clamping is logged, not silent.** For `box`/`box_user`, any parameter range the prior-box
  clamp reduces is reported and counted at build.

## Reference

The negative log-likelihood the pool is drawn under is the logarithmic score, a strictly proper scoring
rule (Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," *Journal of
the American Statistical Association*, 2007). The multivariate-Gaussian fit stores the sample mean and
covariance; the boxes are `BoxUniform` distributions over log₁₀ ranges.

Real recordings: MET single-particle-tracking data, BioImage Archive accession S-BSST712. The imaging
model and the `g/C` (`kappa_g`/`kappa_c`) degeneracy are described under the EMCCD noise model in
`REFERENCE_EMCCD_NOISE_MODEL.md`; the nuisance and artifact design, the value-based parameter table, and
the imaging prior box are in `DETECTOR_WORKFLOW.md` (section "Nuisance and artifact design"); the
recovery quantification this construction defers to is the Detector Evaluation stage.
