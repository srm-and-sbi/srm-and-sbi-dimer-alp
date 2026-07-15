# Nuisance_DLI construction — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py`. The script turns the
Detector's calibrated imaging posterior into the pooled `Nuisance_DLI` artifact through a
person-in-the-loop, two-mode workflow: it first shows the calibrated imaging posterior and
emits a value-based specification pre-filled with suggestions, then — after the person edits
that specification — validates it and builds the samplable artifact. This note explains how
to run each mode and how to read its outputs, so the construction can be used and understood
without reverse-engineering the code.

This is a post-hoc, user-driven analysis step, not one of the canonical pipeline stages. It
lives in `Script_Bank/Analysis`, is never wired into the stage dispatcher, and is purely
additive: it imports the core modules (`detector_nuisance_dli`, `detector_parameterization`)
and reads the completed Detector Experiment outputs, modifying nothing. It is the required
first step before any `Nuisance_DLI` is used downstream — see "Constructing the `Nuisance_DLI`
(the analysis step)" in `DETECTOR_WORKFLOW.md` (§7). Downstream consumers reach the artifact
only through the module's validating gate (`require_nuisance_dli`), which fails loudly and
points back to this analysis when the finalized specification is absent, malformed, or carries
a value outside the imaging prior box.

## What it does

The Detector Experiment stage estimates the imaging parameters (camera, point-spread,
brightness, flicker) of each real recording by maximum a posteriori (MAP), producing one MAP
vector per analysis window. This script consumes that pooled set of per-window MAP estimates
and helps a person decide how the calibration should feed production, expressed in the
value-based vocabulary shared with the Detector parameter table (`DETECTOR_WORKFLOW.md` §5):
each imaging parameter is assigned a role — **fixed**, **uniform**, **posterior**, or
**samples** — and the assembled block is built into a single pooled `Nuisance` object over the
imaging domain. The `Nuisance_DLI` cannot be declared a priori because its content *is* the
calibration result; the analysis therefore suggests numbers, and the person decides the
ultimate values.

The script runs in one of two modes.

**Emit-template mode (`--emit-template`, the default).** It reads the completed Detector
Experiment MAP output, summarizes the pooled per-window MAP estimates per parameter, prints
that summary against the imaging prior box, and emits a value-based specification template
(`.toml`) pre-filled with those numbers as suggestions. It also writes the pooled per-window
MAP vector to a separate `.npz` as a candidate source for the `samples` form. The person then
edits the template — setting each parameter's role and its final values.

**Build mode (`--build`).** It reads the person-edited specification, fully validates it (both
structure and the prior-box constraint, via `detector_nuisance_dli.load_spec`), builds the
pooled `Nuisance_DLI`, and persists the samplable artifact. The **posterior** form is not
buildable in this step: materializing it requires a trained Detector imaging estimator
conditioned on the real recordings, so a specification with `form = "posterior"` is accepted
as valid but raises a clear `NotImplementedError` at build, directing the person to the
`samples`, `fixed`, or `uniform` forms meanwhile.

## How to run it

Run on a machine that holds the Detector Experiment MAP database, with the project package
installed and `MACHINE_PROFILE` set so the machine profile resolves the data-bank paths.
Preview either mode first with `--dry-run`, which resolves the input and output paths and
reports what would be read or written without computing:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
      --total-time-seconds 2.0 --emit-template [--dry-run]

    (edit the emitted *_Nuisance_DLI_Spec.toml: set each parameter's role and final values)

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
      --total-time-seconds 2.0 --build [--dry-run]

Arguments:

- `--total-time-seconds` (required) — the recording duration that sets the `timing_label` (for
  example `2.0` → `2S_50FPS`). It locates the Detector-namespaced Experiment output and names
  all specification and artifact files; it does not itself trigger any simulation.
- `--emit-template` / `--build` — mutually exclusive mode selectors; `--emit-template` is the
  default when neither is given.
- `--pool-mode` — `bounded` (default) or `unrestricted`. It governs only the `posterior`
  draw form and mirrors the pooling knob used by the Detector Experiment and Evaluation
  stages (`bounded` = rejection sampling within the prior box; `unrestricted` = raw-flow draws
  that may fall outside the box and are then clipped). It has no effect on the `fixed`,
  `uniform`, or `samples` forms.
- `--dry-run` — resolve the paths and report what would be read and written; write nothing and
  compute nothing.

Input (both modes read from the completed Detector Experiment stage; here and in Outputs,
`<alias>` is the Detector-namespaced project alias `SRM_AND_SBI_DIMER_ALP_DETECTOR`):

- `<data_bank>/<posit>/<alias>_<timing_label>_MAP_Experiment/<same-name>.npz`,
  from which the `inferred_log10` array (the pooled per-window MAP estimates, in log10) is
  read. If this output is absent, empty, or carries a different number of imaging columns than
  the Detector parameterization declares, the script raises a precise error directing the
  person to run (or re-run) the Detector Experiment stage first.

## Outputs

All files are written to the Detector-namespaced Posit subdirectory under the data bank.

Emit-template mode writes:

- `<alias>_<timing_label>_Nuisance_DLI_Spec.toml` — the value-based specification template.
  It carries one `[imaging.<KEY>]` table per imaging parameter and a `[block]` table with the
  block `form`. Every value is in log10 (the sampling space), and every number is a
  suggestion; the file records its own provenance (the source Experiment `.npz`, the number of
  pooled windows, and the pooled-MAP samples path). It is authoritative only once the person
  saves their edits.
- `<alias>_<timing_label>_Nuisance_DLI_PosteriorSamples.npz` — the pooled per-window MAP
  vector (`samples` array), offered as a ready candidate for the `samples` form.

Build mode writes:

- `<alias>_<timing_label>_Nuisance_DLI.npz` — the built, samplable `Nuisance` artifact
  (`domain = "DLI"`), carrying its own `parameter_keys` manifest so each draw is
  self-labeling. Its underlying `kind` is `box` for the per-parameter form (a `BoxUniform`,
  with a `fixed` parameter contributing a degenerate single-point bin) or `samples` for the
  drawn forms (clipped to the imaging prior box with never-silent logging).

## How to read the emitted summary and choose a specification

Before writing the template, emit-template mode prints the calibrated imaging posterior — the
"see it" step — as one row per imaging parameter, all in log10:

- **MAP** — the median of the pooled per-window MAP estimates;
- **CI5** / **CI95** — the 5th and 95th percentiles of that same pooled set, i.e. the spread
  of the per-window point estimates across the real recordings;
- **prior box** — the Detector imaging prior interval `[lower, upper]` for that parameter, so
  the summary can be read against the range within which the calibration operates.

These numbers are decision support, not a finding. They summarize where the calibrated imaging
posterior places each parameter across the real recordings; they are seeds for a human choice,
copied into the template as the suggested `value` (from the MAP) and `[low, high]` (from the
5th/95th percentiles).

Choosing a role per parameter (`DETECTOR_WORKFLOW.md` §7):

- **fixed** (`role = "fixed"`, under block `form = "perparam"`) — hold the parameter at a
  chosen log10 `value`, which must lie inside the imaging prior box.
- **uniform** (`role = "uniform"`, under block `form = "perparam"`) — a `BoxUniform` over a
  chosen log10 `[low, high]`, which must lie within the prior box; this is the direct analogue
  of how the RDS nuisance is declared.
- **samples** (block `form = "samples"`) — draw the whole imaging block jointly from a stored
  sample vector at `[block].samples_path`; pointing it at the emitted
  `*_Nuisance_DLI_PosteriorSamples.npz` reuses the pooled per-window MAP vector as the sample
  source. Draws are clipped to the prior box.
- **posterior** (block `form = "posterior"`) — draw the block jointly from the trained imaging
  estimator conditioned on the real recordings. This form validates but is not built in this
  step (see the caveat below).

The `fixed` and `uniform` roles are set per parameter; the `samples` and `posterior` forms are
whole-block, because a joint sample vector (or a conditioned posterior) cannot be split
parameter-by-parameter. Build-mode validation rejects any `fixed` value outside the prior box
and any `uniform` interval that leaves it, with a message naming the offending parameter — the
calibration operates inside that box, so a `Nuisance_DLI` cannot assert imaging outside it.

## Caveats

- **The suggestions are calibrated estimates on real data, not ground-truth recovery.** The
  printed MAP and percentile spread come from applying the Detector's imaging estimator to real
  microscopy recordings, which have no ground truth. They describe the calibrated posterior's
  placement, not a demonstrated recovery of the true imaging parameters. Recovery of the
  imaging parameters is quantified only on held-out synthetic EVAL data with known values, in
  the Detector Evaluation stage — not by this analysis and not on the real recordings.
- **The percentile band is an inter-window spread, not a posterior credible interval.** CI5 and
  CI95 summarize how the per-window point estimates vary across the pooled real windows; they
  are not the width of any single posterior. Read them as the empirical dispersion of the
  calibration across recordings.
- **The posterior form is deferred.** Building `form = "posterior"` needs a trained Detector
  imaging estimator conditioned on the real recordings; until that exists the build stops with
  a clear message. Use the `samples` form (pointed at the emitted pooled-MAP vector) or the
  `fixed` / `uniform` forms in the interim. The `--pool-mode` argument therefore has no effect
  on a currently buildable specification.
- **The person decides; the analysis only suggests.** Nothing the template contains is
  authoritative until the person edits and saves it. The pre-filled numbers are a starting
  point, and the ultimate role and values are a human judgment recorded in the specification's
  provenance.
- **Prior-box clipping is silent-safe, not silent.** The drawn forms (`samples`, and later
  `posterior`) are clipped to the imaging prior box at build; the clip is logged rather than
  hidden, so any draw that would leave the box is reported.

## Reference

Real recordings: MET single-particle-tracking data, BioImage Archive accession S-BSST712. The
Detector calibration workflow, the value-based role scheme, the imaging prior ranges, and the
`Nuisance_DLI` construction step (§7, "Constructing the `Nuisance_DLI` (the analysis step)")
are documented in `DETECTOR_WORKFLOW.md`; the shared imaging model and its parameters are
described under DLI Imaging in `PROJECT_CONTEXT.md`.
