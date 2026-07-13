# Detector Calibration Workflow — design and justification

*The Detector workflow calibrates the diffraction-limited-imaging model — camera, point-spread function, emitter brightness, and brightness flicker — by inferring it with the physics fixed to pure diffusion, so that the reaction-diffusion inference downstream rests on a justified, reproducible imaging model. "DLI" (diffraction-limited imaging) and "Detector" denote the same model throughout.*

**Scope.** This document records the Detector-workflow design, the parameter ranges and their justification, and the implementation plan. It is self-contained: it justifies choices on physics, the in-repository numerical study, publicly available data and methods, and the current codebase (package `srm_and_sbi_dimer_alp`). American spelling; ranges in log10 space where noted.

**Why this workflow exists.** The imaging parameters (camera gain, offset, read-noise, photon conversion; PSF widths; emitter brightness distribution; the brightness-flicker generator) are currently fixed constants in the reaction-diffusion production model. For peer review these choices must be *justified and reproducible* rather than set by visual matching. The Detector workflow calibrates them by simulation-based inference against real videos, producing a provenanced parameter artifact with quantified justification, and exposes an explicit route from that artifact into the production runs.

---

## 1. Motivation and problem

The production model infers biology (diffusion coefficients, reaction rates, populations) from single-particle-tracking microscopy videos via an amortized neural posterior estimator with a normalizing-flow density. Every synthetic training video is rendered through an imaging model whose parameters are held fixed. If those fixed values are misspecified relative to the real acquisitions, real videos fall off the synthetic manifold the estimator was trained on, and the posterior extrapolates — the mechanism behind the observed out-of-prior estimates on real data.

Two deliverables follow:

1. **Justification.** Each imaging parameter must have a defensible value with a stated basis, not a hand-tuned constant.
2. **Reproducibility.** The route from real videos → calibrated parameters → production priors must be auditable and re-runnable.

The Detector workflow provides both by treating the imaging model itself as the inference target.

---

## 2. Role in the pipeline — how it feeds production

The Detector workflow is a **complete, first-class calibration workflow**, permanent and fundamental to the methodology. It runs the *same four-step process* as the canonical pipeline (simulate → infer → evaluate → experiment), differing only in its inference target (below), and it calibrates the diffraction-limited-imaging model the production pipeline depends on, so the imaging parameters are justified and reproducible rather than hand-tuned and the real-versus-synthetic domain gap is measured rather than assumed away. Its entry scripts live under `Script_Bank/Prime` alongside the canonical ones, and it has its **own committed HPC submission machinery** — the `SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_*` stage wrappers (Simulation, Inference, Evaluation so far; the Experiment and gap/coverage wrappers follow as those stages are built) plus a `_DETECTOR_HPC_Submit` dispatcher, mirroring the canonical stage-wrappers-plus-`Submit.sh` pattern (generic, `hpc_local.env`-driven, dry-run default), filename-namespaced alongside the canonical wrappers in `Script_Bank/HPC/` — the same filename-alias scheme (Option F) as the Prime entry scripts, not a subfolder. The one firm boundary: the Detector is a **separate, parallel workflow that is never wired into the canonical `Submit.sh` dispatcher or its four canonical stage wrappers**, which stay canonical-only. It reuses the canonical low-level building blocks by import and coexists with the canonical pipeline, but never plugs into it.

Its output is a **versioned, provenanced DLI-parameter artifact**: for each imaging parameter, the calibrated estimate with a credible interval, an in/out-of-prior flag, the source video conditions and public accessions, the estimator checksum, and the training configuration.

That artifact **feeds the production model's imaging parameters**. The production treatment of those parameters — holding them fixed at the calibrated values (the current behavior), inferring them jointly under priors *centered on the calibrated values*, or marginalizing them as a nuisance (§7) — is decided when the artifact is ported into production; the calibration itself is agnostic to that choice, and the guess → refinement → prior chain stays fully auditable under any of them.

**Workflow architecture.** The Detector and canonical workflows run the *same process* — simulate → infer → evaluate → experiment — differing only in the target: the canonical workflow infers the reaction-diffusion parameters with the imaging model held fixed, while the Detector infers the imaging parameters with the reaction-diffusion parameters marginalized as a nuisance. Because the learnable target is what every stage samples, places priors over, trains on, and recovers, the Detector needs its own adapted forward-model support and entry scripts; they reuse the shared low-level building blocks by import and change only the parameter sourcing. New Detector files therefore *coexist* with the canonical ones — a file is created anew only where genuinely different behavior forces it, and canonical code is modified only where reuse is impossible. The Detector code is built and validated standalone; the canonical modifications during construction are three small, behavior-preserving, default-canonical injections into shared machinery (§9.2 Guardrails), each defaulting to the canonical configuration so the existing stages are byte-identical. Integration into production is a later, deliberate step (§9, Phase D), delicate because it touches the shared parameterization; §5 identifies the exact hazard.

---

## 3. Method — two-stage factorized calibration

Inferring biology and imaging jointly is a high-dimensional problem (roughly twenty-one parameters) that a single amortized estimator calibrates poorly. The workflow factorizes it into a cascade in which each stage marginalizes the other's parameters as a nuisance drawn from a restricted distribution, approximating the intractable joint.

**Stage 1 — simulation-based inference over the imaging model.** The physics is frozen to pure diffusion (§4) so the video statistics constrain the detector, PSF, and flicker rather than the kinetics. An amortized neural posterior estimator whose parameter vector *is* the imaging model is trained on synthetic videos, then maximum-a-posteriori estimated on real videos, per condition. This yields a principled initial estimate for every imaging parameter, with a posterior (not only a point) so that uncertainty and identifiability are visible.

**Stage 2 — classical calibration and quantitative matching.** The Stage-1 estimate is refined with closed-form estimators grounded in detector physics:

- **Offset** from the mode of the pixel-intensity histogram.
- **Read-noise** from the variance of background pixels.
- **EM gain** from the slope of a photon-transfer curve, `Var[I] − σ²_background = G·μ`.
- **Photon conversion** from the same photon-transfer analysis.
- **Brightness** from a log-normal fit to the real photon histogram.
- **PSF width** from a direct fit to isolated emitters.

Stage 2 adds a **quantitative real-versus-synthetic distance** (§8) so that "the videos match" becomes a measured scalar with a significance test, not a visual impression. The stage-1 → stage-2 refinement rule is recorded explicitly as part of the provenance.

---

## 4. The pure-diffusion physics model

Stage 1 requires a diffusion-only variant of the simulator: the biology is frozen so the imaging parameters are identifiable. This is exposed as `build_system(pure_diffusion=…)` in `simulation_rds_support.py`, defaulting to the current reactive behavior (`False`) and adding a diffusion-only mode (`True`). This flag is one of the three behavior-preserving canonical injections the Detector needs (§9.2 Guardrails); the Detector's adapted RDS forward model (`detector_simulation_rds_support.py`, §9.2) calls it with `True` and draws the reaction-diffusion parameters from the nuisance, while the adapted DLI forward model (`detector_simulation_dli_support.py`) reuses the canonical imaging building blocks but sources the imaging parameters from the sampled θ — the canonical `simulate_dli` orchestrator is left untouched.

**What the diffusion-only mode does.** It registers all three species — A (monomer), B (mobile dimer), C (immobile dimer) — with their diffusion constants (`D_B = R_B·D_A`, `D_C = R_C·D_A`), seeds initial particles per species from the population counts, and **skips only the reaction registrations**.

**Why detector parameters transfer (verified against the code).** `build_simulation` seeds initial particles into every species directly from its own count parameter (`count_alp`, `count_bet`, `count_chi`) — the dimer species are *not* created by reactions. They are present from the first frame regardless of whether reactions are registered, and all three counts share the same prior, so B and C carry substantial, non-zero populations. Consequently:

- `extract_trajectory_poses` builds a **non-empty dimer mask** from the particles literally typed B and C;
- `compute_intensity` applies the dimer brightness multiplier `dimer_mule = √2` to those emitters;
- the monomer-plus-dimer photon distribution used to calibrate the detector is therefore preserved.

The camera, PSF, and photo-physics parameters are species-agnostic and applied per emitter by identical code in both modes, so a diffusion-only variant calibrates them faithfully.

**Congruence is qualified.** The diffusion-only variant is congruent with the reactive system in **species set, diffusion constants, initial particle placement, and the entire imaging pipeline** (PSF, photo-physics state machine, EMCCD noise, and the √2 dimer factor). It is **not** congruent in **population dynamics**: with no reactions firing, each species count and each particle's dimer identity stay fixed for the whole clip, so the variant reproduces only the *initial* (seeded) dimer fraction, not the reaction-determined interconversion steady state, and no particle changes its dimer state over time. This is sufficient for calibrating species-agnostic, per-emitter detector parameters, and must not be presented as reproducing the reactive system's population or temporal dynamics.

**Implementation prerequisite.** The diffusion-only mode must **retain the species-registration loop** (all three species with their diffusion constants) and skip only the reaction registrations. Dropping the B/C species would empty the dimer mask and remove the √2 dimer brightness contribution, breaking transfer.

**Cost.** The only bimolecular reaction is the A + A → B dimerization, which requires a within-capture-radius partner search; the dissociation and B ↔ C conversions are unimolecular. Omitting the reactions removes that pair search and leaves per-particle Brownian updates, so a speedup is expected. This is architecturally sound but not proven from the package source (the solver's internal cost breakdown is not exposed) — treat it as expected pending a direct timing measurement.

---

## 5. Parameterization mechanism — value-based roles

The Detector parameterization lives in its **own module** (`detector_parameterization.py`), decoupled from the canonical `parameterization.py`, because the detector-calibration system is similar to but distinct from the production system and its ranges differ by design.

**Canonical scheme (for context and porting).** In `parameterization.py` the role of each parameter is a strict two-way partition keyed **solely on `PRIOR_RANGE`**: a `(low, high)` tuple marks a learnable parameter (included in the inference prior and the θ vector, which `build_prior` turns into a `BoxUniform`); `None` marks a fixed parameter (a Known constant or a Hyperparameter), excluded from the prior, its `VALUE` read directly by consumers. `VALUE` is never consulted to decide the role, and it may be a scalar or a list (for example `brightness_quantile` is a list). `LOG_FLAG` and `LOG_BASE` are documentation fields only — `build_prior` returns the raw log10 bounds and the base-10 exponentiation (`10**theta`) is applied downstream by convention. For every learnable row the stored center satisfies the invariant `VALUE = 10**((low+high)/2)`: bounds live in log10 space, `VALUE` is the linear-space center.

**The value-based scheme (Detector).** To let one specification express learnable, fixed, nuisance, and (future) posterior-drawn parameters, the role selector moves to the `VALUE` field via two reserved string sentinels. The complete role matrix over `VALUE × PRIOR_RANGE`:

| `VALUE` | `PRIOR_RANGE` | role |
|---|---|---|
| concrete value (scalar or list) | `(low, high)` | **Learnable** — inferred; `VALUE` is the prior center |
| concrete value | `None` | **Fixed** — constant read directly |
| `"NUISANCE"` | `(low, high)` | **Nuisance from spec** — drawn from an inline `BoxUniform` over the range |
| `"NUISANCE"` | `None` | **Nuisance from object** — drawn from a supplied distribution artifact carrying its own parameter-key manifest |
| `"POSTERIOR"` | `None` | **Posterior draw** — future multiround inference |
| `"POSTERIOR"` | `(low, high)` | **undefined** — rejected by a build-time assertion |

**Design constraints (verified against the current code — these are the port hazards).**

1. **Role dispatch must be sentinel-based**, testing `VALUE in {"NUISANCE", "POSTERIOR"}`, and must **never** test whether `VALUE` is numeric — a list-valued fixed parameter already exists in the specification and would be misclassified by a numeric test.
2. **The learnable-subset selector must be `VALUE-is-not-a-sentinel AND PRIOR_RANGE-is-not-None`.** The canonical filter selects learnable rows by `PRIOR_RANGE is not None` alone; under the value-based scheme a nuisance-from-spec row *also* carries a range, so reusing the canonical filter verbatim would pull nuisance parameters into the inference prior and the θ vector. This is the single most important incompatibility to handle at port time: nuisance rows carry a range that feeds only their own draw and must be excluded from the inference prior.
3. **Log semantics must be explicit.** The scheme honors `LOG_BASE` consistently for both the center invariant and the sample-to-physical mapping (or enforces log10 for all ranged rows), and it specifies, for the supplied-distribution nuisance and the posterior cases, whether draws arrive already in physical space or in log space requiring exponentiation.

---

## 6. Parameter ranges and their justification

Two groups: the **RDS nuisance** (biology marginalized during detector calibration) and the **learnable imaging parameters** (the calibration targets). All ranges are log10; absolute ranges are shown for reference. Every learnable range brackets the current production operating point (the value used as a fixed constant today), so the calibration can only refine, never contradict by construction, that operating point.

### 6.1 RDS nuisance (drawn per simulation from a restricted `BoxUniform`)

| parameter | log10 range | absolute | justification |
|---|---|---|---|
| `diffusivity_alp` (D_A) | [−1.25, −0.25] | 0.056–0.562 µm²/s | Brackets a representative measured monomer diffusion of ~0.10 µm²/s (log10 −1.0), which sits in the lower half; the range extends upward to admit faster monomer diffusion. Grounded in reported SPT diffusion measurements for the relevant receptor systems (public accessions S-BSST712, S-BIAD1369). |
| `relative_diffusivity_bet` (R_B) | [−0.625, −0.125] | 0.237–0.750 | Brackets a representative measured value of ~0.60 (log10 −0.222). The production prior for this parameter spans 0.178–0.562, whose upper edge lies below 0.60 and therefore *excludes* the measurement; the wider nuisance interval is chosen to include it — a coverage improvement to carry into the production port. |
| `relative_diffusivity_chi` (R_C) | [−2.0, −1.0] | 0.01–0.1 | Immobile/slow dimer. The upper bound (0.1) lies below the R_B lower bound (0.237), so the immobile species is strictly slower than the mobile dimer throughout the sampled box, with a small nonzero residual mobility rather than exactly zero. |
| `count_alp/bet/chi` | [1.0, 2.5] | 10–316 per species | Wider than the production count prior (56–178). With reactions disabled the populations are static, so co-localization is spatial rather than kinetic: more uniformly placed emitters raise the probability that PSFs overlap, reproducing the crowded appearance dynamic binding would otherwise produce. The wide range lets a single configuration span sparse-to-crowded appearances. This emulates the co-localization *appearance*, not dimer photophysics or kinetics — acceptable because the calibration target is the imaging model, not the reaction rates. |

### 6.2 Learnable imaging parameters (calibration targets)

Log-uniform priors, each bracketing the production operating point shown.

| parameter | log10 range | absolute | operating point | forward-model role |
|---|---|---|---|---|
| `kappa_c` | (0, 1) | 1–10 | 4.75 | photon→ADU conversion; scales offset and readout variance |
| `kappa_o` | (2, 3) | 100–1000 | 250 | camera offset/pedestal |
| `kappa_g` | (2, 3) | 100–1000 | 150 | EM gain |
| `kappa_v` | (1, 2) | 10–100 | 50 | read-noise standard deviation |
| `mu_r` | (0, 0.5) | 1–3.16 | 1.5 | median PSF spread (log-normal scale) |
| `sigma_r` | (−1, 0) | 0.1–1 | 0.15 | emitter-to-emitter PSF variability (log-normal shape) |
| `mu_pc` | (2, 3) | 100–1000 | 250 | median emitter brightness (log-normal scale) |
| `sigma_pc` | (−1, 0) | 0.1–1 | 0.5 | brightness spread (log-normal shape) |
| `prob_photo_bleach` | (−2, −0.5) | 0.01–0.316 | 0.1 | photobleaching probability over the reference frame count; sets the rate into the dark state |
| `lambda_rate` | **(−1, 1.5)** | **0.1–31.6** | 5 | brightness-switching rate scale (tightened — §6.3) |
| `gamma_penalty` | **(−2, −1)** | **0.01–0.1** | 0.025 | brightness-transition distance penalty (tightened — §6.3) |

### 6.3 The brightness-flicker generator study (why `lambda_rate` and `gamma_penalty` were tightened)

Emitter brightness evolves as a continuous-time Markov chain assembled by `compute_matrices` (`simulation_dli_support.py`). For two non-dark brightness levels *i* and *j* the generator entry is

`Q[i,j] = lambda_rate · exp(−gamma_penalty · |brightness_i − brightness_j|)`,

every non-dark state has an additional rate into the absorbing photobleached state 0 (`epsilon = −ln(1 − prob_1) / delta_frame` with `prob_1 = 1 − (1 − prob_photo_bleach)^(1/numb_photo_bleach)`), the diagonal is `Q[i,i] = −Σ_j Q[i,j]`, and the per-frame discrete-time transition matrix is `P = expm(Q · delta_frame)`. With the operating brightness parameters (`mu_pc=250`, `sigma_pc=0.5`, quantiles `[0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`, `prob_photo_bleach=0.1`, `numb_photo_bleach=100`, `delta_frame=0.020 s`) the eight brightness states take integer photon values `[0, 110, 132, 178, 250, 350, 474, 569]`.

`lambda_rate` is the overall transition-rate scale; `gamma_penalty` is the steepness of the fall-off with brightness distance. A sweep over the two parameters (reproducible in-repository via `compute_matrices`) establishes:

- **Operating point** (`lambda_rate=5`, `gamma_penalty=0.025`): per-frame probability of a non-dark brightness switch ≈ **4.1–4.3 %** (4.27 % averaged uniformly over the non-dark states, 4.07 % weighted by the initial brightness distribution), i.e. a mean dwell of ≈ **0.47–0.49 s** between switches at 50 frames per second.
- **Degeneracy ridge.** Iso-flicker contours run diagonally in log-log space: a fixed flicker level (~4 %) is held by raising `lambda_rate` together with `gamma_penalty` (from `lambda_rate ≈ 0.8` at `gamma_penalty = 0.005` up to `lambda_rate ≈ 110` at `gamma_penalty = 0.1`). Along such a ridge the two parameters are distinguished only by the far-jump probability (a step of two or more brightness levels), which falls monotonically from ~2.2 % to ~0.1 % as `gamma_penalty` increases.
- **Limiting regimes** (over `lambda_rate ∈ [10⁻³, 10³]`, `gamma_penalty ∈ [10⁻³, 1]`). The chain is effectively frozen (per-frame flicker below ~0.6 %, dwell approaching the whole clip) whenever `lambda_rate < ~0.1`, for any `gamma_penalty`. Freezing that is **independent of `lambda_rate`** requires `gamma_penalty > ~0.3`; near `gamma_penalty ≈ 0.1` the flicker can still rise to ~14 % at very large `lambda_rate`. Chain freezing is therefore a **joint property of both parameters, not a function of `gamma_penalty` alone**. At low `gamma_penalty` and `lambda_rate > ~100` the flicker saturates near 0.85.
- **Stationarity note.** The true stationary distribution of `P` is degenerate (all mass in the absorbing dark state — every emitter eventually bleaches), so the relevant weighting for flicker statistics is the initial brightness (quantile) distribution, not the stationary vector of `P`.

**Tightening.** The wide exploratory ranges (roughly six decades in `lambda_rate`, three in `gamma_penalty`) are dominated by the frozen and hyper-flicker regimes, which are unphysical for the real emitters. The tightened box `lambda_rate ∈ [0.1, 31.6]` (log10 (−1, 1.5)) and `gamma_penalty ∈ [0.01, 0.1]` (log10 (−2, −1)) contains the operating point and spans a plausible few-percent flicker band while trimming the frozen (low-`lambda_rate`) and hyper-flicker corners; the exact bounds should be anchored to the observed on-time (dwell) of the specific fluorophore, which the dwell-time map provides directly.

**Residual degeneracy.** Tightening the box reduces wasted volume but does not remove the diagonal ridge — the two parameters remain partially non-identifiable within any box. The proper fix is a reparameterization into an effective flicker-rate and a locality/steepness parameter that are closer to orthogonal; this is a model-structure change flagged for a later iteration, not this one.

---

## 7. Nuisance and artifact design

**Nuisance artifact.** A samplable object exposing the same `.sample(n)` interface as a prior or posterior, wrapping either a `BoxUniform` (from-spec), a conditioned posterior, or a stored sample-set — and carrying its own **`parameter_keys` manifest** so a draw is self-labeling. It is named by the parameter domain it covers (`Nuisance_RDS` for the biology marginalized during detector calibration; `Nuisance_DLI` symmetrically for the imaging parameters marginalized during the production run), and it is **pooled across conditions**. Two knobs:

1. **Out-of-bound handling.** Draws are clipped to the prior box (`clip_to_prior=True`), and clipping is **never silent**: each draw carries a `clipped` flag recorded alongside that simulation's θ, a log line fires when it occurs, and a running count is kept. It is always possible to tell which training videos used a clamped value.
2. **Distribution storage.** The numeric parameters of the distribution (for a Gaussian, its mean and covariance) are stored in the artifact so it is reconstructible and inspectable.

**Constructing the `Nuisance_DLI` (the analysis step).** `Nuisance_RDS` and `Nuisance_DLI` share this artifact and its `.sample(n)` interface but differ in their source of truth. `Nuisance_RDS` is self-contained — declared a priori in the value-based table (§5; ranges §6.1) and drawn as a `BoxUniform`. `Nuisance_DLI` cannot be declared a priori — its content *is* the calibration result — so a person constructs it from the Detector's imaging posterior through a dedicated **analysis step** that precedes any production use.

The analysis mirrors the `Nuisance_RDS` definition in the **same value-based vocabulary** (§5), one row per imaging parameter, so the three construction forms are exactly the value-based roles, and the spec names them in that vocabulary: **fixed** (spec `role = "fixed"`; the §5 fixed role, `PRIOR_RANGE` = `None`; the parameter is held at a chosen `value`), **from-spec `BoxUniform`** (spec `role = "uniform"`; the §5 nuisance-from-spec role, `VALUE` = `"NUISANCE"` with a `PRIOR_RANGE`; the direct analogue of the `Nuisance_RDS` construction), or **from-object** (the §5 nuisance-from-object role, `VALUE` = `"NUISANCE"` / `"POSTERIOR"`, no range; draws come from the trained imaging posterior itself — spec `form = "posterior"` — or from a stored sample vector — spec `form = "samples"`). The **fixed** and **from-spec** forms are chosen per parameter under the block form `form = "perparam"`; the **from-object** forms set the whole-block `form` (`"posterior"` or `"samples"`) instead, because a conditioned posterior (or a joint sample vector) cannot be split parameter-by-parameter. The one irreducible difference from `Nuisance_RDS` is the **medium**: because its source of truth is the calibration plus a human judgment — both of which vary per experiment — the `Nuisance_DLI` spec is a **user-authored, analysis-emitted artifact**, not a hardcoded table.

Three properties govern the step:

1. **Prerequisite, made obvious and enforced by a validating gate.** The analysis step — not any automated stage — *emits* the `Nuisance_DLI` spec, so a spec cannot exist without having run it. Downstream consumers (the production import; the matched-synthetic generation of §8) call a single gate that is *validating*, not a mere existence check: it fails loud, and points back to the analysis, when the finalized spec is **absent**, **malformed** (wrong structure, unknown role or block form, missing fields), or carries a **value outside the imaging prior box** — because the calibration operates within that box (§6.2), a user-authored fixed value or uniform range that leaves it is rejected rather than used. The workflow documentation signposts the analysis as the required first step before any `Nuisance_DLI` is built.
2. **The analysis suggests; the user decides.** The step presents the imaging posterior — per-parameter marginals, credible intervals, and the maximum-a-posteriori estimate, pooled over the real recordings — and pre-fills the spec template with posterior-derived *suggestions* (credible-interval bounds for a from-spec range, the maximum-a-posteriori vector for a fixed value), each clearly marked as a suggestion. The user edits the template to set each parameter's role and its ultimate values.
3. **The build reads the finalized spec.** The analysis step's build mode turns the user-finalized value-based spec into the pooled `Nuisance_DLI` through the builder (`detector_nuisance_dli.py`) — the posterior form is materialized to a stored sample-set at build, so the persisted artifact is reconstructible — and flushes the samplable artifact (`.npz`) with its manifest; the construction provenance is recorded in the emitted spec itself. (The per-simulation `Nuisance_DLI_Theta_Set` record of draws is a *generation-time* artifact, written later when the imaging block is marginalized during production — §9, Phase D — not by this analysis build.) Any downstream stage that consumes the `Nuisance_DLI` must obtain it through the module's validating gate (`require_nuisance_dli`), which re-validates the spec (property 1) before building, so a nonconforming or absent spec stops the consumer rather than reaching production.

The step is a post-hoc analysis, not a canonical stage — it lives with the analyses that run on completed outputs — and it is where the two-stage calibration's human refinement (§3) becomes explicit: the SBI posterior is the initial guess, and the analyst decides how it feeds production.

**Persisted nuisance set (naming).** The per-simulation nuisance draws are recorded on disk as a `Theta_Set` variant — one file per task, in the `Theta/` subdirectory beside the learnable `Theta_Set` (no separate directory), derived by swapping the `Theta_Set` token of the canonical theta-set path (no new path code). The token embeds the marginalized domain, so the persisted file reads unambiguously as a theta set of the nuisance block: `Nuisance_RDS_Theta_Set` for the Detector (reaction-diffusion biology marginalized), and `Nuisance_DLI_Theta_Set` for the production workflow when the imaging parameters are marginalized (§9, Phase D). The general pattern is `{project_alias}_{timing_label}_Nuisance_<DOMAIN>_Theta_Set_TASK_{n}_{split}.{ext}`. This persisted set is distinct from the samplable `Nuisance_RDS` / `Nuisance_DLI` object above: the object is the sampler, and the `Nuisance_<DOMAIN>_Theta_Set` file is the self-labeling record of what it drew.

**Self-describing estimator artifact.** The current serialization path (`save_posterior` / `load_posterior` in `inference_support.py`) is a plain pickle of the live posterior object: it bundles no metadata, and the pickled `BoxUniform` prior stores only its low/high bound tensors positionally, with no parameter-name keys — so the artifact is not self-describing. Worse, the Inference stage wraps the embedding network with `torch.compile` before training; plain-pickling the resulting posterior embeds `torch._dynamo` internals captured during compilation, and because those private class layouts change between torch releases, the artifact is **torch-version-locked** (unpickling under a different torch version can fail to resolve the compiled internals).

The Detector workflow persists estimators as **three separable components**:

- **(a) a compile-stripped `state_dict`** — tensor weights only, with any `_orig_mod.` compile prefix removed;
- **(b) a rebuild spec** — the architecture hyperparameters and normalizing-flow configuration sufficient to reconstruct the estimator under whatever torch version is loading it;
- **(c) a metadata block** — the ordered parameter-key manifest mapped to prior bounds, plus provenance (torch version, timing label, test loss, source conditions and accessions, checksum).

Loading reconstructs the module from the rebuild spec under the current torch, applies `load_state_dict`, and reattaches a freshly built device-aware prior. This round-trips without deserializing any torch-internal code, so the artifact is both self-describing and version-portable. It is consistent with how the codebase already handles weights during training (raw `state_dict` saved and reloaded with `weights_only=True`). The canonical `save_posterior` path is left untouched.

---

## 8. Quantitative real-versus-synthetic distance

A sim-versus-real domain gap in embedding space is expected whenever the imaging model is misspecified, and it must be *measured*, not assumed away. An initial qualitative projection of embeddings motivated this but cannot quantify it: low-dimensional non-metric projections are seed-dependent, hyperparameter-sensitive, and distort inter-cluster distances, so "this condition is far" is a topological impression, not a measured distance.

**The measurement.** Compute a two-sample distance on the raw embeddings — **Maximum Mean Discrepancy (MMD)** (Gretton et al. 2012) and/or a **Classifier Two-Sample Test (C2ST)** (Lopez-Paz & Oquab 2017) — per condition, with **cell-level** significance (video chunks within a cell are correlated, not independent). This gives a seed-independent scalar gap and a hypothesis test. The measurement is implemented in `detector_embedding_space_distance.py` — embedding through the trained `Complex3DCNN`; an RBF-MMD on the raw embeddings with a median-heuristic bandwidth and a cell-block permutation test; and a classifier two-sample test on standardized features with cell-grouped cross-validation (`GroupKFold`), whose significance is read from the across-fold accuracies rather than from pooled per-chunk predictions.

**Discriminating an imaging gap from biology.** Run the distance against **imaging-parameter-matched** synthetic draws — rendered by drawing the imaging from the pooled `Nuisance_DLI` (the same calibrated imaging that feeds production, §7), the gap read out per condition (the real recordings sliced by condition) against those pooled-imaging synthetics. If matching the imaging collapses a condition's gap toward the baseline, the imaging-gap hypothesis is supported; a residual gap under matched imaging is the candidate genuine-biology signal. The gap has at least three candidate sources that must not be conflated: an **imaging/nuisance gap** (the simulator renders the wrong detector model), a **physics/signal difference** (genuine biology the embedding correctly encodes — signal to preserve, not remove), and a **prior-coverage gap** (the training prior does not span the real regime — fixed by widening the prior, not by embedding tricks).

**Layered plan for closing the gap.**

- **Layer 0 — quantify** (measurement; belongs to this iteration): the MMD / C2ST gap per condition, used as the objective to reduce and as a runtime out-of-distribution gate.
- **Layer 1 — close the imaging gap at the source** (this iteration; likely the largest, fully interpretable win): render training simulations with the calibrated imaging (pooled, via the `Nuisance_DLI`), or include the imaging variation in the training prior so the simulator spans the real acquisitions. This may close much of the gap with no embedding modification and is straightforward to defend in review.
- **Layer 2 — residual embedding alignment** (a later iteration; an architecture/training-objective change): if a gap survives Layer 1, add a *stable* alignment term to the estimator training between a synthetic and an (unlabeled) real mini-batch:
  - **CORAL** (Sun & Saenko, ECCV 2016): `λ·‖Cov(emb_sim) − Cov(emb_real)‖²_F` — cheap, stable, differentiable; recommended first.
  - **MMD** (kernel; Long et al., DAN, ICML 2015): `λ·MMD²(emb_sim, emb_real)`.
  - **Domain-adversarial training** (Ganin et al., JMLR 2016): most powerful, least stable on few real videos, highest risk of erasing signal; use only if the above are insufficient and scope it to nuisance-invariance.

  The embedding must remain a near-sufficient statistic for the biology while being invariant to the *imaging nuisance only*: marginal feature alignment can *increase* target error under label shift (Zhao et al., ICML 2019), so forcing all embeddings to overlap could erase genuine biological signal and yield in-prior-but-wrong parameters.

- **Guardrails (mandatory).** Alignment must not degrade in-distribution recovery on held-out synthetic data (the reference floor); run coverage diagnostics after alignment; keep the alignment weight small; report the gap before and after.

---

## 9. Peer-review gaps and the execution plan

### 9.1 Gaps this workflow closes, and those it defers

**Closed:** justification and reproducibility of every imaging parameter; a posterior summary (estimate plus credible interval) rather than a bare point; a prior-bounded (or explicitly logged out-of-bound) maximum-a-posteriori estimate; simulation-based calibration and coverage diagnostics (SBC, Talts et al. 2018; expected coverage and the trust crisis, Hermans et al. 2022; TARP, Lemos et al. 2023; local C2ST, Linhart et al. 2023); a quantitative real-versus-synthetic distance; disclosed training configuration; and a first identifiability analysis (the `lambda_rate` × `gamma_penalty` study, §6.3).

**Deferred (flagged for a later iteration):** the brightness-flicker degeneracy and other imaging-parameter degeneracies (gain × conversion, offset × baseline, brightness scale × PSF width) call for a reparameterization rather than a wider box; and the forward model has no stochastic EM-gain (gamma/Erlang) register — gain is a scalar multiply, with no excess-noise factor — a fidelity upgrade for a later iteration.

The estimator-evaluation methods that operationalize the calibration/coverage and quantitative-distance gaps above — the per-example loss distribution, the paired cross-run comparison, the central-limit-assumption checks, and the diagnostics cited here — are detailed with full references in the Validation and Diagnostics section of `PROJECT_CONTEXT.md`.

### 9.2 Implementation plan

**Guardrails.** Phases A–C add new code only, save three small, generic, behavior-preserving, default-canonical injections into shared machinery: `build_system(pure_diffusion=…)` (§4); an optional `paths=` on `VideoDataset` / `build_datasets` / `setup_training` (so a Detector training stage loads its own filename-namespaced data); and an optional `paths=` / `data_bank_root=` on `console_log_context` (so a Detector debug transcript carries the `_DETECTOR` tag). Each defaults to the canonical configuration, so the existing stages are byte-identical. The canonical stage scripts, the canonical forward-model orchestrator `simulate_dli`, and the behavior of `parameterization.py` are otherwise not modified — the adapted Detector forward-model support and entry scripts reuse the shared building blocks by import and coexist with their canonical counterparts. The port-back (Phase D) is the only other deliberate change to the existing pipeline, done after the Detector side is validated. Everything here is code and documentation; no simulation or training compute runs — on rcl01 or the HPC machines — without explicit approval (rcl01 is used to prove the workflow; production runs on the HPC machines).

**Phase A — schema, parameter machinery, and adapted forward models (new modules).**
- **A1 `detector_parameterization.py`** — the value-based parameter table (imaging parameters Learnable §6.2; diffusion coefficients and counts Nuisance-from-spec §6.1; `capture_radius`, `brightness_quantile`, `delta_frame`, `numb_photo_bleach`, and `dimer_mule = √2` Fixed; reaction-rate parameters absent, never read under `pure_diffusion=True`), together with the **role resolver and θ constructor**: the value-based dispatch (§5), the learnable-subset selector (`VALUE`-not-a-sentinel AND `PRIOR_RANGE`-not-`None`), nuisance draws from an inline `BoxUniform` or a supplied `Nuisance` artifact, and the physical-space mapping the forward models consume.
- **A2 `detector_simulation_rds_support.py`** — the Detector's adapted RDS forward model: a diffusion-only three-species simulation that reuses the canonical `build_system(pure_diffusion=True)` and `build_simulation` building blocks by import, drawing the diffusion coefficients and counts from the RDS nuisance rather than the canonical learnable θ.
- **A3 `detector_simulation_dli_support.py`** — the Detector's adapted DLI forward model: renders videos by reusing the canonical imaging building blocks (`compute_matrices`, `EMCCD`, `Gaussian`, `sample_psf_width`, `generate_state_trajectories`, `compute_intensity`, `add_noise`, `generate_frames`) but sourcing the imaging parameters from the sampled θ instead of the fixed table. The canonical `simulate_dli` orchestrator is left untouched.
- **A4 `nuisance.py`** — the `Nuisance` artifact (§7): a samplable object with a `parameter_keys` manifest and the two knobs, with builders from spec and from a conditioned posterior.
- **A5 `artifacts.py`** — the self-describing estimator format (§7): compile-stripped `state_dict` + rebuild spec + metadata, with a loader that rebuilds eagerly. The canonical `save_posterior` is untouched.

**Phase B — Detector entry scripts (new; C28-named `SRM_AND_SBI_DIMER_ALP_DETECTOR_*`; each drives the adapted forward-model support of Phase A).**
- **B1 Detector RDS simulation** (`..._DETECTOR_Simulation_RDS.py`) — drives `detector_simulation_rds_support.py`: diffusion-only three-species simulation, drawing diffusion and counts from the RDS nuisance.
- **B2 Detector DLI simulation** (`..._DETECTOR_Simulation_DLI.py`) — drives `detector_simulation_dli_support.py`: renders videos with the imaging θ drawn per simulation from the Detector prior.
- **B3 Detector inference** (`..._DETECTOR_Inference.py`) — train the imaging posterior on (video, imaging-θ) pairs, reusing the embedding and flow machinery; save via A5. Multi-GPU data-parallel like the canonical Inference (`torchrun`, one process per GPU under `DistributedDataParallel`; the single-GPU path is used when one GPU is allocated).
- **B4 Detector experiment** (`..._DETECTOR_Experiment.py`) — maximum-a-posteriori estimate on real videos per condition. The MMD / C2ST gap check (`detector_embedding_space_distance.py`, §8) is provided as a standalone measure and wired into this stage as the completing B4 step. The `Nuisance_DLI` is *not* exported here: it is constructed by the separate, user-driven analysis step (§7, "Constructing the `Nuisance_DLI`"), which the production import requires via a validating gate. Multi-GPU sharded like the canonical Experiment (per-worker shard, then a combine step).
- **B5 Detector evaluation** (`..._DETECTOR_Evaluation.py`) — coverage and held-out recovery for the imaging posterior. Multi-GPU sharded like the canonical Evaluation: the held-out tasks split round-robin across one worker per GPU under `torchrun`, each worker writes its partial recovery arrays as a shard, and a separate `--merge` step (single process, no GPU) concatenates the shards into one report; the single-worker path writes the report directly.

The Detector downstream stages (B3–B5) therefore match the parallelism of the canonical stages they mirror — data-parallel training and sharded recovery — rather than running single-process.

**Phase C — validation and gap closure.** Coverage (B5); per-condition gap quantification (B4); prior-bounded or logged-out-of-bound estimation; and one versioned, provenanced imaging-parameter file.

**Phase D — port back to the production workflow (later; the only edit to existing pipeline behavior; explicit go).** Extend `parameterization.py` with the value-based roles, **taking care that the learnable-subset selector becomes `VALUE`-not-a-sentinel AND `PRIOR_RANGE`-not-`None`** so nuisance rows are not pulled into the inference prior (§5, constraint 2); add the self-describing artifact format to the main Inference; and apply the chosen production treatment of the imaging parameters — held fixed at the calibrated values (the current behavior), inferred jointly under calibrated-centered priors, or drawn from the supplied `Nuisance_DLI` artifact and marginalized — the choice decided at this point rather than in advance.

**Ordering.** A1–A5 → B1–B5 → C → (validate) → D.

**Deferred, separately and cautiously:** applying the revised diffusion and count ranges (§6.1) to the canonical `parameterization.py` for a prior-revised production re-run — a candidate fix for the prior-coverage component of the real-data out-of-prior behavior, not done here.

**Forward-facing (beyond this iteration, recorded now so the future port is seamless).** The value-based role scheme (§5) is more than the Detector's parameterization: once the Detector workflow is proven, it is intended to be ported into the *canonical* codebase as a general capability — the flexible treatment of any parameter block as a **posterior, a nuisance, or a fixed vector**, the same three modes by which the Detector's imaging output feeds production (§2), generalized so a block's role can be switched without structural change. Today's fixed-imaging behavior then becomes one setting of that mechanism rather than a hardcoded default. The mechanism, the self-describing artifact format (§7), and the learnable-subset selector hazard (§5, constraint 2) are designed now precisely so this later port is seamless. The Detector workflow is the proving ground for that canonical improvement; the port itself is not this iteration's work.
