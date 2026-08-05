# EMCCD Noise Model — reference specification

*This document specifies the diffraction-limited-imaging detector noise model: the physically grounded electron-multiplying CCD (EMCCD) forward chain that maps expected incident photons to digital pixel values, together with the optical-background model that supplies those photon expectations. It is the reference against which the imaging stage's detector draw is defined and validated.*

**Scope.** This specification is self-contained: it justifies each stage on detector physics and publicly available sources, states its assumptions and their accuracy, and gives a reference implementation in the conventions of the package `srm_and_sbi_dimer_alp`. American spelling; photon and electron counts are per pixel per frame unless noted. Parameter values are quoted only to make the arithmetic concrete — they are representative acquisition settings, not authoritative constants. "DLI" (diffraction-limited imaging) and "detector" denote the same imaging model as elsewhere in the repository.

---

## 1. Purpose

The imaging stage renders each synthetic training video by transforming a noise-free photon-intensity image into camera counts. The fidelity of that transformation determines whether real acquisitions lie on the synthetic manifold the estimator is trained on: a misspecified detector model pushes real videos off that manifold and the posterior extrapolates.

This document specifies the transformation as a **Poisson–Gamma–Normal** EMCCD model whose every stage maps to a named physical process, so that the rendered statistics are defensible against the detector-physics literature rather than matched by eye, and so that each detector parameter carries an interpretable unit. It covers two coupled pieces — the **EMCCD forward chain** (§2–§4: photons → photoelectrons → electron multiplication → digitization, with read noise and bias) and the **optical-background model** (§5: the expected-photon field the chain consumes) — and closes with a reference implementation (§7), a validation protocol (§8), and the route into the calibrated imaging model (§9).

---

## 2. The observation chain

Let `λ_ph` be the expected incident photons per pixel per frame — the sum of emitter and optical-background contributions (the emitter signal — its brightness population and the dimer combination — is specified in `DETECTOR_WORKFLOW.md` §6.4–§6.5; the optical background in §5). It is an expectation (a non-negative real field), not a realized count. The detector maps `λ_ph` to a digital value `A` in analog-to-digital units (ADU) through the following ordered stages.

**Stage 1 — photoelectrons (Poisson).** Each incident photon produces a photoelectron with probability `QE` (quantum efficiency). By Poisson thinning, the photoelectron count is

```
N ~ Poisson(λ_e),    λ_e = QE·λ_ph + d·t_exp + c
```

where `d` is dark current (e⁻ px⁻¹ s⁻¹), `t_exp` the exposure time, and `c` an approximate clock-induced-charge (CIC) term (e⁻ px⁻¹ frame⁻¹). Sampling `Poisson(QE·λ_ph)` directly is exact — thinning a Poisson process by an independent keep-probability yields a Poisson process — so no separate binomial quantum-efficiency draw is required. Dark current and CIC are optional (§6), default to zero, and when present are added to the mean input before the Poisson draw. Adding them there applies the full register gain to the added charge; charge generated *within* the multiplication register instead experiences a position-dependent, lower effective gain, so this pre-register CIC term is an approximation adequate only while spurious charge is small — a more detailed model is required when it is not.

**Stage 2 — electron multiplication (Gamma register).** The register multiplies the `N` input electrons stochastically. Modeling each input electron's output as an exponential with mean `g` (the standard high-gain approximation for a many-stage register), the `N`-electron output is a sum of `N` independent exponentials — a Gamma variate:

```
Y | N ~ Gamma(shape = N, scale = g),    Y = 0 for N = 0
```

so `E[Y|N] = N·g` and `Var[Y|N] = N·g²`. The register is stochastic, not a deterministic multiply; the empty-input case produces exactly zero output. Spurious charge is represented as CIC in Stage 1, never by assigning a small nonzero Gamma shape to a zero-electron pixel.

**Stage 3 — digitization (conversion to ADU).** Output electrons become counts through the conversion factor `C` (electrons per ADU):

```
A_signal = Y / C
```

**Stage 4 — read noise (Gaussian, gain-independent).** The output amplifier and readout electronics add a zero-mean Gaussian, expressed in ADU with standard deviation `σ`, **after** multiplication and independent of `g`. The register amplifies signal but not this downstream noise — the origin of the EMCCD's low-light advantage:

```
A = A_signal + R + b,    R ~ Normal(0, σ²)
```

If read noise is characterized in output electrons (`σ_e`), the ADU standard deviation is `σ = σ_e / C`.

**Stage 5 — bias and optional digitization.** A constant electronic baseline `b` (ADU) is added last, keeping stored values non-negative. For bit-exact comparison with recorded movies the result may be rounded to the nearest ADU, clipped to the sensor range, and cast to the stored integer type; for training it is defensible to keep floating-point output, provided the difference from the recorded format is documented and shown not to affect inference.

**Unit contract.** The chain distinguishes three quantities that a single array named "intensity" would conflate — incident photons, photoelectrons, and ADU — and names every parameter with its unit:

| Quantity | Symbol | Units | Role |
|---|---|---|---|
| Expected photons | `λ_ph` | photons px⁻¹ frame⁻¹ | Poisson rate input (an expectation) |
| Quantum efficiency | `QE` | dimensionless | photon → photoelectron probability |
| EM gain | `g` | output e⁻ per input e⁻ | mean stochastic multiplication |
| Conversion factor | `C` | e⁻ / ADU | electrons → counts |
| Read noise | `σ` | ADU | Gaussian amplifier noise, after conversion |
| Bias | `b` | ADU | electronic baseline |
| Dark current | `d` | e⁻ px⁻¹ s⁻¹ | thermal electrons (optional) |
| CIC | `c` | e⁻ px⁻¹ frame⁻¹ | clock-induced charge (optional) |
| Exposure | `t_exp` | s | scales dark current to per-frame electrons |

---

## 3. Moments and the excess-noise factor

Set read noise and bias aside and ignore dark current and CIC (`λ_e = QE·λ_ph`). By the laws of total expectation and total variance,

```
E[Y]   = E[E[Y|N]]                      = g·λ_e
Var[Y] = E[Var[Y|N]] + Var[E[Y|N]]
       = E[N·g²] + Var[N·g]
       = g²·λ_e   +   g²·λ_e            = 2·g²·λ_e
```

The two equal terms are the amplified shot noise and the multiplication noise; their sum is the **excess-noise factor** `F² = 2`, defined relative to noiseless multiplication of a Poisson input (whose variance would be `g²·λ_e`). In ADU,

```
E[A]   = b + g·λ_e / C
Var[A] = 2·g²·λ_e / C²  +  σ²
```

**Accuracy of `F² = 2`.** This value is the many-stage limit of the Gamma model. The exact finite-stage register has `F² = 2 − 1/g` (for example `1.995` at `g = 200`): the Gamma model overstates the excess-noise factor by `1/g`, a `0.25 %` effect on variance at that gain, negligible here. The Gamma register is the standard, accurate model in this regime; its single approximation is stated so the fidelity level is explicit.

---

## 4. Constructions that misstate the noise

Three constructions reproduce the correct mean `g·λ_e` but give the wrong variance. Each is recorded as a form to avoid.

**Deterministic multiplication.** Replacing the register with `Y = g·N` omits its stochasticity, giving `Var[Y] = g²·λ_e` — excess-noise factor `1` — and understating the signal variance by a factor of two. It also discards the non-Gaussian register tail that carries the low-photon structure.

**Read noise before multiplication.** Adding the Gaussian read term to the input electrons and then multiplying scales it by the register: its contribution becomes `F²·g²·σ²` instead of `σ²`, so the read noise is amplified by the gain and the low-light advantage is lost. Read noise must be added after multiplication.

**Compounding a Poisson draw with a marginal approximation.** The single-Gamma form `Gamma(shape = λ_e/2, scale = 2g)` matches the first two moments of the compound Poisson–Gamma *marginal* (mean `g·λ_e`, variance `2·g²·λ_e`) and is a valid stand-alone approximation at sufficient intensity. Applying it to an *already-sampled* Poisson count — `Gamma(N/2, 2g)` after drawing `N ~ Poisson(λ_e)` — counts the shot-noise variance twice:

```
Var[Y] = E[2·N·g²] + Var[N·g] = 2·g²·λ_e + g²·λ_e = 3·g²·λ_e
```

an excess-noise factor of `3` and a stochastic standard deviation inflated by `√(3/2) ≈ 1.225`. The marginal Gamma and the conditional Gamma each give `F² = 2` in isolation; only their composition inflates it. After an explicit Poisson draw, the correct choice is the **conditional** register `Gamma(shape = N, scale = g)` of §2.

---

## 5. Optical background model

The chain consumes an expected-photon field `λ_ph`. **In the settled (corrected) model, the background component is a single scalar optical background offset `kappa_o` (inferred, one value per movie); the separable spatial×temporal lognormal derived in this section is deferred as a future refinement (§9).** The derivation below specifies that fuller field for when raw-background checks warrant it.

**Model.** A fixed spatial illumination map `B_ij` and a global per-frame multiplier `F_t`, both lognormal and independent, give

```
Λ_bg[i,j,t] = B_ij · F_t
B ~ LogNormal(median = m, log-σ = s_B)
F ~ LogNormal(median = 1, log-σ = s_F)
```

The temporal multiplier has unit median, so the product has median `m`. This rank-one model captures fixed field nonuniformity, global frame-to-frame illumination fluctuation, and positive skew; it does not capture local temporal changes, drifting gradients, row/column patterns, or spatially varying temporal noise. It is a defensible first-order model whose adequacy is tested against raw background movies (§8).

**Median-based lognormal algebra.** With SciPy's `lognorm(s, loc=0, scale=m)`, `log X ~ Normal(log m, s²)`, so the `scale` `m` is the **median**, not the mean; the mean is `m·e^{s²/2}` and the variance is `m²·e^{s²}·(e^{s²} − 1)`. Parameterizing the marginal by its median `m` and arithmetic standard deviation `d`, with `r = d/m` and `z = e^{s_tot²}`, the lognormal variance identity `r² = z(z − 1)` gives

```
s_tot² = ln( (1 + √(1 + 4·r²)) / 2 )
```

which differs from the mean-based `ln(1 + CV²)` precisely because `r` uses the median. For independent lognormal factors `log(BF) = log B + log F`, so log-variances add:

```
s_tot² = s_B² + s_F²        s_B² = s_tot² − s_F²
```

with the temporal factor set from a coefficient of variation, `s_F² = ln(1 + CV_F²)`. A requested `CV_F` that forces `s_B² < 0` is infeasible under the model and must **raise an error**, not be silently clipped; a value within floating-point tolerance of zero is set to zero.

**Median versus mean.** The Poisson rate (§2) is the photon *expectation*, so the field-average electron rate uses the background **mean** `m·e^{s_tot²/2}`, not the median `m` — for a representative `s_tot ≈ 0.35` the mean exceeds the median by about `6.3 %`. Specifying the background by its median therefore fixes an average level above `m`; a per-pixel moment check quoted at the median is local, and the field-average output mean uses `QE·m·e^{s_tot²/2}`.

**Localization statistics are not latent parameters.** Single-molecule-localization summary columns — a fitted local peak baseline and a local background standard deviation — are post-detection, per-localization quantities: they combine realized photon and detector noise, unresolved structure, and fitting-window effects, and are reported only around accepted detections. Their ratio is not a measurement of latent frame-to-frame illumination flicker and must not define `CV_F`. Latent temporal variation is estimated from raw background-only frames (§8); the localization columns are retained only as downstream posterior-predictive targets after full image formation.

---

## 6. Parameters and representative values

The chain's parameters and roles are the unit contract of §2. Representative acquisition settings for a cooled EMCCD in the single-molecule regime — quoted only to make the moment arithmetic concrete, **not** as authoritative constants — are `QE ≈ 0.95`, conversion `C ≈ 4.78 e⁻/ADU`, EM gain `g ≈ 200`, bias `b ≈ 79.7 ADU`, and exposure on the order of `10–20 ms`. (Pixel size is a geometric acquisition setting, not a camera‑noise parameter, so it is not listed here; it is per‑dataset metadata — `158 nm` for the MET dataset this model is matched to.) Here `g` and `C` are the MET spec values, whereas `QE ≈ 0.95` and bias `b ≈ 79.7 ADU` are round illustrative figures for the moment arithmetic below — the MET-matched fixed values are `QE = 0.9` and baseline `b ≈ 175 ADU` (the camera parameterization and prior table below). Nominal camera settings drift with sensor aging and depend on readout rate and preamplifier; values used for rendering should be tied to the acquisition settings of the data being matched and, where possible, determined by calibration (§8) rather than adopted as fixed truth. Dark current and CIC default to zero and are included only when dark-frame calibration shows they matter.

**Camera parameterization.** The five camera parameters — `gamma` (`= g/C`, the **ADU-per-photoelectron**, the only gain quantity the videos identify), `kappa_o` (the optical background offset, incident photons — §5), `kappa_s` (read noise `σ`), `kappa_b` (the camera baseline `b`, the electronic bias added last), and `kappa_q` (the quantum efficiency) — are non-identifiable from the videos and are **marginalized as the SCOPE camera nuisance** rather than inferred (`DETECTOR_WORKFLOW.md` §9.3): each is drawn per simulation from its a-priori box and rendered into every video. The EM gain `g` and conversion `C` are **not** resolved separately — only their ratio `γ` is identifiable (§9) — so they are held fixed at their nominal spec values (keys `kappa_g`, `kappa_c`), retained solely as metadata for the `γ`-vs-`g_spec/C_spec` drift check (§8). `QE` (key `kappa_q`) is applied once in the Poisson step (§2) — no longer held at 1 and absorbed into the emitter brightness scale — and only the product `γ·QE` is identifiable from the videos (§9), which is why the camera is marginalized rather than fit. The physics symbols `g, C, σ, b` are used in this document for the math; the `kappa_`/`gamma` keys are the code identifiers. The MET-matched values (`g = 200`, `C = 4.78`, baseline `b ≈ 175`, `QE = 0.9`, pixel `158 nm`, hence `γ = g/C ≈ 41.84`) are those recorded in the public dataset's per-cell ThunderSTORM `cameraSettings` protocol (accession `S-BSST712`) — the same source the emitter-brightness and PSF values are drawn from (`DETECTOR_WORKFLOW.md` §6.5).

As a concrete moment check, with these representative values and a per-pixel background of `m ≈ 30.81` photons treated as a fixed rate, `λ_e = QE·m ≈ 29.27 e⁻` gives an output mean `g·λ_e/C + b ≈ 1304 ADU` and a pre-read stochastic standard deviation `√(2·g²·λ_e)/C ≈ 320 ADU`. The compounded construction of §4 would instead give `≈ 392 ADU` — the `√(3/2)` inflation.

**A-priori boxes for the camera nuisance.** In the calibration workflow the five camera parameters `gamma`, `kappa_o`, `kappa_s`, `kappa_b`, and `kappa_q` are marginalized as the SCOPE nuisance, each drawn from a log-uniform a-priori box whose geometric center is `10^{mid}`; the EM gain `g` and conversion `C` are fixed to their nominal spec values (metadata only). The constants known from the acquisition protocol — `gamma`, `kappa_b`, `kappa_q` — and the optical background `kappa_o` are given tight anchor bands around their measured values rather than broad decade brackets: a broad camera box would let the amplified-background floor `γ·QE·kappa_o` dominate the video-to-video variation and collapse the detector embedding onto that single axis (DETECTOR_WORKFLOW.md §6.2). The read noise `σ`, weakly identifiable, is likewise pinned to a tight band at its datasheet value.

| Parameter | Symbol | Role | Prior (log₁₀) | Linear band | Center | Acquisition value (MET) | Identified by |
|---|---|---|---|---|---|---|---|
| `gamma` | `γ = g/C` | SCOPE nuisance | (1.62, 1.625) | 41.7 – 42.2 ADU/e⁻ | 41.9 | nominal `g/C = 200/4.78 ≈ 41.84` | data (tight anchor) |
| `kappa_o` | optical background offset | SCOPE nuisance | (1.455, 1.465) | 28.5 – 29.2 photons | 28.8 | offset median ≈ 28.7 (Fab 28.9 / InlB 28.6) | data (tight anchor) |
| `kappa_b` | camera baseline `b` | SCOPE nuisance | (2.24, 2.25) | 173.8 – 177.8 ADU | 175.8 | configured baseline ≈ 175 ADU | data (tight anchor) |
| `kappa_s` | read noise `σ` | SCOPE nuisance | (1.02, 1.025) | 10.5 – 10.6 ADU | 10.5 | read noise ≈ 50 e⁻ at 10 MHz ≈ 10.5 ADU at `C ≈ 4.78` | datasheet (weak) |
| `kappa_q` | `QE` | SCOPE nuisance | (−0.05, −0.04) | 0.89 – 0.91 | 0.90 | configured QE = 0.9 (per-dataset) | γ·QE only |
| `kappa_g` | `g` | fixed (metadata) | — | — | — | 200 (nominal, drift check) | — |
| `kappa_c` | `C` | fixed (metadata) | — | — | — | 4.78 e⁻/ADU (nominal, drift check) | — |

The gain and conversion enter the image likelihood only through the ratio `γ = g/C` (§9), so `g` and `C` are fixed nominal metadata and are not resolved by the videos; `gamma` and `kappa_q` are marginalized as the SCOPE nuisance, since only their product `γ·QE` is identifiable. The optical background `kappa_o` and camera baseline `b` are tight anchors around their measured values; the read noise `σ`, weakly identifiable, is likewise pinned tight to its datasheet value. Each MET acquisition value above lies at or near the center of its band. The read-noise datasheet figure is read from public camera specifications (BioImage Archive `S-BIAD1369`); the acquisition-matched values (`gamma`, `kappa_o`, `kappa_b`, `kappa_q`) are the MET values, and the bands are re-examined, and re-centered if required, against the calibration for any other acquisition (§8).

---

## 7. Reference implementation

The forward chain is a single detector-draw function, in the conventions of `srm_and_sbi_dimer_alp` (seedless by default; the RNG is derived from an optional seed). The `EMCCD` attributes are the unit contract of §2.

```python
import numpy as np
from typing import Optional


class EMCCD(Detector):
    """Electron-multiplying CCD detector: Poisson–Gamma–Normal forward model.

    Attributes:
        quantum_efficiency: photon -> photoelectron probability, in [0, 1].
        em_gain: mean electron-multiplication gain, output e- per input e- (> 0).
        electrons_per_adu: conversion factor C, output e- per ADU (> 0).
        read_noise_adu: Gaussian read-noise standard deviation, in ADU (>= 0).
        bias_adu: electronic baseline added after conversion, in ADU.
        dark_current_e_per_s: thermal dark current, e- per pixel per s (default 0).
        cic_e_per_frame: clock-induced charge, e- per pixel per frame (default 0).
        exposure_s: exposure time in seconds; scales dark current per frame (default 0).
    """

    def __init__(self, quantum_efficiency: float, em_gain: float,
                 electrons_per_adu: float, read_noise_adu: float, bias_adu: float,
                 dark_current_e_per_s: float = 0.0, cic_e_per_frame: float = 0.0,
                 exposure_s: float = 0.0):
        if not 0.0 <= quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must lie in [0, 1]")
        if em_gain <= 0:
            raise ValueError("em_gain must be positive")
        if electrons_per_adu <= 0:
            raise ValueError("electrons_per_adu must be positive")
        if read_noise_adu < 0:
            raise ValueError("read_noise_adu must be non-negative")
        if dark_current_e_per_s < 0 or cic_e_per_frame < 0 or exposure_s < 0:
            raise ValueError("dark current, CIC, and exposure must be non-negative")
        self.quantum_efficiency = quantum_efficiency
        self.em_gain = em_gain
        self.electrons_per_adu = electrons_per_adu
        self.read_noise_adu = read_noise_adu
        self.bias_adu = bias_adu
        self.dark_current_e_per_s = dark_current_e_per_s
        self.cic_e_per_frame = cic_e_per_frame
        self.exposure_s = exposure_s


def add_noise(intensity: np.ndarray,
              detector: EMCCD,
              seed: Optional[int] = None) -> np.ndarray:
    """Render EMCCD counts from expected incident photons.

    Applies the Poisson-Gamma-Normal chain (sec. 2): Poisson photoelectrons,
    stochastic Gamma electron multiplication, conversion to ADU, gain-independent
    Gaussian read noise, and bias.

    Args:
        intensity: Expected incident photons per pixel per frame; finite and
            non-negative. Emitter signal and optical background must already be summed.
        detector: An EMCCD instance (sec. 2 unit contract).
        seed: Optional RNG seed; None (default) draws non-deterministically.

    Returns:
        Array of the same shape as `intensity`, in ADU (floating point).
    """
    rng = np.random.default_rng(seed)
    expected_photons = np.asarray(intensity, dtype=np.float64)
    if not np.all(np.isfinite(expected_photons)):
        raise ValueError("intensity contains non-finite values")
    if np.any(expected_photons < 0):
        raise ValueError("intensity must be non-negative")

    # Stage 1 - photoelectrons (Poisson thinning; optional dark current + CIC).
    mean_input_electrons = (
        detector.quantum_efficiency * expected_photons
        + detector.dark_current_e_per_s * detector.exposure_s
        + detector.cic_e_per_frame
    )
    input_electrons = rng.poisson(mean_input_electrons)

    # Stage 2 - electron multiplication: Gamma(shape=N, scale=g); zero input -> zero output.
    output_electrons = np.zeros(expected_photons.shape, dtype=np.float64)
    positive = input_electrons > 0
    output_electrons[positive] = rng.gamma(
        shape=input_electrons[positive].astype(np.float64),
        scale=detector.em_gain,
    )

    # Stage 3 - conversion to ADU.
    frames = output_electrons / detector.electrons_per_adu
    # Stage 4 - gain-independent Gaussian read noise (ADU), after multiplication.
    frames += rng.standard_normal(frames.shape) * detector.read_noise_adu
    # Stage 5 - electronic bias.
    frames += detector.bias_adu
    return frames
```

Two adjacent hazards belong to the rendering interface rather than the detector draw: the expected-photon array must be **owned (copied), not aliased and mutated in place**, so emitter signal does not accumulate across frames or calls; and background supplied as a 2-D spatial map must be broadcast to the full frame count rather than shared by reference.

---

## 8. Validation

The model is validated at three levels: analytic moment tests, camera calibration, and end-to-end posterior-predictive checks.

**Moment unit tests.**

- **Zero input.** With zero photons, dark current, CIC, read noise, and bias, every output is exactly zero.
- **Detector moments.** For a large constant photon field with zero read noise and bias, `E[A] = g·QE·λ_ph / C` and `Var[A] = 2·g²·QE·λ_ph / C²` (the `√2` excess-noise factor).
- **Read-noise moments.** With zero photons and nonzero read noise, `E[A] = b` and `Var[A] = σ²`.
- **Reproducibility.** Equal seeds give identical output; different seeds do not.
- **Background moments.** The generated log-variances satisfy `s_B² + s_F² = s_tot²`, and Monte Carlo confirms the marginal median and standard deviation approach their targets.
- **Infeasible background.** A temporal `CV_F` exceeding the total variability raises an error rather than clipping.

**Camera calibration.** Acquire dark stacks (bias structure, read noise, spurious charge) and uniform-illumination stacks over several intensities (the effective system gain from the mean–variance photon-transfer relation `Var[A] − σ²_background = F²·(g/C)·μ_signal`, with excess-noise factor `F² = 2` and `μ_signal` the mean above bias; this slope gives `γ` (§6), which the videos identify but cannot decompose into `g` and `C`; the split is carried by the spec-informed priors, backed if needed by a gain-series or a gain-off conversion measurement. Cross-check the calibrated `γ` against the spec ratio `g_spec/C_spec` as a gain-drift diagnostic — §9) under the same gain, preamplifier, readout rate, region of interest, temperature, and exposure as the data being matched; recalibrate periodically because nominal gain drifts.

**Raw-background validation.** For experimental and simulated background-only movies, compare the ADU histogram, per-pixel temporal mean and variance, the mean–variance relation, temporal autocorrelation of the frame statistic, spatial structure, and the fraction of variance explained by a rank-one separable component. Retain the separable assumption only if one global temporal mode explains most coherent frame-to-frame variation.

**Localization-level and inference checks.** Run the same localization configuration on simulated and experimental movies and compare distributions — not only medians — of the localization columns; this is the stage at which those columns become useful targets. Because the detector model is part of the inference simulator, add: simulation-based calibration (rank/coverage on known simulated parameters); posterior-predictive checks that videos sampled from the inferred posterior reproduce raw-pixel and localization statistics; nuisance sensitivity to plausible calibrated camera and background values; an ablation isolating which biological parameters shift under a change in detector variance; and a domain-gap diagnostic comparing simulated and experimental control movies in the estimator's embedding.

---

## 9. Adoption path and scope

Relative to a rendering that multiplies gain deterministically and adds read noise before the register, this specification changes two things: it makes multiplication **stochastic** (restoring the `F² = 2` excess-noise factor) and moves read noise **after** the register as a gain-independent term. Both changes live in the single detector-draw function. Their consequence is that rendered pixel statistics change — signal variance and read-noise scaling both move — so the imaging parameters must be **re-calibrated** (§8, and the two-stage calibration in `DETECTOR_WORKFLOW.md` §3) rather than carried over; this is not an in-place substitution.

The camera is parameterized by the five nuisance quantities of §6 — `gamma`, `kappa_o`, `σ`, `b`, and `QE` — the read noise `σ` and camera baseline `b` independent of the gain. The gain and conversion enter the image likelihood only through `γ = g/C`: the register output scales as `Gamma(N, g) / C = Gamma(N, g/C)`, so two acquisitions with equal `γ` are statistically identical, and `γ` is the only camera-gain quantity the data can identify. Because even `γ` is only weakly identifiable — the videos pin the amplitude through the product `γ·QE`, not `γ` alone — the settled model **marginalizes the five camera parameters as the SCOPE nuisance** (drawing each from its a-priori box) and **fixes `g` and `C`** to their nominal spec values, retained only as metadata for the drift check. The individual `g`/`C` split — which the videos cannot resolve — is not carried as free parameters. This is the value-based-role scheme of `DETECTOR_WORKFLOW.md` §5, §9.2, and §9.3 applied to the camera: the five camera parameters marginalized, `g` and `C` fixed.

**`γ` as a model-quality diagnostic.** Since `γ` is exactly what the videos measure, compare the inferred `γ` against the fixed spec ratio `g_spec/C_spec = 200/4.78 ≈ 41.84`: agreement corroborates the sheet, and a departure is a genuine, reportable gain drift rather than an inference artifact. `g` and `C` themselves are fixed metadata and carry no posterior. Treat a `γ` posterior far from the spec ratio as real drift or a misspecified noise model; and require the `kappa_o`, `σ`, and `b` posteriors to agree with the background and dark-frame calibration.

Adopting this model is a parameterization change, sequenced with the calibration rework and gated by the validation protocol above; it is scoped to that rework, not to the current imaging stage.

The background model is an independent track. The settled model uses a **single scalar optical background offset `kappa_o`** (incident photons, one value per movie, inferred), which enters the expected-photon field before QE (§2, §5) and is amplified by `γ·QE` to dominate the ADU floor. The separable spatial×temporal lognormal field derived in §5 is **superseded** by this scalar for the corrected model and **deferred** as a future refinement — to be re-introduced only if raw-background checks show the scalar floor inadequate; its derivation is retained here as the reference for that refinement.

---

## Sources

- Hirsch, M., Wareham, R. J., Martin-Fernandez, M. L., Hobson, M. P., Rolfe, D. J. (2013). A Stochastic Model for Electron Multiplication Charge-Coupled Devices — From Theory to Practice. *PLOS ONE* 8(1): e53671. doi:10.1371/journal.pone.0053671.
- Krog, J., et al. (2024). Photophysical image analysis: Unsupervised probabilistic thresholding for images from electron-multiplying charge-coupled devices. *PLOS ONE* 19(3): e0300122. doi:10.1371/journal.pone.0300122.
- Robbins, M. S., Hadwen, B. J. (2003). The noise performance of electron multiplying charge-coupled devices. *IEEE Transactions on Electron Devices* 50(5): 1227–1232.
- Basden, A. G., Haniff, C. A., Mackay, C. D. (2003). Photon counting strategies with low-light-level CCDs. *Monthly Notices of the Royal Astronomical Society* 345(3): 985–991.
- Ryan, D. P., et al. (2021). A gain series method for accurate EMCCD calibration. *Scientific Reports* 11: 18348. doi:10.1038/s41598-021-97759-6.
- Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., Hagen, G. M. (2014). ThunderSTORM: a comprehensive ImageJ plug-in for PALM and STORM data analysis and super-resolution imaging. *Bioinformatics* 30(16): 2389–2390. doi:10.1093/bioinformatics/btu202.
- SciPy documentation, `scipy.stats.lognorm` (log-normal parameterization: shape `s` is the log-space standard deviation; `scale` is the median at `loc = 0`).
- Andor Technology. iXon Ultra & Life 897 hardware guide — camera read noise, EM gain, sensitivity/conversion factor, and base-level (bias) specifications. Available at www.andor.com.
- BioImage Archive, accession S-BIAD1369 — single-molecule imaging conditions for the matched acquisition (camera datasheet values used as representative settings).
