# Experiment Population Composition

Companion to `SRM_AND_SBI_DIMER_ALP_Experiment_Population_Composition.py`. It reports the relative
abundance of the three modeled species across the experimental MET recordings — what fraction of the
population is monomer, mobile dimer, and immobile dimer — and reports beside it the error of that
same readout measured on held-out synthetic videos whose parameters are known.

The two halves belong in one document. An experimental estimate is only as readable as the measured
accuracy of the instrument that produced it, and here the instrument can be measured exactly: the
identical function of the identical estimator, applied where the answer is known.

## Why proportions and not counts

The estimator infers three absolute species counts — A (monomer), B (mobile dimer), C (immobile
dimer). Those are the worst-identified coordinates the model has, and the reason is
information-theoretic rather than a defect of training. Counting few emitters in a
diffraction-limited scene is a square-root-of-n problem, so the relative precision of a count
degrades as the count falls; and the three counts trade off against one another inside the posterior
because the reactions interconvert them, most strongly between the two dimer states, which absorb the
same signal.

On the held-out synthetic set that shows up directly:

| quantity | error |
|---|---|
| A (monomer) alone | 0.117 dex |
| B (mobile dimer) alone | 0.158 dex |
| C (immobile dimer) alone | 0.130 dex |
| **T = A + B + C** | **0.0117 dex** (r = 0.9995) |

The sum is recovered an order of magnitude better than any of its parts. That is only possible if the
parts' errors are strongly anti-correlated — the posterior moves abundance between species while
holding the total nearly fixed — and it means any quantity dividing one part by the total inherits
that cancellation. The composition is therefore a well-identified function of a badly-identified
vector, which is the whole reason this analysis exists.

## Why the fractions must be formed inside each draw

A ratio of correlated coordinates is not a function of their marginals. Building a fraction from
per-parameter medians — `median(B)` over `median(A) + median(B) + median(C)` — asserts a combination
the posterior never drew, and it discards exactly the cancellation that makes the ratio identifiable.

Every fraction here is therefore computed **within** a single posterior draw, and only then averaged.
This is the same principle the Sample Geometric Median analysis applies to a single summary vector,
applied here to a derived quantity: a per-coordinate composite is not a configuration the system was
ever inferred to be in.

The consequence is a hard input requirement. The analysis needs the Experiment stage's raw per-window
posterior draws (`--dump-posterior-samples`); the stored marginal quantiles cannot substitute,
because a fraction of marginals is a *different quantity*, not a coarser version of the same one. The
script fails loudly and says which re-run is needed rather than silently reporting the wrong thing.

## The quantities

With `T = A + B + C`:

| symbol | definition | reads as |
|---|---|---|
| `f_A` | `A / T` | share of complexes that are monomers |
| `f_B` | `B / T` | share that are mobile dimers |
| `f_C` | `C / T` | share that are immobile dimers |
| `f_D` | `(B + C) / T` | share that are dimers of either kind, `= 1 - f_A` |
| `f_R` | `2(B + C) / (A + 2B + 2C)` | share of individual **receptors** sitting in a dimer |
| `T` | `A + B + C` | total complexes in the scene (a count, not a share) |

`f_D` and `f_R` differ because a dimer holds two receptors: a population half dimeric by complex is
more than half dimeric by receptor. `f_A + f_B + f_C = 1` for every draw by construction, and the
report checks it (residual ~1e-16) as the cheapest possible detection of a coordinate mix-up.

Because `f_D = 1 - f_A`, their recovery errors are equal in magnitude by identity. The two rows
agreeing in the validation table is an internal consistency check, not two independent results.

## The aggregation ladder, and the replicate unit

Four levels, each an unweighted mean over the level below:

```
draws   -> window       posterior mean of the fraction for that window
windows -> recording    the recording's composition over its analyzed span
cells   -> condition    the reported value; the standard error is taken HERE
```

**The replicate unit is the recording, never the window.** Ten windows of one cell are ten views of
one biological object, not ten independent measurements; treating them as independent would divide
the standard error by the square root of the window count and manufacture significance out of
pseudo-replication. Every interval and every test in the report is computed over recordings, and the
report states the count it used.

## What it reports

**Experimental half**

- **Span-averaged composition** per condition, mean ± SEM over recordings, plus the total count.
- **First-window composition** — the earliest state the recordings resolve, reported separately
  because a span average over a changing population describes no instant of it.
- **Bootstrap check** — a percentile bootstrap resampling recordings, so the reported error bars can
  be seen not to depend on the normal-theory assumption behind the SEM.
- **Sensitivity** — the headline recomputed under two independent challenges: restricting posterior
  draws to the trained prior box (counts only, then all ten parameters), and replacing the arithmetic
  mean with the compositional (closed geometric) center, which is the natural center on the simplex.
  The table reports what each variant costs in discarded draws and in windows left with none. The
  spread across variants is the model-conditional bracket on the value.
- **Per-recording spread** of the monomer share — whether the condition mean describes a typical cell
  or falls between two groups of them.
- **Within-recording time course** — per-window means, the median per-recording Spearman correlation
  against window index, and a first-versus-last signed-rank test paired within recording.
- **Condition contrast** at recording level, two-sided rank-sum, per quantity.

**Synthetic half**

- **Recovery of the same quantities** on the held-out evaluation set: mean absolute error, its 95th
  percentile, bias, and correlation with truth. Fractions in percentage points, the total in dex.
- **Parts versus whole** — the per-count errors beside the total's, the comparison the composition
  argument rests on.
- **Recovery in the dimer-rich region** — the same statistics restricted to videos whose *true* `f_D`
  exceeds a threshold (default 0.7), the corner the activated condition occupies. Selection is on
  truth, never on the estimate, so it is independent of the estimator's own error.

Figures: `composition_conditions` (both conditions per quantity, with the synthetic error drawn on
each bar), `composition_recovery` (true versus inferred share on held-out videos),
`composition_time_course` (`f_D` across the recording per condition), `composition_per_cell` (every
recording's monomer share against the condition mean).

The derived arrays — per-window, per-recording, and per-condition — are written alongside as `.npz`,
so the tables can be re-tabulated or carried downstream without re-running.

## How to run

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Population_Composition.py \
    --total-time-seconds 2.0 [--bootstrap 10000] [--stratum-threshold 0.7] [--dry-run]
```

CPU only, seconds to run — it reads two completed outputs and neither loads the estimator nor renders
videos. It is a post-hoc analysis, never wired into the stage dispatcher.

It requires a biology Experiment output produced with `--dump-posterior-samples`. The MAP_Recovery
output is optional: without it the experimental half still runs, and the report states that the
validation column is unavailable rather than presenting experimental numbers with no measured error.

The tokens `ALP` and `BET` appear nowhere in the interface, the report, or the figures; they survive
only inside the Experiment output's stored `kinds` field and are translated at the boundary.

## Shared engine, and why this one is biology only

The analysis runs on the engine `srm_and_sbi_dimer_alp.population_composition_runner` over the
workflow-agnostic kernel `srm_and_sbi_dimer_alp.population_composition`. The kernel knows nothing
about receptors: it takes draws, the indices of the coordinates holding species counts, and a prior
box. It reuses the temporal-dynamics kernel's grid builder, so "a (condition, recording, window)
grid" means one thing across both analyses and cannot drift between them.

Unlike the mirrored stage runners there is **no detector counterpart**, and the asymmetry is
scientific rather than incidental: the composition is a function of the three inferred species
counts, and the detector workflow infers imaging parameters and no counts at all — it treats the
population implicitly, as part of what it marginalizes. There is nothing there to compose, in the
same way the detector's `Nuisance_DLI` pool has no biology counterpart. The spec resolver checks for
the count parameters and fails loudly for a workflow that lacks them, rather than composing six
imaging coordinates into a well-formatted meaningless number.

## What it found on the 2 s MET data

Run on the 500-window Experiment output (250 windows per condition, 1,000 draws each) with the
10,000-video held-out recovery:

| condition | monomers | mobile dimers | immobile dimers | all dimers | receptors in dimers |
|---|---|---|---|---|---|
| MET-FAB (resting) | 68.6 ± 2.0 % | 12.9 ± 1.7 % | 18.4 ± 0.7 % | 31.4 ± 2.0 % | 46.4 ± 2.2 % |
| MET-INLB (activated) | 21.8 ± 3.7 % | 15.6 ± 1.7 % | 62.6 ± 2.7 % | 78.2 ± 3.7 % | 86.2 ± 2.5 % |

Against a synthetic in-model error of **2.8 pp** on `f_D` (2.6 pp with +0.3 pp bias in the dimer-rich
region the activated condition occupies), the 46.8 pp difference between conditions is roughly
seventeen times the estimator's own error. At recording level the contrast is `f_A` ×0.31,
`f_C` ×3.42, `f_D` ×2.57 (all p ≈ 2e-9, 25 versus 25 recordings), and total abundance ×1.88
(p = 0.014). The mobile-dimer subdivision does not separate (p = 0.19) — and it is also the least
identified quantity of the set (5.0 pp), which is the honest reading of that null.

Three qualifications the report carries with those numbers:

- **The absolute value has a bracket.** Restricting to draws inside the full ten-parameter prior box
  keeps only 3.4 % of MET-INLB's draws (123 of its 250 windows keep none) and gives `f_D` = 73.1 %;
  the compositional center gives 86.4 %. The arithmetic mean, 78.2 %, is the conservative headline,
  and the honest model-conditional bracket is **73–86 %**. Under every variant the contrast against
  MET-FAB (29–36 %) survives — the *comparison* is far more robust than the absolute level.
- **The composition changes within the recording.** MET-INLB's dimer share falls from 87.0 % in the
  first window to 70.2 % in the last (median per-recording Spearman −0.71; first-versus-last
  p = 0.0015), while MET-FAB is flat over the same span (32.5 % to 31.7 %, p = 0.52). A condition that
  moves while its control does not cannot be explained by anything shared between the two
  acquisitions. The composition is therefore a trajectory, and the first window is the initial
  post-activation state rather than a noisier version of the average.
- **The condition mean is not a typical cell.** The MET-INLB per-recording monomer share spans 1.2 %
  to 52.8 %, with 9 of 25 recordings below 10 % and 6 above 40 %. Downstream use should carry the
  per-recording values, which are in the saved arrays.

## Reading the result against published measurements

Two published measurements of this receptor system are direction checks, and neither is
unit-compatible with a complex census, so magnitudes are not directly comparable:

- Harwardt et al., *FEBS Open Bio* **7**(9):1422–1440 (2017), doi:10.1002/2211-5463.12285 — the study
  whose recordings these are (BioStudies accession S-BSST712). It classifies *trajectories* by
  motion, reporting immobile/confined/free occurrences of 12 ± 1 % / 31 ± 3 % / 57 ± 3 % for
  Fab-bound MET and 23 ± 1 % / 17 ± 2 % / 60 ± 1 % for InlB321-bound MET (mean ± SEM over N = 60
  cells, at 23 °C). It reports **no** receptor density and **no** dimer fractions. Activation roughly
  doubling the immobile share is the same direction this analysis finds; the magnitudes belong to
  different estimands (a detection-weighted trajectory classification versus a model-based complex
  census) and should not be equated.
- Dietz et al., *BMC Biophysics* **6**:6 (2013), doi:10.1186/2046-1682-6-6 — single-molecule
  photobleaching in chemically fixed cells with sub-stoichiometric labeling, giving 18 % dimeric
  spots unstimulated rising to 29 % after InlB321. The publication itself states its dimer fraction
  "is clearly underestimated" for that reason, so these are **protocol lower bounds**: they confirm
  the direction and bound the value from below, and a lower bound of 29 % neither corroborates nor
  contradicts a live-cell estimate near 78 %.

## What this analysis does not establish

- **It is not a molecular census.** The composition describes the three-species model as fitted to
  these recordings. Where the model has no state for something slow, bright, and persistent, the
  immobile-dimer state absorbs it.
- **It does not validate the posterior's interval.** The synthetic half measures point-estimate
  error. Whether a credible interval on `f_D` covers truth at its nominal rate needs joint posterior
  draws on the held-out synthetic set, which a recovery artifact storing per-parameter marginal
  quantiles does not carry. The kernel's composition function takes any draws array, so that check
  becomes available as soon as such draws exist.
- **It does not place the recordings inside the trained domain.** That is the embedding-space distance
  analysis's question, and its answer conditions everything here.
- **It does not attribute the within-recording change** to a mechanism. It establishes that one
  condition changes and its control does not.
