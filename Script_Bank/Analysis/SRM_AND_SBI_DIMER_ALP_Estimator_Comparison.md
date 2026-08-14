# Estimator Comparison

Companion to `SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py` (biology) and
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py` (detector). This is the
authoritative reference for both; the detector companion points here.

## What it does

Decides whether one trained estimator generalizes better than another. The obvious approach
— compare their mean test losses — is **invalid** when the two means come from different-sized
or differently-sampled sets: the per-video negative log-density is an intrinsic
posterior-entropy floor (a property of the *video*, not the estimator) plus the estimator's
KL error, so two means confound the sets' entropy content with the estimators' quality.

This diagnostic pairs instead. On the **shared `(task, sim)` subset** — the videos both
estimators were scored on — it takes the per-video log-score *difference*. Pairing cancels
the common entropy floor, so the difference isolates the difference of the two estimators'
KL divergences to the truth (Amisano & Giacomini 2007). It then tests whether that
difference is zero with three complementary tests and returns a verdict.

It is an **Analysis diagnostic**, not a pipeline stage: read-only, never wired into the
`Submit.sh` dispatcher, no GPU. It reads two finished Test-Loss-Distribution artifacts and
writes a report to the `Posit/` tier.

## One tool, both workflows

Both workflows produce a Test-Loss-Distribution artifact — the best epoch's per-video loss,
keyed by the stable `(task, sim)` identifier — so the comparison is workflow-agnostic and
built once over the shared-engine pattern: a workflow-agnostic **kernel**
(`estimator_comparison.py`, pure numpy + scipy), a shared **runner**
(`estimator_comparison_runner.py`), and two thin namespaced shims. The only per-workflow
difference is which alias-qualified `Posit/` the artifacts live under; the entry-point name
carries the namespace.

## The three tests

Run on the per-video improvement `improvement = loss_B − loss_A` over the shared subset
(**positive ⇒ A has the lower loss ⇒ A is better**):

- **Diebold-Mariano** (Diebold & Mariano 1995) — the mean-based test of equal predictive
  accuracy. Because these are per-video (exchangeable) forecasts rather than a time series,
  its long-run variance is just the i.i.d. variance, so the statistic is the paired z-test
  `mean / (std / √n)`. This drives the verdict.
- **Wilcoxon signed-rank** — median/rank-based, so it is robust to the heavy upper tail of
  the per-video loss (a few catastrophic videos cannot dominate it).
- **Paired bootstrap** — a percentile confidence interval of the mean improvement, resampling
  video pairs; if the interval excludes zero the mean gain is real.

The mean-based Diebold-Mariano and the rank/interval companions agreeing is the strong
signal; a split (e.g. a significant mean but a Wilcoxon that is not) points to a difference
carried by the tail rather than a broad shift.

## Inputs

Each estimator is named by `--a` / `--b`, which accept any of:

- **`canonical`** — the current best-epoch TLD (`<...>_Test_Loss_Distribution.npz`).
- **a loss tag** such as `-6.91` — globs the workflow's `Posit/` for the matching provenance
  backup (`<...>_Test_Loss_Distribution_..._TEST_LOSS_-6.91.npz`).
- **an explicit `.npz` path** — for a TLD anywhere on disk.

The two sets need not have identical coverage or ordering: the pairing is by `(task, sim)`
identity, so the shared subset is found automatically.

## How to run

**1 — Select the machine.** `export MACHINE_PROFILE=<profile>` (resolves every path).

**2 — Dry-run** to resolve + verify the two artifacts without computing:

```bash
python SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py \
    --total-time-seconds 2.0 --a canonical --b -6.91 --dry-run
```

**3 — Run** (biology, or the detector twin). It is a seconds-long local read — no GPU:

```bash
# biology: is the current best a real improvement over the -6.91 checkpoint?
python SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py \
    --total-time-seconds 2.0 --a canonical --b -6.91

# detector twin, same options
python SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py \
    --total-time-seconds 2.0 --a canonical --b -12.16
```

Options: `--n-boot` (bootstrap resamples, default 10000), `--seed` (bootstrap RNG),
`--alpha` (significance, default 0.05).

## Reading the report — what every metric means

`improvement = loss_B − loss_A`; **positive favors A**.

| Metric | What it is | Reading |
|---|---|---|
| **shared (task,sim) videos** | size of the paired sample (∩ of the two key sets) | the comparison uses exactly these; `A=…, B=…` show each set's own size |
| **mean improvement** | average per-video log-score gain of A over B | sign = which is better; magnitude = average loss gap |
| **median improvement** | robust center of the gain | ≈ mean ⇒ a *broad* shift; ≪ mean ⇒ the mean is tail-driven |
| **fraction where A wins** | share of videos with lower loss under A | ~0.5 ⇒ a wash; ≫0.5 ⇒ A better on most videos |
| **Diebold-Mariano** | `stat`, two-sided `p` (H0: equal accuracy) | small `p` ⇒ the mean gain is significant; `stat` sign matches the mean |
| **Wilcoxon signed-rank** | `stat`, two-sided `p` (H0: symmetric about 0) | the tail-robust confirmation of the sign |
| **bootstrap CI** | interval for the mean improvement | excludes 0 ⇒ significant; its width is the mean's precision |
| **verdict** | `A better` / `B better` / `no significant difference` | follows the Diebold-Mariano `p` at `alpha` |

The figure `figures/improvement_distribution.png` shows the full per-video improvement
distribution with the zero line, the mean, and the CI band — its *location* relative to zero
is the signal (the raw loss scale is cancelled by pairing).

## Outputs

Under `<data_bank>/Posit/<project_alias>_<timing_label>_Estimator_Comparison/`:

- `report.md` — the paired-comparison table + verdict.
- `figures/improvement_distribution.png` — the per-video improvement histogram.
- `<...>_Estimator_Comparison.npz` — the saved per-video improvement array (for re-analysis).

## Reuse scope

Read-only, never dispatched. Serves both workflows through the two namespaced shims over one
shared engine. Run it to decide whether a training change (a new checkpoint, a longer run, a
warm restart) is a *real* generalization gain rather than noise or a shift in the TEST set.

## References

- Diebold & Mariano (1995), *Comparing Predictive Accuracy*, JBES.
- Amisano & Giacomini (2007), *Comparing Density Forecasts via Weighted Likelihood Ratio
  Tests*, JBES.
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA
  (the log score as a proper scoring rule; its KL-divergence decomposition).
