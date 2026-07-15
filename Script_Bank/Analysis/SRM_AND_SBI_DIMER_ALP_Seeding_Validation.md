# Generation seeding and file-label validation — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_Seeding_Validation.py`. The script asserts that
a generation fan-out's output file labels are collision-free and that its seeding
behavior is non-deterministic by default yet reproducible under an explicit seed;
this note explains what it checks, how to run it, and how to read the verdict, so the
check can be used and understood without reverse-engineering the code.

This is a standalone, ad-hoc reproducibility check, run by hand before a generation
fan-out. It is not one of the canonical pipeline stages and is kept out of the stage
dispatcher.

## What it checks

Generation is non-deterministic by design: the entry points pass no seed by default,
so every simulation is a fresh independent draw and the sampled theta vector is
persisted per task rather than regenerated from a seed. Dataset integrity therefore
cannot rest on value reproducibility; it rests entirely on the output file **labels**
being collision-free across the fan-out. Each output path is built from the global
task index, the split namespace (TRAIN / TEST / EVAL), the per-task simulation index,
and the run's timing label. If any two of these mapped to the same path, one task
would silently overwrite another's output and corrupt the assembled dataset with no
error raised. The check exercises the real path builders (`PARAMETERS.paths.*`) over a
representative fan-out and asserts the labels are unique before a large run is
launched.

The fan-out it runs over is a small representative shape: TRAIN as two offset blocks
(tasks 0–7, then an appended 8–15), TEST tasks 0–1, and EVAL task 0, at four
simulations per task. This miniature reflects the production split structure (the
CORE = TRAIN + TEST sizing rule, with EVAL held out) plus one incremental grow — the
appended TRAIN block being the case most at risk of a collision.

The checks:

- **A + C — global uniqueness.** Every output path the whole fan-out would write (the
  theta set, the video set, and each per-simulation trajectory) is distinct. This
  covers both within-split uniqueness (a distinct task/simulation always maps to a
  distinct path) and the offset append.
- **C — offset append disjoint (explicit).** The original TRAIN block (tasks 0–7) and
  the appended block (tasks 8–15) share no path. This mirrors the documented "grow
  TRAIN later" workflow, in which a second generation job is launched with a task
  offset to append tasks without regenerating the first block (the `TASK_OFFSET` knob
  in the HPC simulation script). Because the append draws fresh, independent theta, it
  must land on new labels or it would overwrite the original block.
- **B — cross-split disjoint.** The TRAIN, TEST, and EVAL path sets never intersect;
  the split suffix namespaces them apart.
- **D — non-determinism sanity.** Reproducing the production theta sampler exactly (a
  log-uniform draw over the prior bounds, then exponentiated to physical units), the
  check confirms that: two default draws (no seed) differ, so the default is not
  silently forcing a fixed seed; the rows within one draw are distinct; and an explicit
  seed reproduces the draw bit-for-bit. When ReaDDy is available it additionally
  confirms the same behavior for initial particle placement — two unseeded placements
  of the same theta differ, and an explicitly seeded placement is reproducible. If
  ReaDDy is not installed the placement portion is skipped, not failed.

## How to run it

Run on any machine with the project package installed and a valid `MACHINE_PROFILE`
(the configuration is validated at import, so the profile's `data_bank_root` must
exist). The check builds path strings and samples small in-memory arrays only; it never
reads or writes the data bank and needs no GPU. The optional placement portion
additionally requires ReaDDy in the environment.

```bash
MACHINE_PROFILE=<profile> python \
  Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Seeding_Validation.py \
  --total-time-seconds 2.0
```

The one argument is `--total-time-seconds` (required): the run duration, which sets the
timing label (for example `2.0` → `2S_50FPS`) whose labels are checked, so the check
matches the fan-out being validated. The frame cadence (frames per second) is fixed
globally; only the duration varies per run.

Note on seeds: the seed values the check uses — a fixed value for the reproducibility
assertions, no seed for the non-determinism assertions — are internal to the check
functions. This diagnostic has no `--seed` flag of its own; the seed it exercises is
the one the generation entry points accept.

## Outputs

The check writes no files. It prints, to standard output, a header (the timing label,
simulations per task, and the `data_bank_root` the labels are built against), the path
count per split, one `[PASS]` / `[FAIL]` line per check, and a final `RESULT` line. It
exits 0 when every check passes and nonzero when any check fails, so it can gate a
generation launch from a script.

The header names the `data_bank_root` only for transparency; because the check compares
path strings and never touches disk, the verdict is identical whether the machine is
single-tier or routes TRAIN/TEST to a scratch tier and EVAL to permanent storage. The
path builders' `root_for(split)` is used so the displayed paths match the on-disk tier,
but only the labels are compared.

## How to read the result

A `PASS` on the A/B/C checks means the label algebra is collision-free for that timing
label and fan-out shape: every task and simulation maps to its own path, the splits are
namespaced apart, and an incremental append cannot clobber an existing block. This is
the precondition for launching a fan-out safely — especially a grow, where a label
collision would silently overwrite already-generated data.

A `FAIL` on any A/B/C check points to a defect in the path builders or the naming
pattern (a missing field in a filename template, a split suffix that fails to
disambiguate, an offset that overlaps) and must be fixed before generating; the detail
string on the failing line localizes it. A `FAIL` on a D check means the seeding
behavior has drifted from the design — most consequentially, a default draw that no
longer varies run to run, which would indicate a seed was accidentally re-forced,
silently collapsing the intended independent draws.

## What it does and does not guarantee

Because this concerns determinism and seeding, its scope is stated precisely:

- **Guaranteed (exact).** The theta sampler is bit-reproducible under an explicit seed,
  and the file labels are unique — both are exact, deterministic properties the check
  verifies directly. Theta bit-reproducibility is the same determinism pillar the
  reproducibility methodology rests on (the semantic-equivalence pillars in
  `VALIDATION.md`).
- **Detected, not proven.** The non-determinism assertions compare a single pair of
  draws (and placements). Observing that they differ demonstrates the default is not
  forcing a fixed seed; it is a guard against an accidental re-seeding regression, not a
  statistical proof of independence.
- **Out of scope by design.** The check does not exercise the reaction-diffusion
  stepper's own randomness, which is not controlled by the NumPy seed: the placement
  seed governs only the initial particle placement, while the ReaDDy stepper draws its
  own numbers for diffusive motion and reaction-event timing. Trajectories, videos, and
  trained networks therefore vary run to run even at a fixed seed, by design — and the
  label uniqueness this check proves is exactly what keeps that intended variation from
  corrupting the dataset. The complementary theta-only regression test (bit-identical
  theta across same-seed runs, with downstream outputs varying) is described in
  `VALIDATION.md`.
- **Filesystem not touched.** The check proves label uniqueness, not that the paths are
  writable or that the storage tiers are correctly mounted; those are runtime concerns
  of the generation stage itself.

## Caveats

- The placement portion is skipped when ReaDDy is absent, so in an environment without
  ReaDDy the placement guarantees are unverified. The theta and label checks still run
  and still gate the exit code.
- The fan-out is a small representative shape, not the full production task count. It
  exercises the label algebra (within-split, cross-split, offset append), not the volume
  of a production run.
- The check is tied to one timing label per invocation. Run it once per duration when a
  project generates at more than one recording length, since each duration is a separate
  namespace.

## Reference

This check operationalizes, at the level of the HPC fan-out, the reproducibility
guarantees documented in `VALIDATION.md` (the three-pillar semantic-equivalence checks
and the theta-only regression test). The incremental-grow workflow it validates (the
`TASK_OFFSET` append) is defined in the HPC simulation script
`Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh`; the CORE = TRAIN + TEST
split-sizing rule and the leak-proof TRAIN / TEST / EVAL construction are documented in
`PROJECT_CONTEXT.md`.
