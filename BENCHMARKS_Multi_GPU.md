# Multi-GPU timing benchmark — srm-and-sbi-dimer-alp

> Scope: the three GPU stages that shard across every allocated device —
> data-parallel training (Inference), and the sharded MAP passes (Evaluation,
> Experiment). Companion to `BENCHMARKS_Single_GPU.md`, which fixes the
> single-device baseline on a small synthetic check. The numbers here are
> different in kind: they come from the **actual production-scale runs** that
> produced the working 2 s and 5 s posteriors, so this file doubles as a timing
> reference and as a provenance record of those runs. Read the two files together
> — the single-GPU file isolates one device on a micro-check; this one measures
> the real allocation on real data.

## What was measured (2026-07-01)

Two settings, both real pipeline runs (no synthetic micro-check):

- **Single GPU, full production training run** — one RTX 6000 Ada card on a
  shared local workstation (rcl01). A complete 25-epoch 2 s training to the
  working checkpoint.
- **Multi-GPU, production-scale runs** — four MI210 devices of an eight-GPU HPC
  node (Goethe), exclusive allocation, networked scratch filesystem. The 2 s and
  5 s training, evaluation, and experiment runs from the current pipeline.

| Setting | Device(s) | Filesystem | Exclusivity | `num_workers` |
|---|---|---|---|---|
| Single-GPU workstation | 1× RTX 6000 Ada (48 GB) | local NVMe | shared (interactive use + background transfers during the run) | 16 |
| Multi-GPU HPC node | 4× MI210 (ROCm) of 8 | networked scratch | exclusive node allocation | 64 (16/rank) |

Timings are **run-only** — the script's own `Total elapsed` timer and Slurm
`sacct` `Elapsed`; queue wait is excluded. Losses quoted are the TEST-namespace
model-selection score (negative log-probability; more negative is better), given
only as run provenance — this file is about wall-clock, not accuracy.

---

## 1. Training (Inference) — data-parallel

Training shards the batch across all allocated GPUs (distributed data-parallel):
each device holds a replica and processes its slice of every batch, gradients are
averaged, and one device writes the checkpoint. The comparison below holds the
**workload identical** — same tasks, same batch size — so per-epoch wall-clock is
the honest unit of comparison.

### 1a. The direct single-vs-multi comparison — 2 s, identical 50-task workload

Both runs trained the **same** data: 50 TRAIN tasks (50 000 videos), 10 TEST
tasks, batch size 32. Only the hardware and epoch budget differ.

| Setting | Device(s) | Per-epoch (steady) | Full run | Selection loss |
|---|---|---|---|---|
| Single-GPU (rcl01) | 1× RTX 6000 Ada | ~1800 s quiet floor · 3151 s run average | 78 778 s / 25 ep (21.9 h) | −13.74 (25 ep) |
| Multi-GPU (Goethe) | 4× MI210 | ~1267 s (flat) | 13 246 s / 10 ep (3.7 h) | −9.9 (10-ep timing run) |

The single number "N× faster" would be misleading here, because the two settings
differ in **three** ways at once — GPU count, GPU model, and machine exclusivity.
What the per-epoch series actually shows:

- **The MI210 node is flat and predictable.** Steady epochs land at 1267 / 1266 /
  1267 / 1266 s — a spread of under 1 %. On an exclusive node the shard time is
  effectively deterministic, and the one-time compile amortizes to a fixed
  warm-up (~576 s on a resurrect-resumed job; ~898 s on a cold first compile),
  after which every epoch costs the same.
- **The single card is capable but contended.** Its quiet epochs reach a floor of
  ~1720–1880 s, but four of the 25 epochs spiked to 6.4–7.8 ks because the
  workstation was shared (interactive use and background transfers ran during the
  22-hour job). That variance is external load, not a property of the card or the
  code.

So the **actual per-epoch multiplier of the four-card node over the single card
is 1.4×–2.5×** — stated as a range because it depends on which single-card figure
is fair to use:

- **1.4×** against the RTX card's contention-free floor (1267 vs. ~1800 s/epoch)
  — the closest thing to a pure hardware comparison.
- **2.5×** against its full-run average (1267 vs. 3151 s/epoch) — what the
  multi-GPU HPC path actually delivered on this run, shared workstation and all.

In whole-run terms the same 2.5× shows up directly: the single card took **21.9 h**
for the 25-epoch training, whereas the four-card node covers the identical 25
epochs in **≈ 8.9 h**.

The operational takeaway is as much about **predictability** as raw speed: the
exclusive multi-GPU node delivers a fixed, plannable per-epoch time, whereas a
shared workstation delivers a capable-but-variable one. For scheduling long
training, predictability is worth as much as the throughput.

### 1b. 5 s training — 100-task workload, 4× MI210

| Config | Per-epoch (steady) | 10-epoch projection | Observed |
|---|---|---|---|
| 5 s, 100 TRAIN tasks, batch 16 | ~2878 s | ~8.0 h | hit the 8 h wall → TIMEOUT |

The 5 s epoch costs ~2.3× the 2 s epoch (longer videos, and twice the TRAIN
tasks), so a 10-epoch run lands right at the 8-hour wall and times out mid-run.
This is the concrete reason 5 s training relies on the checkpoint-resume
("resurrect") path: a single job cannot finish it within an 8-hour allocation, so
it must resume across jobs (or run on a partition with a longer wall). It is a
crash-safety and wall-budget mechanism, not a workaround for slow code.

---

## 2. Evaluation — sharded MAP recovery, 4× MI210

Evaluation recovers MAP estimates for held-out EVAL videos with known ground
truth. Work is sharded **by task** across ranks (round-robin: task *t* goes to
rank *t* mod *world_size*); each rank runs its videos independently and one rank
merges the shards.

| Timing | EVAL videos | Sharding (5 or 10 tasks over 4 ranks) | Wall-clock | Per-GPU rate |
|---|---|---|---|---|
| 2 s | 5000 (5 tasks × 1000) | busiest rank: 2 tasks = 2000 videos | 6270 s (1:44:30) | ~3.14 s/video |
| 5 s | 5000 (10 tasks × 500) | busiest rank: 3 tasks = 1500 videos | 4992 s (1:23:12) | ~3.33 s/video |

Two things this makes visible:

- **The per-GPU recovery rate is ~3.1–3.3 s/video** and nearly duration-independent
  (5 s is marginally slower — longer videos to embed). Both timings cross-check
  against the same per-video rate; the wall-clocks differ only because the shards
  differ.
- **The wall-clock is set by the busiest shard, not the video total.** Both runs
  process 5000 videos, but 2 s took *longer* — because 5 tasks over 4 ranks is
  uneven (one rank gets 2 tasks / 2000 videos while three ranks get 1 task / 1000
  and then idle), whereas 10 tasks over 4 ranks is closer to balanced (max 1500).
  Sharding is task-granular: when the task count is not a multiple of the rank
  count, the largest shard dictates the wall. Choosing an EVAL task count
  divisible by the allocated rank count keeps all devices busy to the end.

---

## 3. Experiment — sharded real-data MAP, 4× MI210

Experiment applies the posterior to real microscopy cells (no ground truth). Each
cell video is split into non-overlapping temporal chunks; work is sharded by cell.

| Timing | Cells | Chunks/cell | Estimates | Wall-clock | Per-estimate rate |
|---|---|---|---|---|---|
| 2 s | 50 (ALP + BET) | 10 | 500 | 458 s (0:07:38) | ~3.5 s |
| 5 s | 50 (ALP + BET) | 4 | 200 | 257 s (0:04:17) | ~5.0 s |

Experiment is the cheapest GPU stage — minutes, not hours — because it touches
only the real cells (tens of videos), not thousands of synthetic ones. The 2 s
run takes longer than 5 s despite each 5 s chunk being costlier, because a 20 s
cell yields more 2 s chunks (10) than 5 s chunks (4), so there are more estimates
to compute. The per-estimate rate tracks the Evaluation per-video rate (each
chunk estimate is one MAP recovery), with 5 s slower per estimate for the same
longer-video reason.

---

## Interpretation

- **Training** scales usefully across the node, but the honest gain over a single
  strong card is a **1.4×–2.5× range**, dominated by machine exclusivity as much
  as by GPU count and model. The larger practical value is a **flat, predictable
  per-epoch time** on an exclusive allocation.
- **Evaluation** is the heaviest MAP stage (thousands of held-out videos → ~1.5 h
  on four GPUs) and is **load-balanced by task granularity**; keep the EVAL task
  count divisible by the rank count.
- **Experiment** is cheap (minutes) and chunk-count-bound; nothing to optimize.
- **5 s training is wall-bound** at ~8 h for 10 epochs on four MI210, which is why
  it depends on checkpoint-resume rather than a single job.

## Measurement gaps (what these numbers do not establish)

- **No single-MI210 production point.** Without a one-device MI210 run at this
  workload, the training ratio cannot be decomposed into "GPU count" vs. "GPU
  model" vs. "exclusivity" — it is a whole-setting wall-clock ratio only. A clean
  scaling study (1 / 2 / 4 / 8 MI210 at a fixed workload on one node) would isolate
  data-parallel efficiency; it has not been run.
- **No multi-RTX point**, so the RTX side is single-device only.
- **No eight-GPU whole-node point yet.** The full production training runs (2 s
  and 5 s, whole `gpu` node) are queued; when they complete they will add the
  eight-GPU data point and supersede the intermediate posteriors these timings
  came from.
- The single-GPU workstation figures were taken under **shared use**; the quiet
  floor is the better hardware estimate, the average the better cost estimate.

## Extrapolation (`T ≈ per-epoch × epochs + warm-up`, run-only)

- **2 s, 25-epoch training on 4× MI210:** ~1267 s × 25 + ~600 s warm-up ≈ **8.9 h**
  — versus 21.9 h observed on the shared single card.
- **5 s, 25-epoch training on 4× MI210:** ~2878 s × 25 ≈ **20 h** → must span
  multiple jobs via checkpoint-resume, or run on a longer-wall partition.
- **Evaluation** scales as `busiest-shard videos × ~3.2 s`; balance the shards to
  approach `total videos × 3.2 s ÷ ranks`.
- **Eight-GPU node** (pending runs) should roughly halve the four-GPU training
  per-epoch and the Evaluation wall, subject to the same task-divisibility caveat
  for the sharded stages.
