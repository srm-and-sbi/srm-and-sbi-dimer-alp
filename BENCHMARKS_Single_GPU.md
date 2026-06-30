# Single-GPU timing benchmark — srm-and-sbi-dimer-alp

> Scope: **single GPU per machine** (one device used, no DataParallel/sharding).
> These are the dated single-GPU baseline numbers. Data-parallel training and
> MAP-recovery sharding shipped in 0.2.0; their timings belong in a companion
> `BENCHMARKS_Multi_GPU.md` so the single-vs-multi comparison stays a direct one.

## Baseline (2026-06-18)

Cross-machine timing for the standard check: **16 train / 4 test / 2 eval tasks @
10 sims/task**, batch size 32, 10 epochs. **Single GPU per machine** (one device
used, no data-parallel training or sharding, to fix this baseline);
`num_workers` = 4 (workstations) / 8 (HPC node). HPC times are
**run-only** (`sacct Elapsed` / script `Total elapsed`; Slurm queue excluded).
These timings characterize the single-GPU configuration with current data-I/O
(worker-count) and compile settings; the optimization opportunities below are
measured against them.

### Machines
| Machine | Device | Storage | Notes |
|---|---|---|---|
| Workstation (CPU only) | CPU only | local | Quadro T2000 unusable: CPU-only torch build + driver 470 / ~4 GB VRAM |
| CUDA workstation (RTX 6000 Ada, 48 GB) | 1× RTX 6000 Ada (48 GB) | local SSD | |
| AMD HPC node (MI210) | 1× MI210 (ROCm) of 8 | networked scratch filesystem | |

### Stage timings (run-only)
| Stage | Workstation (CPU only) | RTX 6000 Ada | MI210 |
|---|---|---|---|
| Generation (22 tasks, 220 sims) | 3410 s (sequential) | 1985 s (sequential) | parallel-packed; ~5–9 min/node-element (~5.3 s/sim node throughput) |
| Inference — total (10 epochs) | 6396 s | 78 s | 1241 s |
| · epoch 1 (compile) | 652 s | 12 s | **898 s** |
| · steady / epoch | ~638 s | **6.7 s** | ~29 s |
| Evaluation (20 EVAL videos) | (suppressed) | 527 s (~26 s/video) | 1047 s (~52 s/video; task 1 steady ~44 s) |
| Experiment (per cell) | (suppressed) | ~260 s/cell (4 cells: 2 ALP+2 BET) | ~437 s/cell steady (2 ALP cells; see caveat) |

### Normalized per-unit rates (steady, single-GPU)
- **Inference, s/epoch:** CPU ~638 · RTX 6000 Ada **6.7** · MI210 ~29  → RTX is ~95× CPU and ~4.3× MI210.
- **MAP recovery, s/video-or-chunk (1000 opt steps):** RTX ~26 · MI210 ~44  → MI210 ~1.7× slower.

### Interpretation
- The **inference gap (RTX 4.3× MI210) exceeds the recovery gap (~2×)**. MAP recovery reads the posterior once and is compute-bound; inference decompresses 160 `.zarr` videos/epoch from the networked scratch filesystem with only 8 workers. The extra ~2× in inference is therefore **data-I/O bound** → motivates more `DataLoader` workers on the HPC.
- The **MI210 `torch.compile` cold-start (~898 s) recompiles every run** (Inductor cache not persisting — likely per-job `/tmp`) → recurring cost, fixable via a persistent `TORCHINDUCTOR_CACHE_DIR`.

### Extrapolation (`T ≈ rate × units ÷ parallelism`)
- **50-cell production Experiment, single-GPU:** ~3.6 h (RTX 6000 Ada rate) / ~7.3 h (MI210 rate). **8-GPU shard → ~27–55 min** (embarrassingly parallel across cells).
- **Production inference:** steady epoch scales ~linearly with #batches (data size); plus the one-time compile (amortized, or cached).

### Notes on this benchmark run
- The MI210 node's Experiment run processed **2 ALP cells only** — `--export=ALL,KINDS=ALP,BET` split on the comma (→ `KINDS=ALP`). Per-cell rate valid; for full ALP+BET, rely on the `.sh` `KINDS` default rather than overriding via a comma-bearing `--export`.
- The HPC node's generation scheduled its two TRAIN array elements on the **same node** (sequential), not two nodes — scheduler placement, not a code issue.

### Known optimization opportunities (to be measured against this baseline)
1. **Data I/O:** the HPC node's `num_workers` 8 → 16 → 32; re-measure inference steady epoch.
2. **Compile:** add a persistent Inductor cache and a `--compile/--no-compile` flag, then re-measure the cold-start cost.
3. **Multi-GPU:** data-parallel training and MAP-recovery sharding across the node's GPUs shipped in 0.2.0; their timings belong in the companion `BENCHMARKS_Multi_GPU.md`, measured against this single-GPU baseline.
