"""Entry-point: generate the TRAIN / TEST / EVAL datasets in one command.

Runs the full RDS -> DLI flow three times -- once per split -- with the task
counts derived from the dataset-sizing rule. The three namespaces are
independent draws that never mix: by default no seed is passed, so each split
draws fresh entropy (non-deterministic).

Sizing rule (CORE = TRAIN + TEST):
    test_tasks  = round(test_fraction * core_tasks)              # default 0.2
    train_tasks = core_tasks - test_tasks                        # 0.8 of CORE
    eval_tasks  = max(ceil(eval_floor / sims), round(eval_fraction * core_tasks))
                                                                 # EVAL = max(floor, 0.1*CORE)
So EVAL is floored at `eval_floor` samples (for a stable recovery number) and
only the proportional `0.1 * CORE` exceeds it once CORE >= eval_floor / eval_fraction.

Seeds: none by default (non-deterministic; the global task index is the
provenance handle and theta is persisted per task). An optional --seed restores
a deterministic distinct-seed-per-split run: train = seed, test = seed + 1,
eval = seed + 2.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py \\
        --core-tasks 10 --task-simulations 10 --seed None
    # preview the plan without generating anything:
    ... --core-tasks 10 --task-simulations 10 --dry-run
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from srm_and_sbi_dimer_alp.parameterization import PARAMETERS

_PRIME = Path(__file__).resolve().parent
_RDS = _PRIME / "SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py"
_DLI = _PRIME / "SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py"


def _stage_label(script: Path) -> str:
    """RDS / DLI, derived from the script filename."""
    return script.stem.split("_")[-1]


def _run(script: Path, split: str, tasks: int, seed: Optional[int],
         args: argparse.Namespace) -> None:
    """Run one stage (RDS or DLI) for one split, or print it under --dry-run."""
    cmd = [
        sys.executable, str(script),
        "--split", split,
        "--tasks", str(tasks),
        "--task-simulations", str(args.task_simulations),
        "--total-time-seconds", str(args.total_time_seconds),
    ]
    if seed is not None:                  # default: omit --seed -> non-deterministic
        cmd += ["--seed", str(seed)]
    if args.no_compress:
        cmd.append("--no-compress")

    label = f"{_stage_label(script):<3} {split:<5} (--tasks {tasks}, --seed {seed})"
    if args.dry_run:
        print(f"  [dry-run] {label}")
        return

    print(f"  - {label} ...", end="", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(" FAILED")
        raise SystemExit(
            f"FAILED: {script.name} --split {split} (exit code {result.returncode}). "
            f"Re-run that command directly to see the error.")
    print(f" done ({time.time() - t0:.1f}s)")


def _plan(args: argparse.Namespace) -> dict:
    """Compute per-split task counts and seeds from the sizing rule."""
    cfg = PARAMETERS.inference.training
    test_fraction = args.test_fraction if args.test_fraction is not None else cfg.test_fraction
    eval_fraction = args.eval_fraction if args.eval_fraction is not None else cfg.eval_fraction
    eval_floor = args.eval_floor if args.eval_floor is not None else cfg.eval_floor

    sims = args.task_simulations
    core_tasks = args.core_tasks
    core_samples = core_tasks * sims

    if core_samples < cfg.core_min:
        raise SystemExit(
            f"CORE = {core_samples} samples ({core_tasks} tasks x {sims} sims) is "
            f"below the minimum of {cfg.core_min}. Increase --core-tasks or "
            f"--task-simulations.")

    test_tasks = max(1, round(test_fraction * core_tasks))
    train_tasks = core_tasks - test_tasks
    if train_tasks < 1:
        raise SystemExit(
            f"train_tasks = {train_tasks} < 1 (core_tasks={core_tasks}, "
            f"test_tasks={test_tasks}); increase --core-tasks.")
    eval_tasks = max(math.ceil(eval_floor / sims), round(eval_fraction * core_tasks))

    # Default: no seed -> each split draws fresh entropy (non-deterministic).
    # An explicit --seed restores a distinct-seed-per-split run for optional determinism.
    base_seeds = (None, None, None) if args.seed is None else (
        args.seed, args.seed + 1, args.seed + 2)

    return {
        "sims": sims,
        "core_tasks": core_tasks,
        "core_samples": core_samples,
        "test_fraction": test_fraction,
        "eval_fraction": eval_fraction,
        "eval_floor": eval_floor,
        "splits": [
            ("train", train_tasks, base_seeds[0]),
            ("test", test_tasks, base_seeds[1]),
            ("eval", eval_tasks, base_seeds[2]),
        ],
    }


def main(args: argparse.Namespace) -> None:
    plan = _plan(args)
    sims = plan["sims"]
    div = "=" * 72

    print(div)
    print(" SRM_AND_SBI_DIMER_ALP — Generate_Datasets"
          + ("   [DRY RUN]" if args.dry_run else ""))
    print(div)
    print(f"  task_simulations : {sims}")
    print(f"  CORE             : {plan['core_tasks']} tasks  "
          f"({plan['core_samples']} samples)")
    print(f"  rule             : test_fraction={plan['test_fraction']}, "
          f"eval_fraction={plan['eval_fraction']}, eval_floor={plan['eval_floor']}")
    print(f"  total-time-secs  : {args.total_time_seconds}")
    print()
    print(f"  {'split':<7}{'tasks':>7}{'samples':>9}{'seed':>7}")
    for split, tasks, seed in plan["splits"]:
        print(f"  {split.upper():<7}{tasks:>7}{tasks * sims:>9}{str(seed):>7}")
    print(div)

    run_start = time.time()
    for split, tasks, seed in plan["splits"]:
        print(f"\n[{split.upper()}]")
        _run(_RDS, split, tasks, seed, args)
        _run(_DLI, split, tasks, seed, args)

    if args.dry_run:
        print("\nDry run complete -- nothing was generated.")
    else:
        print(f"\nAll three datasets generated in {time.time() - run_start:.1f}s.")


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Generate TRAIN/TEST/EVAL datasets (RDS->DLI per split) "
                    "per the dataset-sizing rule.",
    )
    parser.add_argument(
        "--core-tasks", type=int, required=True,
        help="Number of TRAIN+TEST (CORE) task files; split 0.8/0.2 into train/test.",
    )
    parser.add_argument(
        "--task-simulations", type=int, default=10,
        help="Simulations per task (default: 10).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Simulation duration per video in seconds (required; e.g. 2.0, 5.0).",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Optional base seed; if set, train=seed, test=seed+1, eval=seed+2. "
             "Default None -> non-deterministic, each split a fresh draw.",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=None,
        help="TEST as a fraction of CORE (default from config: 0.2).",
    )
    parser.add_argument(
        "--eval-fraction", type=float, default=None,
        help="EVAL as a fraction of CORE (default from config: 0.1).",
    )
    parser.add_argument(
        "--eval-floor", type=int, default=None,
        help="Minimum EVAL samples for a stable recovery number "
             "(default from config: 10).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save sets as .npy instead of compressed .zarr (passed to RDS/DLI).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the computed sizing plan and the per-split commands without "
             "generating anything.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
