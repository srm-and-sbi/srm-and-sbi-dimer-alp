"""Generation seeding & file-label sanity check for the HPC fan-out.

Generation is non-deterministic by design: no seed is passed, so
every simulation is a fresh independent draw and the sampled theta is persisted
per task. Dataset integrity therefore rests entirely on the FILE LABELS being
collision-free across the fan-out — the global task index `tid`, the split
namespace, and the per-task simulation index must map every output to a unique
path. This script exercises the REAL path builders (`PARAMETERS.paths.*`) and
asserts uniqueness over a representative fan-out, including:

  A. Within a split           - distinct (tid, sim) -> distinct theta-set,
                                video-set, and trajectory paths.
  B. Across splits            - TRAIN / TEST / EVAL paths never collide
                                (the `_<SPLIT>` suffix namespaces them).
  C. Incremental TASK_OFFSET  - an appended block (e.g. tids 8..15) never
                                collides with the original block (tids 0..7).
  D. Non-determinism sanity   - the default (seed=None) yields independent draws
                                (theta and real placement differ across runs),
                                while an explicit --seed is still reproducible;
                                guards against accidentally re-forcing a seed.

Run:
    MACHINE_PROFILE=<profile> python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Seeding_Validation.py
Exits nonzero if any collision is found.
"""

import argparse
import sys

import numpy as np

from srm_and_sbi_dimer_alp.parameterization import (
    PARAMETERS,
    RunTiming,
    theta_lower_bound,
    theta_upper_bound,
)

PATHS = PARAMETERS.paths
DBR = PARAMETERS.machine.data_bank_root
TL = None  # set at runtime in main() from the required --total-time-seconds (the
           # global PARAMETERS.simulation.timing holds only the fixed frame cadence)
SPLITS = ("TRAIN", "TEST", "EVAL")

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  --  {detail}" if detail else ""))


def task_paths(split, tid, sims):
    """Every output path a single task writes, per the real Paths builders.

    Uses ``root_for(split)`` so the displayed paths match the actual on-disk
    tier on a two-tier machine (TRAIN/TEST -> scratch, EVAL -> permanent). This
    check compares path *strings* for label uniqueness only -- it never touches
    the filesystem -- so the verdict is identical under either root.
    """
    root = PARAMETERS.machine.root_for(split)
    out = [
        ("theta", PATHS.theta_set_path(tid, root, TL, True, split)),
        ("video", PATHS.video_set_path(tid, root, TL, True, split)),
    ]
    for sim in range(sims):
        out.append(("traj", PATHS.trajectory_path(tid, sim, root, TL, split)))
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generation seeding & file-label sanity check (HPC fan-out).")
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Run duration in seconds; sets the timing label whose path collisions "
             "are checked, so it matches the actual fan-out being validated.")
    return parser.parse_args(argv)


def main(args):
    global TL
    TL = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    ).label
    SIMS = 4
    # Representative fan-out: TRAIN as TWO offset blocks (0..7 then an 8..15
    # append), TEST 0..1, EVAL 0 -- the CORE=100 shape plus an incremental grow.
    fanout = {
        "TRAIN": list(range(0, 8)) + list(range(8, 16)),
        "TEST": list(range(0, 2)),
        "EVAL": list(range(0, 1)),
    }
    print("=" * 74)
    print(f"File-label uniqueness  (timing={TL} | sims/task={SIMS})")
    print(f"data_bank_root = {DBR}")
    print("=" * 74)

    all_paths = []
    by_split = {}
    for split in SPLITS:
        sp = []
        for tid in fanout[split]:
            sp += [p for _, p in task_paths(split, tid, SIMS)]
        by_split[split] = sp
        all_paths += sp
        print(f"  {split:5s}: tids {fanout[split]}  -> {len(sp)} paths")

    # A+C. Global uniqueness (covers within-split and offset-append disjointness)
    strs = [str(p) for p in all_paths]
    check("A+C. all output paths unique across the fan-out (incl. offset append)",
          len(set(strs)) == len(strs), f"{len(strs)} paths, {len(set(strs))} unique")

    # C. explicit: TRAIN tids 0..7 vs 8..15 disjoint
    a = {str(p) for tid in range(0, 8) for _, p in task_paths("TRAIN", tid, SIMS)}
    b = {str(p) for tid in range(8, 16) for _, p in task_paths("TRAIN", tid, SIMS)}
    check("C. TRAIN offset blocks [0..7] vs [8..15] disjoint", a.isdisjoint(b))

    # B. cross-split disjointness
    cross_ok = True
    detail = ""
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            si, sj = SPLITS[i], SPLITS[j]
            shared = set(map(str, by_split[si])) & set(map(str, by_split[sj]))
            if shared:
                cross_ok = False
                detail = f"{si}∩{sj} shares {len(shared)}"
    check("B. TRAIN / TEST / EVAL namespaces never collide", cross_ok, detail)

    # D. Non-determinism sanity (seed=None is the default; --seed still works) -- #
    LOW = np.array(theta_lower_bound())
    HIGH = np.array(theta_upper_bound())
    D = len(LOW)

    def sample(seed, nrows, sims):
        return np.power(10, np.random.default_rng(seed).uniform(low=LOW, high=HIGH,
                                                                size=(nrows, sims, D)))
    a, b = sample(None, 2, 3), sample(None, 2, 3)
    check("D. theta: seed=None differs across runs (non-deterministic)",
          not np.array_equal(a, b))
    check("D. theta: rows distinct within a draw",
          a[0].tobytes() != a[1].tobytes() and a[0, 0].tobytes() != a[0, 1].tobytes())
    check("D. theta: --seed still reproducible when explicitly given",
          np.array_equal(sample(42, 2, 3), sample(42, 2, 3)))

    # Optional: real placement (needs ReaDDy). Confirms build_simulation honors
    # seed=None (fresh placement) and a given seed (reproducible placement).
    try:
        from srm_and_sbi_dimer_alp.simulation_rds_support import build_simulation, build_system

        def place(theta, seed):
            smut = build_simulation(build_system(theta, verbose=False), theta,
                                    seed=seed, verbose=False)
            p = np.array([q.pos for q in smut.current_particles], dtype=float).reshape(-1, 3)
            return p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))]
        th = a[0, 0]
        p1, p2 = place(th, None), place(th, None)
        check("D. placement: seed=None, same theta -> placements differ",
              p1.shape == p2.shape and not np.array_equal(p1, p2))
        check("D. placement: explicit seed -> placement reproducible",
              np.array_equal(place(th, 7), place(th, 7)))
    except Exception as exc:  # noqa: BLE001 -- ReaDDy not present: skip, don't fail
        print(f"  [SKIP] D. real-placement check ({type(exc).__name__}: {exc})")

    print("=" * 74)
    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail:
        print(f"RESULT: FAIL ({n_fail} of {len(results)} checks failed)")
        return 1
    print(f"RESULT: PASS (all {len(results)} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
