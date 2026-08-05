"""Small cross-cutting helpers.

Visual-separator string constants for terminal output (named seek/sink/sock,
uppercased per Python constant convention).

System diagnostics: log_memory_state() prints CPU + GPU memory state;
probe_resources() returns (threads, open_fds, rss_mb) for the current process.

Lightweight imports — safe to import at module top-level anywhere.
"""

import contextlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch


# =============================================================================
# Console transcript capture (--debug-dump)
# =============================================================================

class _Tee:
    """A minimal write-to-many stream: mirror writes to several file objects."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout(path: Path):
    """Mirror everything written to ``stdout`` into ``path`` for the duration.

    Leaves a complete console transcript in the project folder so a debug run
    never depends on shell redirection. tqdm / sbi progress bars write to
    ``stderr`` and are intentionally not captured (carriage-return bars are not
    file-friendly). ``stdout`` is always restored on exit, even on error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, handle)
    try:
        yield path
    finally:
        sys.stdout = original
        handle.close()


def console_log_context(args, stage: str, paths=None, data_bank_root=None):
    """Return a context manager that tees stdout to the run's ``console.log``.

    Under ``--debug-dump`` (``args.debug_dump`` truthy) the transcript is written
    to ``<data_bank>/Labor/Debug/<run_label>/<stage>/console.log`` -- the same
    process-level directory the DiagnosticReporter dumps into. Otherwise -- or
    under ``--dry-run``, which performs no real work to transcribe -- a no-op
    context. ``args`` is expected to carry ``total_time_seconds`` and optionally
    ``split`` (RDS / DLI); both are read defensively.

    ``paths`` / ``data_bank_root`` select the namespace. ``paths`` defaults to the
    canonical ``PARAMETERS.paths``; ``data_bank_root`` defaults to
    ``PARAMETERS.machine.root_for(split)`` -- the same per-split tier the
    DiagnosticReporter dumps into (the scratch tier for TRAIN/TEST) -- so the
    transcript co-locates with the dump instead of splitting across filesystems on
    a two-tier machine. A Detector stage passes its aliased
    Paths (``detector_parameterization.detector_paths(...)``) so the ``run_label``
    carries the ``_DETECTOR`` tag and the transcript lands in a
    Detector-identifiable Debug directory, never colliding with the canonical one.
    """
    if not getattr(args, "debug_dump", False) or getattr(args, "dry_run", False):
        return contextlib.nullcontext()
    # Imported here (not at module top) to avoid a circular import at load time.
    from .parameterization import PARAMETERS, RunTiming
    paths = paths if paths is not None else PARAMETERS.paths
    split = getattr(args, "split", None)
    split = split.upper() if split else None
    if data_bank_root is None:
        # Co-locate the transcript with the DiagnosticReporter dump, which lives on
        # root_for(split) -- the scratch tier for TRAIN/TEST, the permanent tier
        # otherwise. Defaulting to the permanent data_bank_root regardless of split
        # split a TRAIN/TEST debug bundle across two filesystems on a two-tier machine.
        # Guard the split-less case (root_for expects a split string) with the permanent tier.
        data_bank_root = (
            PARAMETERS.machine.root_for(split) if split
            else PARAMETERS.machine.data_bank_root
        )
    total_time_seconds = getattr(args, "total_time_seconds", None)
    if total_time_seconds is None:
        raise ValueError(
            "console_log_context needs args.total_time_seconds (the per-run duration); "
            "every entry-point enabling --debug-dump must pass --total-time-seconds."
        )
    timing = RunTiming(
        total_time_seconds=total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    path = paths.debug_run_dir(
        data_bank_root, timing.label, stage, split) / "console.log"
    return tee_stdout(path)


# =============================================================================
# Visual separators for terminal output
# =============================================================================
# Used in print() calls throughout the codebase to make log output scannable.
# Three lengths corresponding to three nesting levels.

SEEK = "=" * 4    # short separator (e.g., subsection within a function)
SINK = "·" * 8    # medium separator (e.g., between simulations within a task)
SOCK = "~" * 16   # long separator (e.g., between tasks)


# =============================================================================
# Memory-state diagnostic
# =============================================================================

def _bytes_to_human(n: int) -> str:
    """Convert byte count to a human-readable string (1.5G, 250M, 64K, etc.).

    Self-contained so the codebase does not depend on psutil's private
    ``psutil._common.bytes2human`` API.
    """
    if n < 0:
        return f"-{_bytes_to_human(-n)}"
    for unit in ("B", "K", "M", "G", "T", "P"):
        if n < 1024:
            return f"{n:.2f}{unit}" if isinstance(n, float) else f"{n}{unit}"
        n = n / 1024
    return f"{n:.2f}E"  # exabytes and beyond


def log_memory_state(prefix: str = "") -> None:
    """Print current CPU and GPU memory state to stdout.

    GPU info printed only if CUDA is available; otherwise just CPU info.
    Used during inference training to monitor memory pressure.

    Args:
        prefix: Optional prefix string prepended to each line (e.g., "[Epoch 5]").
    """
    tag = f"{prefix} " if prefix else ""

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gpu_info = {
            "current_device": device,
            "memory_free": _bytes_to_human(free_bytes),
            "memory_total": _bytes_to_human(total_bytes),
            "memory_used": _bytes_to_human(total_bytes - free_bytes),
        }
        print(f"{tag}GPU {gpu_info}", flush=True)

    cpu_memory = psutil.virtual_memory()
    cpu_info = {
        "memory_total": _bytes_to_human(cpu_memory.total),
        "memory_available": _bytes_to_human(cpu_memory.available),
        "memory_used": _bytes_to_human(cpu_memory.used),
    }
    print(f"{tag}CPU {cpu_info}", flush=True)


def probe_resources():
    """Return (threads, open_fds, rss_mb) for the current process.

    Lightweight per-iteration resource probe used by the simulation stages'
    ``--probe`` flag to detect resource accumulation across simulations (the
    instrumentation used to diagnose per-simulation resource leaks). psutil-based; cheap.
    """
    proc = psutil.Process()
    try:
        threads = proc.num_threads()
    except Exception:
        threads = -1
    try:
        fds = proc.num_fds()
    except Exception:
        fds = -1
    try:
        rss_mb = proc.memory_info().rss // (1024 * 1024)
    except Exception:
        rss_mb = -1
    return threads, fds, rss_mb


def log_resource_limits() -> None:
    """Log the process resource limits and a baseline resource snapshot.

    The startup half of the simulation stages' ``--probe`` instrumentation; the
    per-step half is ``probe_resources``. Shared by the RDS and DLI stages so
    both report identical diagnostics. Logging only.
    """
    import resource  # Linux rlimits; imported lazily (platform-specific)
    soft_u, hard_u = resource.getrlimit(resource.RLIMIT_NPROC)
    soft_f, hard_f = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"[probe] host={os.uname().nodename} pid={os.getpid()}")
    print(f"[probe] RLIMIT_NPROC (threads/procs): soft={soft_u} hard={hard_u}")
    print(f"[probe] RLIMIT_NOFILE (open files):   soft={soft_f} hard={hard_f}")
    threads, fds, rss_mb = probe_resources()
    print(f"[probe] baseline: threads={threads} fds={fds} rss_mb={rss_mb}", flush=True)
