"""Diagnostic reporting for the DIMER pipeline (debug mode).

This module powers the ``--debug`` / ``--debug-dump`` modes of the three
entry-point scripts. Its single goal is to make the behavior of each
pipeline step *clear to the user*: every step can announce its state (a
"checkpoint"), assert the invariants that must hold (a "check"), and record
quantitative diagnostics (a "stat"). In ``--debug-dump`` mode the same
information is also written to a self-contained Markdown report with linked
PNG figures, so a run leaves an auditable diagnostic trail.

When debug mode is off, every method is a cheap no-op: the reporter can be
threaded through the pipeline at zero cost to normal runs.

Three kinds of diagnostic output, all routed through ``DiagnosticReporter``:

  1. Checks  -- pass/fail invariants. By default a failed check is *fatal*:
                it raises ``DiagnosticCheckError`` naming the stage, the
                check, and the offending value, so a problem surfaces exactly
                where it happens instead of as a confusing downstream crash.
  2. Stats   -- quantitative values, optionally with an expected range.
  3. Figures -- matplotlib figures saved headlessly as PNG (dump mode only).

Console output is the always-on channel in ``--debug``; the Markdown report
and PNG figures are added in ``--debug-dump``. Lightweight top-level imports
(numpy + stdlib); matplotlib is never imported here -- figures are passed in
already-built and saved via ``Figure.savefig``, which needs no display.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .utils import SEEK, SINK, SOCK


class DiagnosticCheckError(AssertionError):
    """Raised when a fatal invariant check fails in debug mode.

    The message names the stage, the check, and the offending detail so the
    failure is located immediately rather than surfacing as a downstream
    crash (for example, a missing-directory error several steps later).
    """


# Plain-language meaning of each standard (convenience-checker) check kind.
# Used to auto-build the report's "What these mean" legend so the report is
# understandable without prior knowledge of the codebase. Keyed by the check
# name prefix (the part before "(").
_CHECK_GLOSSARY = {
    "no_nan_inf": "the array contains no NaN and no infinite values.",
    "shape": "the array has the expected dimensions.",
    "stochastic_matrix": ("each row is a valid probability distribution "
                          "(sums to 1, all entries non-negative)."),
    "in_bounds": "every value lies within the stated [low, high] range.",
    "finite": ("the scalar is a finite number (not NaN/infinity) -- e.g. a "
               "training loss that did not blow up."),
    "positive": "the scalar is strictly greater than zero.",
    "file": "the output file exists on disk and is non-empty (the write succeeded).",
}


# Display labels for the long-form UNIT strings in PARAMETERIZATION, for compact
# table rendering.
_UNIT_DISPLAY = {
    "Count": "count",
    "Square Micrometer Per Second": "um^2/s",
    "Square Micrometer Per (Count*Second)": "um^2/(count*s)",
    "Count Per Second": "1/s",
    "Dimensionless": "dimensionless",
    "": "",
}


def prior_sampling_table(parameterization, theta):
    """Build (headers, rows) summarizing prior ranges vs sampled values.

    For each learnable parameter, reports the log-uniform prior bounds (log10),
    the sampled value in log10 and in physical units, and the unit -- so a
    reader can confirm, for example, that the initial particle counts a
    simulation was seeded with match what the rendered video shows.

    Args:
        parameterization: the PARAMETERIZATION list (dicts with KEY,
            PRIOR_RANGE, UNIT), in the same order as ``theta``.
        theta: physical (already exponentiated) sampled parameter values.

    Returns:
        ``(headers, rows)`` ready to pass to ``DiagnosticReporter.table``.
    """
    headers = ["parameter", "label", "prior log10", "sampled log10",
               "sampled value", "units", "derived unit", "kind"]
    rows = []
    for i, para in enumerate(parameterization):
        lo, hi = para["PRIOR_RANGE"]
        value = float(theta[i])
        log10 = np.log10(value) if value > 0 else float("nan")
        unit = _UNIT_DISPLAY.get(para.get("UNIT", ""), para.get("UNIT", ""))
        derived = para.get("DERIVED_UNIT")
        derived_disp = _UNIT_DISPLAY.get(derived, derived) if derived else "-"
        rows.append([
            para["KEY"],
            para.get("LABEL") or "-",
            f"[{lo:+.2f}, {hi:+.2f}]",
            f"{log10:+.3f}",
            f"{value:.4g}",
            unit,
            derived_disp,
            (para.get("NOTE") or "").replace(" Parameter", "") or "-",
        ])
    return headers, rows


def fixed_parameters_table(parameterization_raw):
    """Build (headers, rows) for the fixed (non-learnable) parameters.

    These are the entries with no prior (``PRIOR_RANGE is None``) -- the Known
    scientific constants and tuning Hyperparameters held the same across every
    simulation. Columns parallel the sampled-parameter table where applicable
    (there are no prior / sampled columns, since each value is fixed).

    Args:
        parameterization_raw: the flat PARAMETERIZATION_RAW list (all entries).

    Returns:
        ``(headers, rows)`` ready to pass to ``DiagnosticReporter.table``.
    """
    headers = ["parameter", "label", "value", "units", "derived unit", "kind"]
    rows = []
    for para in parameterization_raw:
        # Keep only truly fixed rows: a concrete VALUE with no PRIOR_RANGE. Learnable rows
        # (concrete VALUE + range) appear in the prior-sampling table; the marginalized rows
        # -- the SCOPE camera nuisance-from-spec (VALUE sentinel + range) and the
        # calibrated-imaging nuisance-from-object (VALUE sentinel + no range) -- are drawn
        # per simulation, not held fixed, so they belong in neither table. The value-based
        # 'fixed' rule (parameterization.role_of) is replicated inline rather than imported
        # to preserve this module's decoupling from the machine-profile-loading
        # parameterization module; the sentinel strings mirror its NUISANCE_SENTINEL /
        # POSTERIOR_SENTINEL. A ranged row, or a sentinel VALUE, is therefore not fixed.
        value = para.get("VALUE")
        is_sentinel = isinstance(value, str) and value in ("NUISANCE", "POSTERIOR")
        if is_sentinel or para.get("PRIOR_RANGE") is not None:
            continue
        if isinstance(value, (list, tuple)):
            value_str = str(list(value))
        elif isinstance(value, float):
            value_str = f"{value:.4g}"
        else:
            value_str = str(value)
        unit_raw = para.get("UNIT")
        unit = _UNIT_DISPLAY.get(unit_raw, unit_raw) if unit_raw else "-"
        derived = para.get("DERIVED_UNIT")
        derived_disp = _UNIT_DISPLAY.get(derived, derived) if derived else "-"
        rows.append([
            para["KEY"],
            para.get("LABEL") or "-",
            value_str,
            unit,
            derived_disp,
            (para.get("NOTE") or "").replace(" Parameter", "") or "-",
        ])
    return headers, rows


class DiagnosticReporter:
    """Collect and report per-stage diagnostics for one pipeline run.

    A single reporter is created per stage (RDS, DLI, Inference) in the
    entry-point script and used to announce checkpoints, assert invariants,
    record statistics, and (in dump mode) save figures. Call ``summary()``
    at the end of the stage for the console PASS/FAIL tally, and
    ``write_report()`` to emit the Markdown report when dumping.

    Args:
        stage: Short stage name shown in all output (e.g., "RDS", "DLI").
        enabled: Master switch. When False every method is a no-op, so the
            reporter is free to thread through a normal (non-debug) run.
        dump: When True (and ``enabled``), also persist a Markdown report and
            PNG figures under ``dump_dir``. Silently downgraded to False if
            ``dump_dir`` is missing or the disk is nearly full.
        dump_dir: Destination directory for the report and ``figures/``.
        run_label: Human-facing run identifier for the report header
            (e.g., "SRM_AND_SBI_DIMER_ALP_2S_50FPS_TASK_0").
        timestamp: Run time for the report header, as a formatted string. Optional:
            when omitted the reporter stamps the current UTC time itself, so a report
            can never be written without a date and time.
        run_note: Optional free-text remark about the run, rendered on its own line
            beneath the timestamp. Anything a caller wants to SAY about a run belongs
            here; passing prose as ``timestamp`` would displace the stamp.
    """

    # Minimum free space required before writing dump artifacts. Guards the
    # FAT32 data drives, which can be nearly full; console diagnostics stay on.
    _MIN_FREE_BYTES = 200 * 1024 * 1024  # 200 MB headroom

    def __init__(self,
                 stage: str,
                 enabled: bool = False,
                 dump: bool = False,
                 dump_dir: Optional[Union[str, Path]] = None,
                 run_label: str = "",
                 timestamp: str = "",
                 run_note: str = "") -> None:
        self.stage = stage
        self.enabled = bool(enabled)
        self.dump = bool(dump) and self.enabled
        self.run_label = run_label
        # Every report states when it was produced. Defaulted here rather than required
        # from the caller so that no entry point -- a stage runner, a smoke, or a one-off
        # re-analysis harness -- can emit an undated report. A caller with a remark about
        # the run passes ``run_note``, which is rendered separately and never displaces
        # the stamp.
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC")
        self.run_note = run_note
        self.dump_dir = Path(dump_dir) if dump_dir else None

        self._checks = []        # list of (name, passed, detail)
        self._stats = []         # list of (name, value, expected, note)
        self._tables = []        # list of (title, headers, rows, note)
        self._figures = []       # list of (name, relative_png_path, caption)
        self._check_notes = {}   # custom check name -> plain-language note

        if self.enabled:
            header = f"[{self.stage}] diagnostics"
            print(f"\n{SOCK}  {header}  {SOCK}", flush=True)
        if self.dump:
            self._prepare_dump_dir()

    # ------------------------------------------------------------------
    # Checkpoints: announce a step and print its key state values.
    # ------------------------------------------------------------------
    def checkpoint(self, label: str, **state) -> None:
        """Announce a pipeline step and print its key state values."""
        if not self.enabled:
            return
        print(f"{SEEK} checkpoint: {label} {SEEK}", flush=True)
        for key, value in state.items():
            print(f"  {key:<24}: {self._fmt(value)}", flush=True)

    # ------------------------------------------------------------------
    # Checks: pass/fail invariants. Fatal by default (raise on failure).
    # ------------------------------------------------------------------
    def check(self, name: str, passed: bool, detail: str = "",
              fatal: bool = True, note: str = "") -> bool:
        """Record a pass/fail invariant, print it, and raise if fatal-failed.

        Args:
            name: Short check name (shown in console and report).
            passed: Whether the invariant holds.
            detail: One-line explanation or the observed value.
            fatal: If True (default) a failed check raises
                ``DiagnosticCheckError``; if False the failure is recorded and
                printed as a warning but the run continues (useful when you
                want the full report even on failure).
            note: Optional plain-language meaning for a *custom* check (the
                standard convenience checks are explained automatically via the
                report legend). Shown in the report's "What these mean" section.

        Returns:
            The boolean pass/fail (True when disabled, so callers can branch).
        """
        if not self.enabled:
            return True
        passed = bool(passed)
        self._checks.append((name, passed, detail))
        if note:
            self._check_notes[name] = note
        tag = "PASS" if passed else "FAIL"
        suffix = f" - {detail}" if detail else ""
        print(f"  [{tag}] {name}{suffix}", flush=True)
        if not passed and fatal:
            raise DiagnosticCheckError(
                f"[{self.stage}] CHECK FAILED: {name}{suffix}"
            )
        return passed

    # -- convenience checkers ------------------------------------------
    def check_no_nan_inf(self, name: str, array, fatal: bool = True) -> bool:
        """Assert an array contains no NaN and no infinity."""
        if not self.enabled:
            return True
        arr = np.asarray(array)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        ok = (n_nan == 0 and n_inf == 0)
        detail = "clean" if ok else f"{n_nan} NaN, {n_inf} inf"
        return self.check(f"no_nan_inf({name})", ok, detail, fatal=fatal)

    def check_shape(self, name: str, array, expected_shape,
                    fatal: bool = True) -> bool:
        """Assert an array has the expected shape."""
        if not self.enabled:
            return True
        actual = tuple(np.asarray(array).shape)
        expected = tuple(expected_shape)
        ok = actual == expected
        return self.check(f"shape({name})", ok,
                          f"{actual} (expected {expected})", fatal=fatal)

    def check_stochastic_matrix(self, name: str, matrix, axis: int = 1,
                                tol: float = 1e-6, fatal: bool = True) -> bool:
        """Assert rows (or columns) sum to 1 and all entries are non-negative."""
        if not self.enabled:
            return True
        mat = np.asarray(matrix, dtype=float)
        if mat.size == 0:
            return self.check(f"stochastic_matrix({name})", False,
                              "empty matrix", fatal=fatal)
        row_sums = mat.sum(axis=axis)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        nonneg = bool(np.all(mat >= -tol))
        ok = (max_dev <= tol) and nonneg
        return self.check(
            f"stochastic_matrix({name})", ok,
            f"rows sum to 1 (max dev {max_dev:.2e}), nonneg={nonneg}",
            fatal=fatal,
        )

    def check_in_bounds(self, name: str, values, low: float, high: float,
                        fatal: bool = True) -> bool:
        """Assert every value lies within [low, high]."""
        if not self.enabled:
            return True
        arr = np.asarray(values, dtype=float)
        lo = float(np.min(arr)) if arr.size else 0.0
        hi = float(np.max(arr)) if arr.size else 0.0
        ok = bool(lo >= low and hi <= high)
        return self.check(
            f"in_bounds({name})", ok,
            f"range [{lo:.4g}, {hi:.4g}] within [{low:.4g}, {high:.4g}]",
            fatal=fatal,
        )

    def check_finite(self, name: str, value, fatal: bool = True) -> bool:
        """Assert a scalar is finite (catches NaN/inf losses early)."""
        if not self.enabled:
            return True
        ok = bool(np.isfinite(value))
        return self.check(f"finite({name})", ok, f"value={value}", fatal=fatal)

    def check_positive(self, name: str, value, fatal: bool = True) -> bool:
        """Assert a scalar is strictly positive (e.g., a split size)."""
        if not self.enabled:
            return True
        ok = value > 0
        return self.check(f"positive({name})", ok, f"value={value}", fatal=fatal)

    def check_file(self, name: str, path, fatal: bool = True) -> bool:
        """Assert a path exists and is non-empty (confirms a write succeeded)."""
        if not self.enabled:
            return True
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        ok = exists and size > 0
        state = "exists" if exists else "MISSING"
        return self.check(f"file({name})", ok, f"{state}, {size} bytes",
                          fatal=fatal)

    # ------------------------------------------------------------------
    # Stats: quantitative diagnostics (no pass/fail), shown in the report.
    # ------------------------------------------------------------------
    def stat(self, name: str, value, expected: str = "", note: str = "") -> None:
        """Record a quantitative diagnostic value for console and report.

        Args:
            name: Metric name.
            value: Metric value.
            expected: Optional expected value/range, shown for context.
            note: Optional plain-language meaning, shown as the "meaning" column
                in the report so the metric is understandable to any reader.
        """
        if not self.enabled:
            return
        self._stats.append((name, value, expected, note))
        line = f"  {name:<28}: {self._fmt(value)}"
        if expected:
            line += f"   (expected {expected})"
        print(line, flush=True)

    # ------------------------------------------------------------------
    # Tables: titled multi-column tables (e.g. prior ranges vs sampled values).
    # ------------------------------------------------------------------
    def table(self, title: str, headers, rows, note: str = "") -> None:
        """Record a titled table for the report and print it to the console.

        Args:
            title: Section title (shown in the console and as a report heading).
            headers: Column header strings.
            rows: Iterable of row sequences, each the same length as headers.
            note: Optional one-line explanation shown beneath the table.
        """
        if not self.enabled:
            return
        headers = [str(h) for h in headers]
        rows = [[str(cell) for cell in row] for row in rows]
        self._tables.append((title, headers, rows, note))
        # Console: simple aligned plain-text table.
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        print(f"{SEEK} {title} {SEEK}", flush=True)
        print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
              flush=True)
        for row in rows:
            print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)),
                  flush=True)
        if note:
            print(f"  ({note})", flush=True)

    # ------------------------------------------------------------------
    # Figures: save an already-built matplotlib Figure as PNG (dump mode).
    # ------------------------------------------------------------------
    def save_figure(self, name: str, fig, caption: str = "", dpi: int = 200) -> Optional[Path]:
        """Save a matplotlib ``Figure`` as PNG under ``dump_dir/figures/``.

        Uses ``Figure.savefig``, which renders through the Agg canvas and so
        needs no display -- safe on a headless server. No-op unless dumping.

        Args:
            name: Figure name (also the PNG filename stem).
            fig: A matplotlib ``Figure`` to save.
            caption: Optional plain-language description, shown beneath the
                image in the report so the reader knows what the figure shows.
            dpi: Raster resolution; 200 gives a crisp on-screen figure without
                oversized files. Callers may raise it for print/publication.
        """
        if not self.dump or self.dump_dir is None:
            return None
        fig_dir = self.dump_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        png = fig_dir / f"{name}.png"
        fig.savefig(str(png), dpi=dpi, bbox_inches="tight")
        self._figures.append((name, f"figures/{name}.png", caption))
        print(f"  [dump] figure -> {png}", flush=True)
        return png

    # ------------------------------------------------------------------
    # Summary + report
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print the end-of-stage PASS/FAIL tally to the console."""
        if not self.enabled:
            return
        n_pass = sum(1 for _, p, _ in self._checks if p)
        n_fail = sum(1 for _, p, _ in self._checks if not p)
        print(SINK, flush=True)
        print(f"[{self.stage}] CHECKS: {n_pass} passed, {n_fail} failed",
              flush=True)
        if n_fail:
            for name, passed, detail in self._checks:
                if not passed:
                    suffix = f" - {detail}" if detail else ""
                    print(f"  FAILED: {name}{suffix}", flush=True)
        print(SOCK, flush=True)

    def write_report(self) -> Optional[Path]:
        """Write the Markdown diagnostic report (dump mode only)."""
        if not self.dump or self.dump_dir is None:
            return None

        n_pass = sum(1 for _, p, _ in self._checks if p)
        n_fail = sum(1 for _, p, _ in self._checks if not p)

        title = f"{self.stage} diagnostics"
        if self.run_label:
            title += f" - {self.run_label}"

        lines = [f"# {title}", f"Run: {self.timestamp}"]
        if self.run_note:
            lines.append(f"Note: {self.run_note}")
        lines.append("")

        # -- Checks table --------------------------------------------------
        lines.append(f"## Checks - {n_pass} passed, {n_fail} failed")
        lines.append("")
        lines.append("| check | result | detail |")
        lines.append("|---|---|---|")
        for name, passed, detail in self._checks:
            lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} "
                         f"| {detail or '-'} |")
        lines.append("")

        # -- Quantitative table (with a plain-language meaning column) -----
        if self._stats:
            lines.append("## Quantitative")
            lines.append("")
            lines.append("| metric | value | expected | meaning |")
            lines.append("|---|---|---|---|")
            for name, value, expected, note in self._stats:
                lines.append(f"| {name} | {self._fmt(value)} "
                             f"| {expected or '-'} | {note or '-'} |")
            lines.append("")

        # -- Custom tables (e.g. prior ranges vs sampled values) -----------
        for title, headers, rows, note in self._tables:
            lines.append(f"## {title}")
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            if note:
                lines.append(f"*{note}*")
                lines.append("")

        # -- Legend: explain the checks (and PASS/FAIL convention) ---------
        legend = self._legend_lines()
        if legend:
            lines.append("## What these mean")
            lines.append("")
            lines.append("A check **PASS** means the invariant held; a **FAIL** "
                         "aborts the run immediately with a located error message.")
            lines.append("")
            lines.extend(legend)
            lines.append("")

        # -- Figures with captions -----------------------------------------
        if self._figures:
            lines.append("## Figures")
            lines.append("")
            for name, rel, caption in self._figures:
                lines.append(f"![{name}]({rel})")
                lines.append("")
                if caption:
                    lines.append(f"*{caption}*")
                    lines.append("")

        report_path = self.dump_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [dump] report -> {report_path}", flush=True)
        return report_path

    def _legend_lines(self) -> list:
        """Build the 'What these mean' legend from the checks that ran.

        Standard checks are explained from ``_CHECK_GLOSSARY`` (keyed by the
        name prefix before "("); custom checks contribute their ``note`` if one
        was supplied. Returns an empty list when there is nothing to explain.
        """
        entries = []
        seen_kinds = []
        for name, _passed, _detail in self._checks:
            kind = name.split("(", 1)[0]
            if kind in _CHECK_GLOSSARY and kind not in seen_kinds:
                seen_kinds.append(kind)
                entries.append(f"- **{kind}** -- {_CHECK_GLOSSARY[kind]}")
        for name, note in self._check_notes.items():
            entries.append(f"- **{name}** -- {note}")
        return entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _prepare_dump_dir(self) -> None:
        """Create the dump directory, downgrading to console-only on problems."""
        if self.dump_dir is None:
            print("  [dump] WARNING: dump requested but no dump_dir given; "
                  "console diagnostics only", flush=True)
            self.dump = False
            return
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.dump_dir).free
            if free < self._MIN_FREE_BYTES:
                print(f"  [dump] WARNING: only {free / 1e6:.0f} MB free at "
                      f"{self.dump_dir}; skipping artifact dumps "
                      f"(console diagnostics still active)", flush=True)
                self.dump = False
            else:
                print(f"  [dump] artifacts -> {self.dump_dir}", flush=True)
        except OSError as exc:
            print(f"  [dump] WARNING: cannot prepare {self.dump_dir} ({exc}); "
                  f"console diagnostics only", flush=True)
            self.dump = False

    @staticmethod
    def _fmt(value) -> str:
        """Format a value compactly for console/report display."""
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, np.ndarray):
            return f"ndarray{tuple(value.shape)} {value.dtype}"
        if isinstance(value, (list, tuple)) and len(value) > 8:
            return f"{type(value).__name__}[{len(value)}]"
        return str(value)
