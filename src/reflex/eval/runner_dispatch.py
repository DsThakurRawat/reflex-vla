"""Task-runner resolver for `reflex eval`.

Per ADR 2026-04-25-eval-as-a-service-architecture decisions #2 + #8:
- Wrap, not rebuild — production callers route through the existing
  Modal image + osmesa/MuJoCo recipe (lifted from scripts/modal_libero_*.py)
  + the 441-LOC PredictModelServer adapter at
  src/reflex/runtime/adapters/vla_eval.py
- Local fallback is Linux-only (osmesa + MuJoCo + lerobot dep stack);
  NEVER silently falls back to Modal — avoids surprise bills + masks
  real env config issues

Day 3 (this commit) ships the resolver scaffold with stub runners that
emit a structured adapter_error EpisodeResult with a clear deferral
message. This keeps the CLI invokable end-to-end (validate, preflight,
banner, dispatch) while leaving the wire-to-real-runner as Day 4
(Modal subprocess wrapper) + Day 5 (local OffScreenRenderEnv runner).

The pattern is honest: stub runners produce LOUD adapter_error rows,
not silent zeros. Customers running `reflex eval --runtime modal` on
Day 3 see a real failure (not a fake success rate) and the CLI exits
non-zero.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from reflex.eval.libero import (
    ALL_RUNTIMES,
    EpisodeResult,
    LiberoSuiteConfig,
    TaskRunner,
)

logger = logging.getLogger(__name__)


def resolve_task_runner(
    *,
    runtime: str,
    export_dir: Path,
) -> TaskRunner:
    """Return a TaskRunner callable bound to the requested runtime.

    Day 3 stub: both runtimes return a runner that emits adapter_error
    episodes with a deferral message. Day 4 wires the Modal subprocess
    wrapper; Day 5 wires the local OffScreenRenderEnv runner.

    Raises:
        ValueError: runtime not in ALL_RUNTIMES.
    """
    if runtime not in ALL_RUNTIMES:
        raise ValueError(
            f"runtime must be one of {ALL_RUNTIMES}, got {runtime!r}"
        )

    if runtime == "modal":
        return _make_modal_stub_runner(export_dir)
    # runtime == "local"
    return _make_local_stub_runner(export_dir)


def _make_modal_stub_runner(export_dir: Path) -> TaskRunner:
    """Day 3 stub for --runtime modal. Day 4 replaces this with a real
    subprocess wrapper around scripts/modal_libero_*.py."""
    msg = (
        "Modal task runner not yet wired (ships Day 4 — see "
        "features/01_serve/subfeatures/_dx_gaps/eval-as-a-service/"
        "eval-as-a-service_plan.md). For now use --cost-preview to "
        "estimate cost without invoking a real run."
    )

    def _runner(task_id: str, episode_index: int, config: LiberoSuiteConfig) -> EpisodeResult:
        return EpisodeResult(
            task_id=task_id,
            episode_index=episode_index,
            success=False,
            terminal_reason="adapter_error",
            wall_clock_s=0.0,
            n_steps=0,
            video_path=None,
            error_message=msg,
        )

    return _runner


def _make_local_stub_runner(export_dir: Path) -> TaskRunner:
    """Day 3 stub for --runtime local. Day 5 replaces this with a real
    OffScreenRenderEnv runner (Linux-only — fails loud on macOS)."""
    msg = (
        "Local task runner not yet wired (ships Day 5 — see "
        "features/01_serve/subfeatures/_dx_gaps/eval-as-a-service/"
        "eval-as-a-service_plan.md). For now use --runtime modal once "
        "Day 4 wires the Modal subprocess wrapper."
    )

    def _runner(task_id: str, episode_index: int, config: LiberoSuiteConfig) -> EpisodeResult:
        return EpisodeResult(
            task_id=task_id,
            episode_index=episode_index,
            success=False,
            terminal_reason="adapter_error",
            wall_clock_s=0.0,
            n_steps=0,
            video_path=None,
            error_message=msg,
        )

    return _runner


# Phase 1 ships LIBERO-90 task list (per ADR decision #5). Lifts the
# canonical task identifiers from the LIBERO repo. Day 3 substrate uses
# a small representative subset; Day 4-5 wires the full 90.
LIBERO_DEFAULT_TASKS_PHASE1: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def default_libero_tasks() -> list[str]:
    """Default LIBERO task list when --tasks not specified.

    Day 3 ships the 4-suite top-level list; Day 4-5 swaps to the full
    LIBERO-90 task identifiers once the Modal task-runner can target
    them per-task.
    """
    return list(LIBERO_DEFAULT_TASKS_PHASE1)


__all__ = [
    "LIBERO_DEFAULT_TASKS_PHASE1",
    "default_libero_tasks",
    "resolve_task_runner",
]
