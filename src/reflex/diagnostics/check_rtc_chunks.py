"""Check 9 — RTC chunk-boundary alignment (LeRobot #2356, #2531).

THE HEADLINE check — directly maps to the highest-signal LeRobot async
issues with zero maintainer response. Validates that RTC's replan_hz /
execute_hz / chunk_size triple lines up so chunk boundaries don't stall
between replan ticks.

Skips silently when --rtc not set (RTC is opt-in).
"""
from __future__ import annotations

from . import Check, CheckResult, register

CHECK_ID = "check_rtc_chunks"
GH_ISSUE = "https://github.com/huggingface/lerobot/issues/2356"


def _run(embodiment_name: str = "custom", rtc: bool = False, **kwargs) -> CheckResult:
    if not rtc:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="skip",
            expected="--rtc passed for chunk-boundary validation",
            actual="RTC disabled",
            remediation="",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    if embodiment_name == "custom":
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="warn",
            expected="--embodiment <preset> for control rate cross-check",
            actual="embodiment=custom — using lerobot RTC defaults (replan_hz=20, execute_hz=100)",
            remediation=(
                "Pass --embodiment franka|so100|ur5 to validate the chunk_size against "
                "this preset's control rates. Skipping cross-check."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    try:
        from reflex.embodiments import EmbodimentConfig
        cfg = EmbodimentConfig.load_preset(embodiment_name)
    except (ValueError, FileNotFoundError) as e:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected=f"embodiment preset {embodiment_name!r} loads",
            actual=f"load failed: {e}",
            remediation="See docs/embodiment_schema.md for the preset list.",
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    control = cfg.control
    chunk_size = int(control["chunk_size"])
    frequency_hz = float(control["frequency_hz"])  # robot control loop rate
    horizon_s = float(control["rtc_execution_horizon"])

    # actions executed per RTC horizon
    actions_per_horizon = frequency_hz * horizon_s

    # Sanity check 1: horizon must consume at least one action
    if actions_per_horizon < 1:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected="frequency_hz × rtc_execution_horizon ≥ 1 action",
            actual=(
                f"{frequency_hz}Hz × {horizon_s}s = {actions_per_horizon:.2f} actions/horizon"
            ),
            remediation=(
                f"Increase rtc_execution_horizon (currently {horizon_s}s) OR "
                f"frequency_hz (currently {frequency_hz}Hz) so the product ≥ 1. "
                f"With 0 actions per horizon, RTC degenerates to no-RTC."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Sanity check 2: chunk must hold at least one horizon-worth of actions
    if chunk_size < actions_per_horizon:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="fail",
            expected=f"chunk_size ≥ {actions_per_horizon:.0f} actions (one horizon)",
            actual=f"chunk_size={chunk_size} < {actions_per_horizon:.2f} actions/horizon",
            remediation=(
                f"Either increase chunk_size to ≥ {int(actions_per_horizon) + 1} OR "
                f"reduce rtc_execution_horizon. Mismatch causes the boundary stalls "
                f"reported in LeRobot #2356."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    # Sanity check 3 (warn): chunk_size should be a clean multiple of horizon
    # so multiple inferences per chunk align with RTC ticks
    if actions_per_horizon >= 1 and chunk_size % int(actions_per_horizon) != 0:
        return CheckResult(
            check_id=CHECK_ID,
            name="RTC chunk boundary",
            status="warn",
            expected=f"chunk_size ({chunk_size}) is a multiple of actions_per_horizon ({int(actions_per_horizon)})",
            actual=f"chunk_size % horizon_actions = {chunk_size % int(actions_per_horizon)}",
            remediation=(
                f"Non-integer ratio means the last partial-horizon at chunk boundary "
                f"will run with stale guidance. Consider chunk_size="
                f"{int(actions_per_horizon) * (chunk_size // int(actions_per_horizon) + 1)} "
                f"for cleaner alignment. Not a hard failure."
            ),
            duration_ms=0.0,
            github_issue=GH_ISSUE,
        )

    return CheckResult(
        check_id=CHECK_ID,
        name="RTC chunk boundary",
        status="pass",
        expected="chunk_size, frequency_hz, rtc_execution_horizon align cleanly",
        actual=(
            f"chunk_size={chunk_size}, frequency_hz={frequency_hz}, "
            f"horizon={horizon_s}s ({actions_per_horizon:.0f} actions/horizon)"
        ),
        remediation="",
        duration_ms=0.0,
        github_issue=GH_ISSUE,
    )


register(Check(
    check_id=CHECK_ID,
    name="RTC chunk boundary",
    severity="error",
    github_issue=GH_ISSUE,
    run_fn=_run,
))
