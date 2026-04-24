"""Tests for `reflex doctor` deploy diagnostics (B.4 / D.1 Day 1).

Each check has a falsifiability gate (per the plan): at least 1 pass case
+ 1 fail case. CheckResult.__post_init__ enforces "fail must have remediation"
— exercised here too.

Tests are scoped to Day 1's 5 checks. Day 2 checks (action_denorm, gripper,
state_proprio, gpu_memory, hardware_compat) get their own test files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflex.diagnostics import (
    Check,
    CheckResult,
    exit_code,
    format_human,
    format_json,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# CheckResult invariants
# ---------------------------------------------------------------------------


class TestCheckResultInvariants:
    def test_fail_without_remediation_raises(self):
        """The falsifiability gate: every fail must have non-empty remediation."""
        with pytest.raises(ValueError, match="empty remediation"):
            CheckResult(
                check_id="x", name="X", status="fail",
                expected="something", actual="nothing", remediation="",
                duration_ms=0.0,
            )

    def test_pass_without_remediation_ok(self):
        # Passes don't need a remediation
        r = CheckResult(
            check_id="x", name="X", status="pass",
            expected="ok", actual="ok", remediation="",
            duration_ms=0.0,
        )
        assert r.status == "pass"

    def test_to_dict_round_trip(self):
        r = CheckResult(
            check_id="x", name="X", status="warn",
            expected="A", actual="B", remediation="do C",
            duration_ms=12.34, github_issue="https://example/1",
        )
        d = r.to_dict()
        assert d["check_id"] == "x"
        assert d["status"] == "warn"
        assert d["duration_ms"] == 12.34


# ---------------------------------------------------------------------------
# check_model_load
# ---------------------------------------------------------------------------


class TestCheckModelLoad:
    def test_fail_when_path_missing(self, tmp_path):
        """Fail case: nonexistent path."""
        results = run_all_checks(str(tmp_path / "nope"), "custom")
        load = next(r for r in results if r.check_id == "check_model_load")
        assert load.status == "fail"
        assert "does not exist" in load.actual.lower()
        assert load.remediation  # falsifiability gate

    def test_fail_when_path_is_file(self, tmp_path):
        f = tmp_path / "model.onnx"
        f.write_bytes(b"\x00")
        results = run_all_checks(str(f), "custom")
        load = next(r for r in results if r.check_id == "check_model_load")
        assert load.status == "fail"
        assert "directory" in load.actual.lower()

    def test_fail_when_no_onnx_files(self, tmp_path):
        # Empty directory
        results = run_all_checks(str(tmp_path), "custom")
        load = next(r for r in results if r.check_id == "check_model_load")
        assert load.status == "fail"
        assert "0 .onnx files" in load.actual

    def test_pass_when_small_onnx_present(self, tmp_path):
        """Pass case: tiny ONNX file fits in available RAM trivially."""
        (tmp_path / "model.onnx").write_bytes(b"\x00" * 1024)  # 1KB
        results = run_all_checks(str(tmp_path), "custom")
        load = next(r for r in results if r.check_id == "check_model_load")
        # Pass OR warn (warn if psutil missing). Should NOT fail.
        assert load.status in ("pass", "warn")


# ---------------------------------------------------------------------------
# check_onnx_provider
# ---------------------------------------------------------------------------


class TestCheckOnnxProvider:
    def test_runs_without_crashing(self, tmp_path):
        """Provider check doesn't depend on model — runs always."""
        results = run_all_checks(str(tmp_path), "custom")
        prov = next(r for r in results if r.check_id == "check_onnx_provider")
        # On dev macs, expect warn (CPU-only). On CI with GPU runners, expect pass.
        # On systems with no ORT, expect fail.
        assert prov.status in ("pass", "warn", "fail")
        if prov.status in ("warn", "fail"):
            assert prov.remediation


# ---------------------------------------------------------------------------
# check_image_dims
# ---------------------------------------------------------------------------


class TestCheckImageDims:
    def test_skip_on_custom_embodiment(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        results = run_all_checks(str(tmp_path), "custom")
        check = next(r for r in results if r.check_id == "check_image_dims")
        assert check.status == "skip"

    def test_skip_on_unknown_embodiment(self, tmp_path):
        """Unknown preset → check_image_dims fails (preset load fails);
        downstream checks still run."""
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        results = run_all_checks(str(tmp_path), "nonsense-bot")
        check = next(r for r in results if r.check_id == "check_image_dims")
        # The embodiment-load failure surfaces here as fail
        assert check.status == "fail"
        assert check.remediation

    def test_skip_when_no_onnx_in_dir(self, tmp_path):
        results = run_all_checks(str(tmp_path), "franka")
        check = next(r for r in results if r.check_id == "check_image_dims")
        # Skipped because the model dir has no ONNX (caught by check_model_load)
        assert check.status == "skip"


# ---------------------------------------------------------------------------
# check_rtc_chunks
# ---------------------------------------------------------------------------


class TestCheckRtcChunks:
    def test_skip_when_rtc_disabled(self, tmp_path):
        results = run_all_checks(str(tmp_path), "franka", rtc=False)
        check = next(r for r in results if r.check_id == "check_rtc_chunks")
        assert check.status == "skip"

    def test_pass_with_franka_preset_under_rtc(self, tmp_path):
        """Franka preset has frequency_hz=20, horizon=0.5 → 10 actions/horizon,
        chunk_size=50 — clean multiple, should pass."""
        results = run_all_checks(str(tmp_path), "franka", rtc=True)
        check = next(r for r in results if r.check_id == "check_rtc_chunks")
        assert check.status == "pass"

    def test_warn_with_custom_embodiment_under_rtc(self, tmp_path):
        results = run_all_checks(str(tmp_path), "custom", rtc=True)
        check = next(r for r in results if r.check_id == "check_rtc_chunks")
        assert check.status == "warn"  # custom = no preset to cross-check

    def test_fail_when_unknown_embodiment_under_rtc(self, tmp_path):
        results = run_all_checks(str(tmp_path), "nonsense-bot", rtc=True)
        check = next(r for r in results if r.check_id == "check_rtc_chunks")
        assert check.status == "fail"
        assert check.remediation


# ---------------------------------------------------------------------------
# check_vlm_tokenization
# ---------------------------------------------------------------------------


class TestCheckVlmTokenization:
    def test_skip_when_no_export_dir(self, tmp_path):
        results = run_all_checks(str(tmp_path / "missing"), "custom")
        check = next(r for r in results if r.check_id == "check_vlm_tokenization")
        assert check.status == "skip"

    def test_skip_when_no_tokenizer_config(self, tmp_path):
        """Most monolithic exports bake the tokenizer in; no separate config."""
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        results = run_all_checks(str(tmp_path), "custom")
        check = next(r for r in results if r.check_id == "check_vlm_tokenization")
        assert check.status == "skip"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def _sample_results(self):
        return [
            CheckResult(
                check_id="a", name="A", status="pass",
                expected="ok", actual="ok", remediation="",
                duration_ms=1.0,
            ),
            CheckResult(
                check_id="b", name="B", status="fail",
                expected="ok", actual="not ok", remediation="fix it",
                duration_ms=2.0, github_issue="https://example/1",
            ),
            CheckResult(
                check_id="c", name="C", status="skip",
                expected="ok", actual="", remediation="",
                duration_ms=0.0,
            ),
        ]

    def test_format_human_includes_summary_line(self):
        out = format_human(self._sample_results())
        assert "1 pass" in out
        assert "1 fail" in out
        assert "1 skip" in out

    def test_format_human_puts_failures_first(self):
        out = format_human(self._sample_results())
        # The fail should appear before the pass in the output
        fail_idx = out.find("[✗]")
        pass_idx = out.find("[✓]")
        assert 0 <= fail_idx < pass_idx

    def test_format_json_validates_envelope(self):
        out = format_json(
            self._sample_results(),
            model_path="/test", embodiment_name="franka",
        )
        env = json.loads(out)
        assert env["schema_version"] == 1
        assert env["model"] == "/test"
        assert env["embodiment"] == "franka"
        assert env["summary"]["pass"] == 1
        assert env["summary"]["fail"] == 1
        assert env["summary"]["skip"] == 1
        assert len(env["checks"]) == 3


# ---------------------------------------------------------------------------
# Exit code mapping
# ---------------------------------------------------------------------------


class TestExitCode:
    def test_all_pass_returns_zero(self):
        results = [CheckResult(
            check_id="a", name="A", status="pass",
            expected="", actual="", remediation="",
            duration_ms=0.0,
        )]
        assert exit_code(results) == 0

    def test_any_fail_returns_one(self):
        results = [
            CheckResult(check_id="a", name="A", status="pass",
                        expected="", actual="", remediation="", duration_ms=0.0),
            CheckResult(check_id="b", name="B", status="fail",
                        expected="x", actual="y", remediation="z", duration_ms=0.0),
        ]
        assert exit_code(results) == 1

    def test_warn_does_not_fail(self):
        results = [CheckResult(
            check_id="a", name="A", status="warn",
            expected="x", actual="y", remediation="z", duration_ms=0.0,
        )]
        assert exit_code(results) == 0

    def test_skip_does_not_fail(self):
        results = [CheckResult(
            check_id="a", name="A", status="skip",
            expected="", actual="", remediation="", duration_ms=0.0,
        )]
        assert exit_code(results) == 0


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_runs_all_5_day1_checks(self, tmp_path):
        results = run_all_checks(str(tmp_path), "custom", rtc=False)
        assert len(results) == 5
        ids = {r.check_id for r in results}
        assert ids == {
            "check_model_load",
            "check_onnx_provider",
            "check_vlm_tokenization",
            "check_image_dims",
            "check_rtc_chunks",
        }

    @pytest.mark.parametrize("emb", ["franka", "so100", "ur5"])
    def test_runs_against_each_preset(self, emb, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        results = run_all_checks(str(tmp_path), emb, rtc=True)
        assert len(results) == 5
        for r in results:
            assert r.status in {"pass", "fail", "warn", "skip"}
            if r.status == "fail":
                assert r.remediation, f"{r.check_id} fail with empty remediation"

    def test_skip_arg_works(self, tmp_path):
        results = run_all_checks(
            str(tmp_path), "franka",
            skip=["check_image_dims", "check_rtc_chunks"],
        )
        skipped = [r for r in results if r.check_id in (
            "check_image_dims", "check_rtc_chunks"
        )]
        assert all(r.status == "skip" for r in skipped)
        assert all("--skip" in r.expected for r in skipped)
