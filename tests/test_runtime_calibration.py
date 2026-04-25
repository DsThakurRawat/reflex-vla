"""Tests for src/reflex/runtime/calibration.py — Phase 1 auto-calibration Day 1.

Covers: HardwareFingerprint construction + matching + serialization,
MeasurementQuality + MeasurementContext invariants, CalibrationEntry
validation + age + staleness, CalibrationCache schema-v1 invariants
(first field, key shape), lookup with/without fingerprint guard,
load/save round-trip + atomic-write semantics, schema_version refusal
on out-of-range values.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from reflex.runtime.calibration import (
    DEFAULT_STALE_AFTER_DAYS,
    SCHEMA_VERSION,
    CalibrationCache,
    CalibrationEntry,
    HardwareFingerprint,
    MeasurementContext,
    MeasurementQuality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_fp(**overrides) -> HardwareFingerprint:
    defaults = dict(
        gpu_uuid="GPU-abc-123",
        gpu_name="NVIDIA A10G",
        driver_version_major=535,
        driver_version_minor=129,
        cuda_version_major=12,
        cuda_version_minor=2,
        kernel_release="6.1.0",
        cpu_count=8,
        ram_gb=32,
        reflex_version="0.5.0",
    )
    defaults.update(overrides)
    return HardwareFingerprint(**defaults)


def _mk_quality(**overrides) -> MeasurementQuality:
    defaults = dict(
        warmup_iters=10,
        measurement_iters=100,
        median_ms=42.0,
        p99_ms=51.0,
        n_outliers_dropped=2,
        quality_score=0.92,
    )
    defaults.update(overrides)
    return MeasurementQuality(**defaults)


def _mk_context() -> MeasurementContext:
    return MeasurementContext(
        ort_version="1.20.1",
        torch_version="2.5.1",
        numpy_version="1.26.4",
        onnx_version="1.16.0",
    )


def _mk_entry(**overrides) -> CalibrationEntry:
    defaults = dict(
        chunk_size=50,
        nfe=4,
        latency_compensation_ms=42.0,
        provider="TensorrtExecutionProvider",
        variant="fp16",
        measurement_quality=_mk_quality(),
        measurement_context=_mk_context(),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    )
    defaults.update(overrides)
    return CalibrationEntry(**defaults)


# ---------------------------------------------------------------------------
# HardwareFingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_matches_default_tolerance_ignores_kernel_release():
    fp1 = _mk_fp(kernel_release="6.1.0")
    fp2 = _mk_fp(kernel_release="6.1.5")
    assert fp1.matches(fp2)


def test_fingerprint_matches_default_tolerates_ram_rounding():
    fp1 = _mk_fp(ram_gb=32)
    fp2 = _mk_fp(ram_gb=33)
    assert fp1.matches(fp2)


def test_fingerprint_matches_default_rejects_ram_drift():
    fp1 = _mk_fp(ram_gb=32)
    fp2 = _mk_fp(ram_gb=64)
    assert not fp1.matches(fp2)


def test_fingerprint_strict_mode_requires_bitwise_equality():
    fp1 = _mk_fp(kernel_release="6.1.0")
    fp2 = _mk_fp(kernel_release="6.1.5")
    assert not fp1.matches(fp2, strict=True)


def test_fingerprint_matches_rejects_driver_major_drift():
    fp1 = _mk_fp(driver_version_major=535)
    fp2 = _mk_fp(driver_version_major=545)
    assert not fp1.matches(fp2)


def test_fingerprint_matches_rejects_gpu_uuid_change():
    fp1 = _mk_fp(gpu_uuid="GPU-abc-123")
    fp2 = _mk_fp(gpu_uuid="GPU-different")
    assert not fp1.matches(fp2)


def test_fingerprint_matches_rejects_reflex_version_drift():
    fp1 = _mk_fp(reflex_version="0.5.0")
    fp2 = _mk_fp(reflex_version="0.6.0")
    assert not fp1.matches(fp2)


def test_fingerprint_round_trip_via_dict():
    fp = _mk_fp()
    d = fp.to_dict()
    fp2 = HardwareFingerprint.from_dict(d)
    assert fp == fp2


def test_fingerprint_current_returns_a_valid_object():
    fp = HardwareFingerprint.current()
    assert isinstance(fp, HardwareFingerprint)
    # Sentinels populate when probes fail — never None / empty
    assert fp.gpu_uuid != ""
    assert fp.kernel_release != ""


# ---------------------------------------------------------------------------
# MeasurementQuality
# ---------------------------------------------------------------------------


def test_quality_rejects_score_below_zero():
    with pytest.raises(ValueError, match="quality_score"):
        MeasurementQuality(
            warmup_iters=10, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=-0.1,
        )


def test_quality_rejects_score_above_one():
    with pytest.raises(ValueError, match="quality_score"):
        MeasurementQuality(
            warmup_iters=10, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.5,
        )


def test_quality_accepts_boundary_scores():
    MeasurementQuality(warmup_iters=0, measurement_iters=1, median_ms=0.1,
                       p99_ms=0.1, n_outliers_dropped=0, quality_score=0.0)
    MeasurementQuality(warmup_iters=0, measurement_iters=1, median_ms=0.1,
                       p99_ms=0.1, n_outliers_dropped=0, quality_score=1.0)


def test_quality_rejects_zero_measurement_iters():
    with pytest.raises(ValueError, match="measurement_iters"):
        MeasurementQuality(
            warmup_iters=0, measurement_iters=0, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.0,
        )


def test_quality_rejects_negative_warmup_iters():
    with pytest.raises(ValueError, match="warmup_iters"):
        MeasurementQuality(
            warmup_iters=-1, measurement_iters=100, median_ms=42.0,
            p99_ms=51.0, n_outliers_dropped=0, quality_score=1.0,
        )


def test_quality_round_trip():
    q = _mk_quality()
    q2 = MeasurementQuality.from_dict(q.to_dict())
    assert q == q2


# ---------------------------------------------------------------------------
# MeasurementContext
# ---------------------------------------------------------------------------


def test_context_round_trip():
    c = _mk_context()
    c2 = MeasurementContext.from_dict(c.to_dict())
    assert c == c2


def test_context_current_returns_valid_object():
    c = MeasurementContext.current()
    assert isinstance(c, MeasurementContext)
    assert c.numpy_version != ""


# ---------------------------------------------------------------------------
# CalibrationEntry
# ---------------------------------------------------------------------------


def test_entry_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        _mk_entry(chunk_size=0)


def test_entry_rejects_nfe_out_of_range():
    with pytest.raises(ValueError, match="nfe"):
        _mk_entry(nfe=0)
    with pytest.raises(ValueError, match="nfe"):
        _mk_entry(nfe=51)


def test_entry_rejects_negative_latency_comp():
    with pytest.raises(ValueError, match="latency_compensation_ms"):
        _mk_entry(latency_compensation_ms=-1.0)


def test_entry_rejects_empty_provider():
    with pytest.raises(ValueError, match="provider"):
        _mk_entry(provider="")


def test_entry_rejects_empty_variant():
    with pytest.raises(ValueError, match="variant"):
        _mk_entry(variant="")


def test_entry_round_trip_via_dict():
    e = _mk_entry()
    e2 = CalibrationEntry.from_dict(e.to_dict())
    assert e == e2


def test_entry_age_seconds_recent():
    e = _mk_entry()
    assert e.age_seconds() < 60


def test_entry_age_seconds_unparseable_timestamp_returns_inf():
    e = _mk_entry(timestamp="not-an-iso-date")
    assert e.age_seconds() == float("inf")


def test_entry_is_stale_after_threshold():
    old_ts = (datetime.now(timezone.utc) - timedelta(days=31)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    e = _mk_entry(timestamp=old_ts)
    assert e.is_stale(max_age_days=DEFAULT_STALE_AFTER_DAYS)


def test_entry_is_not_stale_when_recent():
    e = _mk_entry()
    assert not e.is_stale()


# ---------------------------------------------------------------------------
# CalibrationCache — schema v1 invariants
# ---------------------------------------------------------------------------


def test_cache_to_dict_has_schema_version_field():
    cache = CalibrationCache(reflex_version="0.5.0")
    d = cache.to_dict()
    assert "schema_version" in d
    assert d["schema_version"] == SCHEMA_VERSION


def test_cache_make_key_format():
    assert CalibrationCache.make_key("franka", "abc123") == "franka::abc123"


def test_cache_make_key_rejects_empty_args():
    with pytest.raises(ValueError):
        CalibrationCache.make_key("", "abc123")
    with pytest.raises(ValueError):
        CalibrationCache.make_key("franka", "")


def test_cache_make_key_rejects_separator_in_args():
    with pytest.raises(ValueError, match="'::'"):
        CalibrationCache.make_key("fran::ka", "abc")
    with pytest.raises(ValueError, match="'::'"):
        CalibrationCache.make_key("franka", "abc::def")


def test_cache_record_then_lookup_roundtrip():
    cache = CalibrationCache(reflex_version="0.5.0")
    entry = _mk_entry()
    cache.record(embodiment="franka", model_hash="abc", entry=entry)
    out = cache.lookup(embodiment="franka", model_hash="abc")
    assert out == entry


def test_cache_lookup_returns_none_on_miss():
    cache = CalibrationCache(reflex_version="0.5.0")
    out = cache.lookup(embodiment="franka", model_hash="abc")
    assert out is None


def test_cache_lookup_with_fingerprint_guard():
    fp_a = _mk_fp(gpu_uuid="GPU-a")
    fp_b = _mk_fp(gpu_uuid="GPU-b")
    cache = CalibrationCache(
        reflex_version="0.5.0", hardware_fingerprint=fp_a,
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    assert cache.lookup(
        embodiment="franka", model_hash="abc",
        require_fingerprint=fp_a,
    ) is not None
    assert cache.lookup(
        embodiment="franka", model_hash="abc",
        require_fingerprint=fp_b,
    ) is None


def test_cache_record_updates_calibration_date():
    cache = CalibrationCache(reflex_version="0.5.0", calibration_date="")
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    assert cache.calibration_date != ""


def test_cache_is_stale_when_fingerprint_mismatches():
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=_mk_fp(gpu_uuid="GPU-a"),
    )
    different = _mk_fp(gpu_uuid="GPU-b")
    assert cache.is_stale(different)


def test_cache_is_stale_when_no_fingerprint():
    cache = CalibrationCache(reflex_version="0.5.0")
    assert cache.is_stale(_mk_fp())


def test_cache_is_not_stale_when_fingerprint_matches_and_recent():
    fp = _mk_fp()
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=fp,
    )
    assert not cache.is_stale(fp)


def test_cache_is_stale_when_old():
    fp = _mk_fp()
    old_date = (datetime.now(timezone.utc) - timedelta(days=31)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=old_date,
        hardware_fingerprint=fp,
    )
    assert cache.is_stale(fp)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    fp = _mk_fp()
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=fp,
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    cache.record(embodiment="so100", model_hash="def", entry=_mk_entry(chunk_size=30))

    path = tmp_path / "cache.json"
    cache.save(path)
    assert path.exists()
    loaded = CalibrationCache.load(path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.hardware_fingerprint == fp
    assert loaded.lookup(embodiment="franka", model_hash="abc").chunk_size == 50
    assert loaded.lookup(embodiment="so100", model_hash="def").chunk_size == 30


def test_save_writes_atomically(tmp_path):
    cache = CalibrationCache(reflex_version="0.5.0")
    path = tmp_path / "cache.json"
    cache.save(path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_load_or_empty_returns_fresh_cache_when_missing(tmp_path):
    cache = CalibrationCache.load_or_empty(tmp_path / "nonexistent.json")
    assert cache.schema_version == SCHEMA_VERSION
    assert cache.entries == {}


def test_load_or_empty_returns_loaded_cache_when_exists(tmp_path):
    path = tmp_path / "cache.json"
    cache = CalibrationCache(
        reflex_version="0.5.0",
        calibration_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        hardware_fingerprint=_mk_fp(),
    )
    cache.record(embodiment="franka", model_hash="abc", entry=_mk_entry())
    cache.save(path)
    loaded = CalibrationCache.load_or_empty(path)
    assert loaded.lookup(embodiment="franka", model_hash="abc") is not None


def test_load_refuses_future_schema_version(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION + 1,
        "reflex_version": "0.6.0",
        "entries": {},
    }))
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationCache.load(path)


def test_load_refuses_zero_schema_version(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "schema_version": 0,
        "entries": {},
    }))
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationCache.load(path)


def test_save_creates_parent_directory(tmp_path):
    nested = tmp_path / "deeply" / "nested" / "path"
    cache = CalibrationCache(reflex_version="0.5.0")
    cache.save(nested / "cache.json")
    assert (nested / "cache.json").exists()


# ---------------------------------------------------------------------------
# measure_latency_profile (Day 2)
# ---------------------------------------------------------------------------


from reflex.runtime.calibration import measure_latency_profile  # noqa: E402


def test_measure_rejects_zero_n_iters():
    with pytest.raises(ValueError, match="n_iters"):
        measure_latency_profile(lambda: None, n_iters=0)


def test_measure_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup_iters"):
        measure_latency_profile(lambda: None, warmup_iters=-1)


def test_measure_rejects_outlier_trim_at_or_above_half():
    with pytest.raises(ValueError, match="outlier_trim_frac"):
        measure_latency_profile(lambda: None, outlier_trim_frac=0.5)


def test_measure_rejects_negative_outlier_trim():
    with pytest.raises(ValueError, match="outlier_trim_frac"):
        measure_latency_profile(lambda: None, outlier_trim_frac=-0.01)


def test_measure_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        measure_latency_profile(42)  # type: ignore[arg-type]


def test_measure_calls_warmup_then_measurement():
    """Warmup forwards count toward total invocations but not measurements."""
    invocations = {"n": 0}

    def predict():
        invocations["n"] += 1

    quality = measure_latency_profile(
        predict, n_iters=10, warmup_iters=3, outlier_trim_frac=0.0,
    )
    assert invocations["n"] == 13  # 3 warmup + 10 measured
    assert quality.measurement_iters == 10
    assert quality.warmup_iters == 3


def test_measure_propagates_predict_exception():
    """Calibration must fail loud on a broken predict path — no silent
    averaging over crashes."""
    def predict():
        raise RuntimeError("policy crashed")

    with pytest.raises(RuntimeError, match="policy crashed"):
        measure_latency_profile(predict, n_iters=10, warmup_iters=0)


def test_measure_returns_positive_median_for_real_work():
    """A predict that does some work produces a positive median wall-clock."""
    import math

    def predict():
        # Cheap CPU-bound work to make the measurement non-trivial
        x = 0.0
        for i in range(10_000):
            x += math.sin(i)

    quality = measure_latency_profile(predict, n_iters=20, warmup_iters=2)
    assert quality.median_ms > 0
    assert quality.p99_ms >= quality.median_ms  # p99 is at least the median


def test_measure_outlier_count_matches_trim_fraction():
    """Trimming N=100 with frac=0.05 drops 2*5 = 10 outliers."""
    quality = measure_latency_profile(
        lambda: None, n_iters=100, warmup_iters=0, outlier_trim_frac=0.05,
    )
    assert quality.n_outliers_dropped == 10


def test_measure_outlier_count_zero_when_trim_zero():
    quality = measure_latency_profile(
        lambda: None, n_iters=100, warmup_iters=0, outlier_trim_frac=0.0,
    )
    assert quality.n_outliers_dropped == 0


def test_measure_quality_score_high_for_constant_work():
    """A perfectly stable workload should score near 1.0."""
    # Use a no-op so wall-clock variance is dominated by clock noise
    # (small) rather than real compute variance.
    quality = measure_latency_profile(
        lambda: None, n_iters=100, warmup_iters=10,
    )
    # Even a no-op has tiny perf_counter jitter; CV is usually well under
    # 0.10 → quality_score should be at the top.
    assert quality.quality_score >= 0.5, (
        f"no-op should produce stable measurements, got "
        f"quality_score={quality.quality_score}, median={quality.median_ms}"
    )


def test_measure_quality_score_zero_for_high_cv():
    """A workload with very high coefficient of variation should score 0.

    Force this by having predict alternate between cheap and expensive paths
    so the trimmed samples still have CV above the noisy threshold."""
    import time as _time
    state = {"i": 0}

    def predict():
        state["i"] += 1
        # 50 fast (no-op) + 50 slow (sleep 1ms) samples
        if state["i"] % 2 == 0:
            _time.sleep(0.005)  # 5ms
        # else: ~0ms

    quality = measure_latency_profile(
        predict, n_iters=100, warmup_iters=0, outlier_trim_frac=0.0,
    )
    # CV should be huge given the bimodal distribution — quality near 0
    assert quality.quality_score < 0.5, (
        f"bimodal workload should produce low quality_score, got "
        f"{quality.quality_score} (median={quality.median_ms}, p99={quality.p99_ms})"
    )


def test_measure_p99_at_or_above_median():
    """p99 should never be below the median by construction."""
    import math

    def predict():
        x = 0.0
        for i in range(5_000):
            x += math.cos(i)

    quality = measure_latency_profile(predict, n_iters=50, warmup_iters=2)
    assert quality.p99_ms >= quality.median_ms


def test_measure_default_args_runs_clean():
    """The 100-iter default with 10 warmup completes quickly + returns
    a sensible MeasurementQuality on a no-op predict."""
    quality = measure_latency_profile(lambda: None)
    assert isinstance(quality, MeasurementQuality)
    assert quality.measurement_iters == 100
    assert quality.warmup_iters == 10
    assert quality.median_ms >= 0
    assert 0.0 <= quality.quality_score <= 1.0


# ---------------------------------------------------------------------------
# GreedyResolver (Day 3)
# ---------------------------------------------------------------------------


from reflex.runtime.calibration import GreedyResolver, ResolverInputs  # noqa: E402


def _mk_inputs(**overrides) -> ResolverInputs:
    defaults = dict(
        available_variants=("fp16",),
        available_providers=(
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
        candidate_nfes=(10, 8, 4, 2, 1),
        hardware=_mk_fp(),
        embodiment="franka",
        chunk_size_default=50,
        control_frequency_hz=20.0,
        max_batch=1,
    )
    defaults.update(overrides)
    return ResolverInputs(**defaults)


def test_resolver_inputs_rejects_empty_variants():
    with pytest.raises(ValueError, match="available_variants"):
        _mk_inputs(available_variants=())


def test_resolver_inputs_rejects_empty_providers():
    with pytest.raises(ValueError, match="available_providers"):
        _mk_inputs(available_providers=())


def test_resolver_inputs_rejects_empty_candidate_nfes():
    with pytest.raises(ValueError, match="candidate_nfes"):
        _mk_inputs(candidate_nfes=())


def test_resolver_inputs_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size_default"):
        _mk_inputs(chunk_size_default=0)


def test_resolver_inputs_rejects_zero_control_hz():
    with pytest.raises(ValueError, match="control_frequency_hz"):
        _mk_inputs(control_frequency_hz=0.0)


def test_resolver_inputs_chunk_period_arithmetic():
    # 50 actions / 20 Hz = 2500ms = 2.5s chunk period
    inputs = _mk_inputs(chunk_size_default=50, control_frequency_hz=20.0)
    assert inputs.chunk_period_ms == pytest.approx(2500.0)


def test_resolver_inputs_expert_budget_subtracts_safety_margin():
    """Default safety margin is 30% — expert budget = 70% of chunk period."""
    inputs = _mk_inputs(chunk_size_default=50, control_frequency_hz=20.0)
    # 2500ms * 0.7 = 1750ms
    assert inputs.expert_budget_ms == pytest.approx(1750.0)


# Variant resolution


def test_resolver_picks_fp16_when_only_one_available():
    resolver = GreedyResolver(_mk_inputs(available_variants=("fp16",)))
    assert resolver.resolve_variant() == "fp16"


def test_resolver_prefers_fp8_on_hopper():
    """A100/A10 don't support fp8 — H100 does. Verify the gpu_name match
    correctly enables fp8 selection."""
    resolver = GreedyResolver(_mk_inputs(
        available_variants=("fp8", "int8", "fp16"),
        hardware=_mk_fp(gpu_name="NVIDIA H100"),
    ))
    assert resolver.resolve_variant() == "fp8"


def test_resolver_skips_fp8_on_a10g():
    """A10G is sm_86 — no fp8 hardware support. Should fall through to int8."""
    resolver = GreedyResolver(_mk_inputs(
        available_variants=("fp8", "int8", "fp16"),
        hardware=_mk_fp(gpu_name="NVIDIA A10G"),
    ))
    assert resolver.resolve_variant() == "int8"


def test_resolver_falls_back_to_fp16_when_int8_missing():
    resolver = GreedyResolver(_mk_inputs(
        available_variants=("fp16",),
        hardware=_mk_fp(gpu_name="NVIDIA A10G"),
    ))
    assert resolver.resolve_variant() == "fp16"


# Provider resolution


def test_resolver_picks_trt_for_fp16_batch1():
    resolver = GreedyResolver(_mk_inputs(max_batch=1))
    assert resolver.resolve_provider("fp16") == "TensorrtExecutionProvider"


def test_resolver_skips_trt_when_batch_above_one():
    """Per ADR 2026-04-14-disable-trt-when-batch-gt-1, TRT EP rebuilds
    engines per batch shape — disabled for batch > 1."""
    resolver = GreedyResolver(_mk_inputs(max_batch=4))
    assert resolver.resolve_provider("fp16") == "CUDAExecutionProvider"


def test_resolver_skips_trt_for_int8():
    """TRT EP requires pre-built int8 calibration cache; not shipped Phase 1."""
    resolver = GreedyResolver(_mk_inputs())
    assert resolver.resolve_provider("int8") == "CUDAExecutionProvider"


def test_resolver_falls_back_to_cpu_when_no_gpu():
    """CPU-only host: CUDAExecutionProvider not in available list."""
    resolver = GreedyResolver(_mk_inputs(
        available_providers=("CPUExecutionProvider",),
    ))
    assert resolver.resolve_provider("fp16") == "CPUExecutionProvider"


# NFE resolution — the falsifiable claim from the ADR


def test_resolver_picks_largest_fitting_nfe():
    """expert_budget = 1750ms, expert_step = 100ms → 17 fits, NFE=10 picked."""
    resolver = GreedyResolver(_mk_inputs())  # default budget 1750ms
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=100.0,
    )
    assert nfe == 10


def test_resolver_falls_back_to_smaller_nfe_when_step_expensive():
    """Step is 200ms; budget 1750ms → only NFE up to 8 fits (8 × 200 = 1600).
    NFE=10 (10 × 200 = 2000 > 1750) excluded."""
    resolver = GreedyResolver(_mk_inputs())
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=200.0,
    )
    assert nfe == 8


def test_resolver_falls_back_to_nfe_one_on_a10g_franka_pi05_teacher():
    """The falsifiable claim from ADR 2026-04-25-auto-calibration-architecture:
    A10G × franka (50 chunk @ 20 Hz) × NFE=10 with pi0.5 teacher (~400ms
    per expert step on A10G) has no legal solution. Resolver falls back to
    NFE=1 (forces SnapFlow path)."""
    inputs = _mk_inputs(
        embodiment="franka", chunk_size_default=50, control_frequency_hz=20.0,
        hardware=_mk_fp(gpu_name="NVIDIA A10G"),
    )
    resolver = GreedyResolver(inputs)
    # 400ms per step × 1 NFE = 400ms; budget = 1750ms → fits at NFE=4 (1600ms)
    # but NFE=8 (3200ms) and NFE=10 (4000ms) don't.
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=400.0,
    )
    assert nfe == 4  # largest fitting

    # With a more expensive teacher (1000ms per step), only NFE=1 fits
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=1000.0,
    )
    assert nfe == 1


def test_resolver_handles_negative_step_defensively():
    """Invalid measurement → fall back to smallest NFE without crashing."""
    resolver = GreedyResolver(_mk_inputs())
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=-1.0,
    )
    assert nfe == 1  # smallest of (10, 8, 4, 2, 1)


def test_resolver_returns_smallest_when_no_nfe_fits():
    """Even NFE=1 exceeds the budget — return smallest with a warning."""
    resolver = GreedyResolver(_mk_inputs())
    # Budget = 1750ms; even 1 × 5000ms doesn't fit
    nfe = resolver.resolve_nfe(
        variant="fp16", provider="CUDAExecutionProvider",
        expert_denoise_ms_per_step=5000.0,
    )
    assert nfe == 1


# Chunk size resolution


def test_resolver_returns_default_chunk_size_when_nfe_fits():
    resolver = GreedyResolver(_mk_inputs(chunk_size_default=50))
    chunk_size = resolver.resolve_chunk_size(
        nfe=4, expert_denoise_ms_per_step=100.0,
    )
    assert chunk_size == 50


def test_resolver_shrinks_chunk_size_when_nfe_too_expensive():
    """Default chunk_size = 50 → budget 1750ms. NFE=10 × 200ms = 2000ms doesn't
    fit at chunk=50 — but the resolver only shrinks ONCE (50 → 25). At chunk=25
    the budget is 875ms — still doesn't fit. Resolver returns 25 with a warning."""
    resolver = GreedyResolver(_mk_inputs(chunk_size_default=50))
    chunk_size = resolver.resolve_chunk_size(
        nfe=10, expert_denoise_ms_per_step=200.0,
    )
    assert chunk_size <= 50  # at minimum 25


# Latency compensation cold-start


def test_resolver_latency_compensation_franka():
    resolver = GreedyResolver(_mk_inputs(embodiment="franka"))
    assert resolver.resolve_latency_compensation_ms() == 40.0


def test_resolver_latency_compensation_so100():
    resolver = GreedyResolver(_mk_inputs(embodiment="so100"))
    assert resolver.resolve_latency_compensation_ms() == 60.0


def test_resolver_latency_compensation_unknown_embodiment_uses_default():
    resolver = GreedyResolver(_mk_inputs(embodiment="custom_robot_xyz"))
    assert resolver.resolve_latency_compensation_ms() == 40.0  # global default
