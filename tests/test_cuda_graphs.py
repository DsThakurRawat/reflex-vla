"""Unit tests for src/reflex/runtime/cuda_graphs.py.

These tests use mock ORT sessions to exercise the wrapper's metric emission +
fallback logic without requiring a GPU. Integration tests covering the actual
ORT capture/replay behavior live in tests/test_cuda_graphs_integration.py and
are gated on CUDA availability.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflex.observability.prometheus import (
    REGISTRY,
    reflex_cuda_graph_captured_total,
    reflex_cuda_graph_eager_fallback_total,
    reflex_cuda_graph_replayed_total,
)
from reflex.runtime.cuda_graphs import (
    VALID_SESSION_NAMES,
    CudaGraphWrapper,
    build_cuda_graph_providers,
)


# ---------------------------------------------------------------------------
# build_cuda_graph_providers
# ---------------------------------------------------------------------------

def test_build_providers_enabled_sets_cuda_graph_flag():
    providers = build_cuda_graph_providers(enabled=True)
    cuda_entry = providers[0]
    assert cuda_entry[0] == "CUDAExecutionProvider"
    assert cuda_entry[1] == {"enable_cuda_graph": "1"}
    assert providers[-1] == "CPUExecutionProvider"


def test_build_providers_disabled_has_empty_cuda_opts():
    providers = build_cuda_graph_providers(enabled=False)
    cuda_entry = providers[0]
    assert cuda_entry[0] == "CUDAExecutionProvider"
    assert cuda_entry[1] == {}
    assert providers[-1] == "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# CudaGraphWrapper construction
# ---------------------------------------------------------------------------

def test_rejects_invalid_session_name():
    mock_sess = MagicMock()
    with pytest.raises(ValueError, match="session_name"):
        CudaGraphWrapper(mock_sess, session_name="bogus", embodiment="franka", model_id="m1")


def test_accepts_both_valid_session_names():
    mock_sess = MagicMock()
    for name in VALID_SESSION_NAMES:
        w = CudaGraphWrapper(mock_sess, session_name=name, embodiment="franka", model_id="m1")
        assert w.captured is False


# ---------------------------------------------------------------------------
# Capture + replay metric emission
# ---------------------------------------------------------------------------

def _get_counter(counter, labels: dict) -> float:
    """Fetch the current value of a Prometheus Counter for given labels."""
    return counter.labels(**labels)._value.get()


def test_first_run_increments_captured_and_replayed():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["output"]
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-1")

    cap_before = _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    )
    rep_before = _get_counter(
        reflex_cuda_graph_replayed_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    )

    result = w.run(None, {"x": [1, 2, 3]})

    assert result == ["output"]
    assert w.captured is True
    assert _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    ) == cap_before + 1
    assert _get_counter(
        reflex_cuda_graph_replayed_total,
        {"embodiment": "franka", "model_id": "cg-test-1", "session": "vlm_prefix"},
    ) == rep_before + 1


def test_subsequent_runs_only_increment_replayed():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["output"]
    w = CudaGraphWrapper(mock_sess, "expert_denoise", embodiment="so100", model_id="cg-test-2")

    w.run(None, {"x": [1]})  # first call (capture)

    cap_after_first = _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    )
    rep_after_first = _get_counter(
        reflex_cuda_graph_replayed_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    )

    for _ in range(5):
        w.run(None, {"x": [1]})

    # Captured stayed flat
    assert _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    ) == cap_after_first
    # Replayed incremented by 5
    assert _get_counter(
        reflex_cuda_graph_replayed_total,
        {"embodiment": "so100", "model_id": "cg-test-2", "session": "expert_denoise"},
    ) == rep_after_first + 5


# ---------------------------------------------------------------------------
# Fallback behavior on exception
# ---------------------------------------------------------------------------

def test_capture_exception_increments_fallback_and_reraises():
    class CudaCaptureError(RuntimeError):
        pass

    mock_sess = MagicMock()
    mock_sess.run.side_effect = CudaCaptureError("mock capture fail")
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-3")

    fb_before = _get_counter(
        reflex_cuda_graph_eager_fallback_total,
        {"embodiment": "franka", "model_id": "cg-test-3", "reason": "capture_failed"},
    )

    with pytest.raises(CudaCaptureError):
        w.run(None, {})

    assert w.captured is False  # never flipped
    assert _get_counter(
        reflex_cuda_graph_eager_fallback_total,
        {"embodiment": "franka", "model_id": "cg-test-3", "reason": "capture_failed"},
    ) == fb_before + 1


def test_replay_exception_increments_fallback_with_replay_reason():
    class CudaReplayError(RuntimeError):
        pass

    mock_sess = MagicMock()
    # First call succeeds (capture), second raises (replay)
    mock_sess.run.side_effect = [["ok"], CudaReplayError("mock replay fail")]

    w = CudaGraphWrapper(mock_sess, "expert_denoise", embodiment="ur5", model_id="cg-test-4")
    w.run(None, {})  # captures
    assert w.captured is True

    fb_before = _get_counter(
        reflex_cuda_graph_eager_fallback_total,
        {"embodiment": "ur5", "model_id": "cg-test-4", "reason": "replay_failed"},
    )

    with pytest.raises(CudaReplayError):
        w.run(None, {})

    assert _get_counter(
        reflex_cuda_graph_eager_fallback_total,
        {"embodiment": "ur5", "model_id": "cg-test-4", "reason": "replay_failed"},
    ) == fb_before + 1


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------

def test_invalidate_resets_captured_flag():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["out"]
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-5")

    w.run(None, {})
    assert w.captured is True

    w.invalidate()
    assert w.captured is False

    # Next run is treated as capture (increments captured counter)
    cap_before = _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-5", "session": "vlm_prefix"},
    )
    w.run(None, {})
    assert w.captured is True
    assert _get_counter(
        reflex_cuda_graph_captured_total,
        {"embodiment": "franka", "model_id": "cg-test-5", "session": "vlm_prefix"},
    ) == cap_before + 1


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------

def test_run_passes_output_names_and_feed_through():
    mock_sess = MagicMock()
    mock_sess.run.return_value = ["tensor"]

    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-6")
    feed = {"lang_tokens": [1, 2, 3], "state": [0.1, 0.2]}
    output_names = ["past_k_0", "past_v_0"]

    w.run(output_names, feed)
    mock_sess.run.assert_called_with(output_names, feed)


def test_session_property_exposes_raw_session():
    mock_sess = MagicMock()
    w = CudaGraphWrapper(mock_sess, "vlm_prefix", embodiment="franka", model_id="cg-test-7")
    assert w.session is mock_sess
