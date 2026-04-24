"""ORT CUDA graphs wrapper for Reflex serve (Phase 1 cuda-graphs feature).

Per ADR 2026-04-24-cuda-graphs-architecture:
  - ORT-native CUDA graphs (NOT torch.cuda.graph)
  - Two separate captured graphs per model (vlm_prefix + expert_denoise)
  - One shape per (model × embodiment) pair — ONNX is shape-specialized at export
  - Opt-in customer flag for Phase 1; flip to always-on in Phase 2 after telemetry

Research sidecar:
  features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_research.md

Usage:

    from reflex.runtime.cuda_graphs import build_cuda_graph_providers, CudaGraphWrapper

    providers = build_cuda_graph_providers(enabled=cuda_graphs_enabled)
    raw = ort.InferenceSession(path, providers=providers)
    wrapped = CudaGraphWrapper(raw, session_name="vlm_prefix",
                               embodiment=embodiment, model_id=model_id)
    result = wrapped.run(output_names, feed)  # first call captures, subsequent replay
    wrapped.invalidate()  # on model swap — caller must build a new session

The wrapper does NOT own session construction. Caller is responsible for
passing a session built with build_cuda_graph_providers(enabled=True). On
capture/replay failure, the wrapper increments the eager-fallback counter
and re-raises — callers are expected to catch the exception and route the
request to an eager fallback path (or surface the error, per their policy).
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from reflex.observability.prometheus import (
    inc_cuda_graph_captured,
    inc_cuda_graph_eager_fallback,
    inc_cuda_graph_replayed,
    observe_cuda_graph_capture_seconds,
    observe_cuda_graph_replay_seconds,
)

if TYPE_CHECKING:
    import onnxruntime as ort  # pragma: no cover

logger = logging.getLogger(__name__)


# Bounded enum of session names — matches the Prometheus label vocabulary.
VALID_SESSION_NAMES = frozenset({"vlm_prefix", "expert_denoise"})


def build_cuda_graph_providers(enabled: bool) -> list:
    """Return an ONNX Runtime providers list with CUDA graphs configured.

    When enabled=True: CUDAExecutionProvider captures on first .run() and
    replays thereafter (session-level graph capture).
    When enabled=False: eager CUDA execution with no graph capture.

    The list includes CPUExecutionProvider as a final fallback per the
    existing Reflex convention (src/reflex/runtime/pi05_decomposed_server.py
    session constructors).
    """
    cuda_opts: dict[str, str] = {}
    if enabled:
        cuda_opts["enable_cuda_graph"] = "1"
    return [
        ("CUDAExecutionProvider", cuda_opts),
        "CPUExecutionProvider",
    ]


class CudaGraphWrapper:
    """Wraps an ORT session with capture/replay metric emission + fallback tracking.

    First call to .run() triggers capture (ORT handles this internally because
    the session was built with enable_cuda_graph=1); wrapper records the
    capture time + increments captured_total counter. Subsequent calls replay;
    wrapper records replay time + increments replayed_total.

    On any exception during .run(), the wrapper increments the eager-fallback
    counter with a reason label and re-raises. Callers are expected to catch
    and route to an eager-path session (or surface the error).

    .invalidate() is a no-op on the session itself — ORT's graph is bound to
    the session lifecycle. Callers must re-create the session (with a fresh
    CudaGraphWrapper) to get a new capture.
    """

    __slots__ = (
        "_session",
        "_session_name",
        "_embodiment",
        "_model_id",
        "_captured",
    )

    def __init__(
        self,
        session: "ort.InferenceSession",
        session_name: str,
        embodiment: str,
        model_id: str,
    ):
        if session_name not in VALID_SESSION_NAMES:
            raise ValueError(
                f"session_name must be one of {sorted(VALID_SESSION_NAMES)}, got {session_name!r}"
            )
        self._session = session
        self._session_name = session_name
        self._embodiment = embodiment
        self._model_id = model_id
        self._captured = False  # flipped true after first successful .run()

    @property
    def session(self) -> "ort.InferenceSession":
        """Underlying ORT session. Exposed for callers that need to introspect
        inputs/outputs (e.g., building a feed dict)."""
        return self._session

    @property
    def captured(self) -> bool:
        """True if at least one successful .run() has occurred (graph is captured)."""
        return self._captured

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> list:
        """Forward to the ORT session with metric emission + fallback tracking.

        - First call (captured == False): records capture wall-clock into the
          capture histogram, increments captured_total on success.
        - Subsequent calls: records replay wall-clock into the replay histogram,
          increments replayed_total.
        - Any exception increments eager_fallback_total with a reason label
          and is re-raised.
        """
        if not self._captured:
            t0 = time.perf_counter()
            try:
                result = self._session.run(output_names, input_feed)
            except Exception as exc:
                inc_cuda_graph_eager_fallback(
                    embodiment=self._embodiment,
                    model_id=self._model_id,
                    reason="capture_failed",
                )
                logger.error(
                    "cuda_graph.capture_failed session=%s model=%s embodiment=%s exc=%s: %s",
                    self._session_name,
                    self._model_id,
                    self._embodiment,
                    type(exc).__name__,
                    exc,
                )
                raise
            elapsed = time.perf_counter() - t0
            observe_cuda_graph_capture_seconds(
                embodiment=self._embodiment,
                session=self._session_name,
                seconds=elapsed,
            )
            self._captured = True
            inc_cuda_graph_captured(
                embodiment=self._embodiment,
                model_id=self._model_id,
                session=self._session_name,
            )
            inc_cuda_graph_replayed(
                embodiment=self._embodiment,
                model_id=self._model_id,
                session=self._session_name,
            )
            logger.info(
                "cuda_graph.captured session=%s model=%s embodiment=%s elapsed_ms=%.1f",
                self._session_name,
                self._model_id,
                self._embodiment,
                elapsed * 1000,
            )
            return result

        # Replay path
        t0 = time.perf_counter()
        try:
            result = self._session.run(output_names, input_feed)
        except Exception as exc:
            inc_cuda_graph_eager_fallback(
                embodiment=self._embodiment,
                model_id=self._model_id,
                reason="replay_failed",
            )
            logger.error(
                "cuda_graph.replay_failed session=%s model=%s embodiment=%s exc=%s: %s",
                self._session_name,
                self._model_id,
                self._embodiment,
                type(exc).__name__,
                exc,
            )
            raise
        elapsed = time.perf_counter() - t0
        observe_cuda_graph_replay_seconds(
            embodiment=self._embodiment,
            session=self._session_name,
            seconds=elapsed,
        )
        inc_cuda_graph_replayed(
            embodiment=self._embodiment,
            model_id=self._model_id,
            session=self._session_name,
        )
        return result

    def invalidate(self) -> None:
        """Reset the captured flag so the next .run() is treated as a capture.

        Note: this does NOT rebuild the underlying ORT session. Callers are
        expected to construct a fresh session (e.g., after a model swap) and
        pass it to a new CudaGraphWrapper. This method exists mainly for test
        fixtures that want to simulate an invalidation event without rebuilding.
        """
        self._captured = False
        logger.info(
            "cuda_graph.invalidated session=%s model=%s embodiment=%s",
            self._session_name,
            self._model_id,
            self._embodiment,
        )
