"""Pi05DecomposedServer -- wraps Pi05DecomposedInference behind the reflex
serve interface.

Per ADR 2026-04-25-decomposed-dispatch-via-reflex-serve: closes the
B.4/B.5 measurement gap. ReflexServer's create_app router falls through
to the legacy ReflexServer for `export_kind=decomposed` exports, which
looks for `expert_stack.onnx` and fails with "No ONNX model found".

This wrapper lets the router add a 4th branch:
    if reflex_config.export_kind == "decomposed":
        server = Pi05DecomposedServer(...)

The class exposes the ReflexServer interface (predict_from_base64_async,
run_batch, _action_guard, etc.) so all the FastAPI plumbing in
create_app composes without per-wedge updates.

Scope per the ADR:
- Phase 1 substrate: this file (load + dispatch + minimal prep pipeline)
- Modal validation: filed as follow-up experiments (see ADR §"Validation
  experiments to file")
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Default camera resolution for pi05 (matches the export's expected input).
# Customers running a non-default export should set this via the export's
# reflex_config.json `camera_resolution` field; we read it at load time.
DEFAULT_CAMERA_RESOLUTION = 224

# Default chunk size + max action dim for pi05 (matches the export). Read
# from reflex_config.json at load time; these are the conservative defaults
# when the config doesn't carry them.
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_ACTION_DIM = 32

# Lang sequence length default. The decomposed export traces with a fixed
# seq length (typically 16 or 48); the exact value comes from the ONNX
# session's lang_tokens input shape.
DEFAULT_LANG_SEQ_LEN = 16


class Pi05DecomposedServer:
    """Server-interface wrapper around Pi05DecomposedInference.

    Mirrors the ReflexServer attributes that create_app + the /act handler
    + the wedges (action_guard, RTC, A2C2, record-replay, prometheus,
    chunk-budget-batching, policy-versioning) read.

    Lifecycle:
        server = Pi05DecomposedServer(export_dir, ...)
        server.load()  # builds the underlying Pi05DecomposedInference
        result = await server.predict_from_base64_async(
            image_b64=..., instruction="pick up the cup", state=[...],
        )
    """

    def __init__(
        self,
        export_dir: str | Path,
        device: str = "cuda",
        providers: list[str] | None = None,
        strict_providers: bool = True,
        safety_config: str | Path | None = None,
        adaptive_steps: bool = False,
        cloud_fallback_url: str = "",
        deadline_ms: float | None = None,
        max_batch: int = 1,
        batch_timeout_ms: float = 5.0,
    ):
        self.export_dir = Path(export_dir)
        self.device = device
        self._requested_device = device
        self._requested_providers = providers
        self._strict_providers = strict_providers
        self._safety_config_path = (
            Path(safety_config) if safety_config else None
        )
        self._adaptive_steps = adaptive_steps
        self._cloud_fallback_url = cloud_fallback_url
        self._deadline_ms = deadline_ms
        self._max_batch = max(1, max_batch)
        self._batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0

        # ReflexServer-interface mirror attributes (populated during load
        # OR after lifespan wedge composition).
        self._ready = False
        self._inference: Any = None  # Pi05DecomposedInference
        self._tokenizer: Any = None  # HF tokenizer
        self._action_guard: Any = None
        self._split_orchestrator = None
        self._last_good_actions: np.ndarray | None = None
        self._deadline_misses = 0
        self._batches_run = 0
        self._batched_requests = 0
        self._latency_history: deque[float] = deque(maxlen=1024)
        self._model_hash: str | None = None
        self._config_hash: str | None = None

        # Loaded from reflex_config.json at .load() time.
        self.config: dict[str, Any] = {}
        self.action_dim: int = 7  # set from config.action_dim during load
        self.chunk_size: int = DEFAULT_CHUNK_SIZE
        self.max_action_dim: int = DEFAULT_MAX_ACTION_DIM
        self.camera_resolution: int = DEFAULT_CAMERA_RESOLUTION
        self.lang_seq_len: int = DEFAULT_LANG_SEQ_LEN

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def _inference_mode(self) -> str:
        """Mirrors ReflexServer._inference_mode for /act response building.
        Decomposed exports always use ORT; expose the actual provider."""
        if self._inference is None:
            return "uninitialized"
        # Pi05DecomposedInference doesn't expose providers directly; report
        # the requested one for now.
        if self._requested_device == "cuda":
            return "onnx_cuda_decomposed"
        return "onnx_cpu_decomposed"

    def _load_config(self) -> dict[str, Any]:
        """Read reflex_config.json from the export dir."""
        config_path = self.export_dir / "reflex_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"reflex_config.json not found in {self.export_dir}. "
                f"Decomposed exports must include the config sibling."
            )
        return json.loads(config_path.read_text())

    def load(self) -> None:
        """Load the decomposed inference + tokenizer + reflex_config."""
        # Lazy import keeps the heavy onnxruntime-gpu import out of module
        # scope (so importing this file doesn't pull torch/ort).
        from reflex.runtime.pi05_decomposed_server import (
            Pi05DecomposedInference,
        )

        self.config = self._load_config()

        # Pull canonical fields from config; validate export_kind.
        export_kind = self.config.get("export_kind", "")
        if export_kind != "decomposed":
            raise ValueError(
                f"Pi05DecomposedServer requires export_kind='decomposed', "
                f"got {export_kind!r}. Use the matching server class for "
                f"this export type (monolithic -> Pi0OnnxServer / "
                f"SmolVLAOnnxServer; legacy decomposed -> ReflexServer)."
            )

        self.action_dim = int(self.config.get("action_dim", 7))
        self.chunk_size = int(
            self.config.get("chunk_size") or self.config.get("action_chunk_size", DEFAULT_CHUNK_SIZE)
        )
        # max_action_dim is the export's padded action dim (usually 32 for pi05).
        # Read from `decomposed.max_action_dim` if present; default to 32.
        decomposed_block = self.config.get("decomposed", {})
        self.max_action_dim = int(
            decomposed_block.get("max_action_dim", DEFAULT_MAX_ACTION_DIM)
        )

        # Build providers list.
        providers = self._requested_providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._requested_device == "cuda"
            else ["CPUExecutionProvider"]
        )

        logger.info(
            "Pi05DecomposedServer loading from %s (action_dim=%d, "
            "chunk_size=%d, max_action_dim=%d, providers=%s)",
            self.export_dir, self.action_dim, self.chunk_size,
            self.max_action_dim, providers,
        )

        # Instantiate the inference primitive. Default cache config keeps
        # the production "episode" cache off until callers explicitly pass
        # episode_id; falls back to the safe "prefix" mode.
        self._inference = Pi05DecomposedInference(
            export_dir=self.export_dir,
            providers=providers,
            enable_cache=True,
            cache_level="prefix",  # safe default; episode mode requires per-call episode_id
        )

        # Probe the lang_tokens input shape from the ONNX session to
        # determine the expected sequence length for tokenization.
        # Pi05DecomposedInference exposes `_expert_session` after init.
        try:
            sess = getattr(self._inference, "_expert_session", None)
            if sess is not None:
                for inp in sess.get_inputs():
                    if inp.name == "lang_tokens":
                        shape = inp.shape
                        if len(shape) >= 2 and isinstance(shape[1], int):
                            self.lang_seq_len = int(shape[1])
                            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "lang_seq_len probe failed (%s); using default %d",
                exc, self.lang_seq_len,
            )

        # Try to load the HF tokenizer from the export. For decomposed
        # pi05 exports the tokenizer config typically lives alongside the
        # ONNX. Lazy-loaded on first /act call to avoid import cost when
        # serving without text instructions.
        self._tokenizer = None  # populated lazily by _ensure_tokenizer

        # Apply ActionGuard if a safety config was provided.
        if self._safety_config_path is not None:
            try:
                from reflex.safety import ActionGuard, SafetyLimits
                limits = SafetyLimits.from_json(self._safety_config_path)
                self._action_guard = ActionGuard(limits=limits, mode="clamp")
                logger.info(
                    "Pi05DecomposedServer ActionGuard loaded: %d joints, mode=clamp",
                    len(limits.joint_names),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pi05DecomposedServer safety load failed: %s", exc)

        self._ready = True
        logger.info("Pi05DecomposedServer ready")

    def _ensure_tokenizer(self) -> None:
        """Lazy-load the HF tokenizer from the export dir on first use.

        Fallback chain (per 2026-04-25 b4 gate v4 finding -- distill outputs
        don't ship a tokenizer + model_id is a local path):
        1. tokenizer.json sibling in the export dir
        2. config.model_id when it looks like an HF repo id (org/name)
        3. distill_provenance.json -> teacher_export when it's an HF id
        4. paligemma-3b-pt-224 (the canonical VLM tokenizer all pi05 students
           use; safe default since SnapFlow distill preserves the input
           tokenization contract from the teacher)
        """
        if self._tokenizer is not None:
            return

        from transformers import AutoTokenizer
        tried: list[str] = []
        last_exc: Exception | None = None

        candidates: list[str] = []

        # 1. Sibling tokenizer in the export
        if (self.export_dir / "tokenizer.json").exists() or (
            self.export_dir / "tokenizer_config.json"
        ).exists():
            candidates.append(str(self.export_dir))

        # 2. config.model_id if it looks like an HF repo id (org/name) and
        # NOT a local path
        model_id = self.config.get("model_id", "")
        if (
            isinstance(model_id, str)
            and "/" in model_id
            and not model_id.startswith("/")
            and "::" not in model_id
        ):
            candidates.append(model_id)

        # 3. distill_provenance.json's teacher_export, if present + HF id
        prov_path = self.export_dir / "distill_provenance.json"
        if prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                teacher = prov.get("teacher_export", "")
                if (
                    isinstance(teacher, str)
                    and "/" in teacher
                    and not teacher.startswith("/")
                ):
                    candidates.append(teacher)
            except Exception:  # noqa: BLE001
                pass

        # 4. Paligemma fallback -- the VLM tokenizer all pi05 students use.
        # Safe because SnapFlow distill preserves the input contract from
        # the teacher (which is paligemma-based for pi0/pi05).
        candidates.append("google/paligemma-3b-pt-224")

        for cand in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(cand)
                self._tokenizer = tok
                logger.info(
                    "Pi05DecomposedServer tokenizer loaded from %r "
                    "(vocab_size=%d)", cand, tok.vocab_size,
                )
                return
            except Exception as exc:  # noqa: BLE001
                tried.append(cand)
                last_exc = exc
                continue

        logger.error(
            "Pi05DecomposedServer tokenizer load failed; tried %d sources: "
            "%r. Last error: %s", len(tried), tried, last_exc,
        )
        raise RuntimeError(
            f"No tokenizer found for {self.export_dir}. Tried {len(tried)} "
            f"sources ({tried!r}). Pi05DecomposedServer needs a tokenizer "
            f"to encode the instruction. Last error: {last_exc}"
        )

    # -- Server-interface contract -----------------------------------------

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Sync entry point. Mirrors ReflexServer.predict_from_base64."""
        # Decode image
        image: np.ndarray | None = None
        if image_b64:
            try:
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Failed to decode image: {exc}"}

        return self._predict(
            image=image, instruction=instruction or "", state=state,
        )

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Async entry point matching ReflexServer's contract."""
        # Pi05DecomposedInference is sync-only; offload to thread to avoid
        # blocking the FastAPI event loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.predict_from_base64,
            image_b64, instruction, state,
        )

    async def run_batch(self, requests: list) -> list[dict[str, Any]]:
        """PolicyRuntime's run_batch_callback. Sequential dispatch (no
        true batching for decomposed; future dynamic-shape exporter
        enables N>1)."""
        results: list[dict[str, Any]] = []
        for req in requests:
            res = await self.predict_from_base64_async(
                image_b64=getattr(req, "image", None),
                instruction=getattr(req, "instruction", "") or "",
                state=getattr(req, "state", None),
            )
            results.append(res)
        return results

    # -- Internal: prep + inference + postprocess --------------------------

    def _predict(
        self,
        *,
        image: np.ndarray | None,
        instruction: str,
        state: list[float] | None,
    ) -> dict[str, Any]:
        """Pure-numpy prep + inference + postprocess. Catches + envelopes
        any exception in the standard error dict (so /act returns a clean
        envelope; circuit breaker fires on the dict-with-error key)."""
        import time
        if not self._ready or self._inference is None:
            return {"error": "server not ready -- load() not called"}

        t0 = time.perf_counter()
        try:
            # 1. Image prep -- 1 customer image becomes 3 cameras (base + 2
            #    padded). Resize/pad to camera_resolution. Float32 [0, 1].
            img_base, mask_base = self._prep_image(image)
            # Padded second + third cameras (per the SmolVLA convention:
            # missing cams are -1 image + zero mask).
            img_pad = np.full_like(img_base, -1.0)
            mask_pad = np.zeros_like(mask_base)

            # 2. Tokenize instruction
            self._ensure_tokenizer()
            lang_tokens, lang_masks = self._tokenize(instruction)

            # 3. State pad
            state_arr = self._prep_state(state)

            # 4. Sample noise
            noise = np.random.randn(
                1, self.chunk_size, self.max_action_dim,
            ).astype(np.float32)

            # 5. Inference
            actions_padded = self._inference.predict_action_chunk(
                img_base=img_base, img_wrist_l=img_pad, img_wrist_r=img_pad,
                mask_base=mask_base, mask_wrist_l=mask_pad, mask_wrist_r=mask_pad,
                lang_tokens=lang_tokens, lang_masks=lang_masks,
                noise=noise, state=state_arr,
            )

            # 6. Postprocess -- slice to action_dim; first batch element.
            if actions_padded.ndim == 3:
                actions_padded = actions_padded[0]
            actions_out = actions_padded[:, : self.action_dim].astype(np.float32)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._latency_history.append(elapsed_ms)

            return {
                "actions": actions_out.tolist(),
                "num_actions": int(actions_out.shape[0]),
                "action_dim": self.action_dim,
                "latency_ms": elapsed_ms,
                "inference_mode": self._inference_mode,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pi05DecomposedServer.predict failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _prep_image(
        self, image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize/pad an HxWx3 uint8 image to (1, 3, R, R) float32 [0, 1] +
        a (1,) bool mask. When image is None, returns a -1 padding image +
        zero mask (signals "missing camera" to the model)."""
        R = self.camera_resolution
        if image is None:
            img = np.full((1, 3, R, R), -1.0, dtype=np.float32)
            mask = np.zeros((1,), dtype=bool)
            return img, mask

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be HxWx3 uint8, got shape {image.shape}"
            )

        # Resize-with-pad to RxR (preserves aspect; pads short side).
        from PIL import Image
        h, w = image.shape[:2]
        if h > w:
            pad = (h - w) // 2
            image = np.pad(
                image, [(0, 0), (pad, h - w - pad), (0, 0)], mode="constant",
            )
        elif w > h:
            pad = (w - h) // 2
            image = np.pad(
                image, [(pad, w - h - pad), (0, 0), (0, 0)], mode="constant",
            )
        pil = Image.fromarray(image)
        pil = pil.resize((R, R), Image.BILINEAR)
        resized = np.asarray(pil)

        # CHW + add batch dim + normalize to [0, 1]
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        return chw[np.newaxis, ...], np.ones((1,), dtype=bool)

    def _tokenize(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the instruction to lang_seq_len; returns (tokens, masks).

        Pads with 0 tokens + zero mask when the instruction is shorter than
        lang_seq_len; truncates when longer. This matches the SmolVLA
        convention used by modal_libero_*.py.
        """
        encoded = self._tokenizer(
            instruction or "",
            padding="max_length",
            max_length=self.lang_seq_len,
            truncation=True,
            return_tensors="np",
        )
        tokens = encoded["input_ids"].astype(np.int64)
        masks = encoded["attention_mask"].astype(bool)
        return tokens, masks

    def _prep_state(self, state: list[float] | None) -> np.ndarray:
        """Pad the state vector to max_action_dim (typically 32 for pi05).
        Customer-side state is usually 7-14 dims; pad with zeros.
        Returns (1, max_action_dim) float32."""
        if state is None:
            return np.zeros((1, self.max_action_dim), dtype=np.float32)
        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.shape[0] > self.max_action_dim:
            arr = arr[: self.max_action_dim]
        elif arr.shape[0] < self.max_action_dim:
            arr = np.concatenate([
                arr, np.zeros(self.max_action_dim - arr.shape[0], dtype=np.float32),
            ])
        return arr[np.newaxis, ...]


__all__ = [
    "DEFAULT_CAMERA_RESOLUTION",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_LANG_SEQ_LEN",
    "DEFAULT_MAX_ACTION_DIM",
    "Pi05DecomposedServer",
]
