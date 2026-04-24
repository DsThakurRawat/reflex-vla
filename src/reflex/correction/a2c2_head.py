"""A2C2 correction head — small MLP that emits per-chunk-position action residuals.

Architecture (paper-derived, configurable via A2C2Config):
    Input  = concat([base_action, obs_features, chunk_pos_enc, latency_log_enc])
    Hidden = 2-3 dense layers, GELU activations
    Output = correction (same shape as base_action)

Default config targets ~100 KB FP16 weights:
    action_dim=7, obs_dim=256, chunk_pos_dim=32, latency_dim=32
    -> input_dim=327 -> 128 -> 96 -> 7
    -> ~50k params -> ~100 KB FP16 / ~200 KB FP32

The head is **forward-only** in this module. Training lives in
scripts/train_a2c2.py (uses standard PyTorch optim/loss). This separation lets us
ship the runtime hook (Phase B.5) without dragging the trainer into prod.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover — torch is required for A2C2
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class A2C2Config:
    """Architecture knobs for the A2C2 head.

    Defaults target the paper's ~100 KB envelope (FP16 storage).
    """

    action_dim: int = 7
    obs_dim: int = 256
    chunk_pos_dim: int = 32
    latency_dim: int = 32
    hidden_dims: tuple[int, ...] = field(default_factory=lambda: (128, 96))
    dropout: float = 0.0
    activation: str = "gelu"
    max_chunk_idx: int = 50
    latency_min_ms: float = 1.0
    latency_max_ms: float = 1000.0

    @property
    def input_dim(self) -> int:
        return self.action_dim + self.obs_dim + self.chunk_pos_dim + self.latency_dim

    def estimated_param_count(self) -> int:
        layers = (self.input_dim,) + tuple(self.hidden_dims) + (self.action_dim,)
        return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))

    def estimated_size_bytes(self, fp16: bool = True) -> int:
        bytes_per_param = 2 if fp16 else 4
        return self.estimated_param_count() * bytes_per_param


def chunk_pos_encoding(chunk_idx: np.ndarray | int, dim: int, max_idx: int = 50) -> np.ndarray:
    """Sinusoidal positional encoding for chunk index.

    Returns shape (..., dim). For scalar input, shape is (dim,).
    """
    if dim % 2 != 0:
        raise ValueError(f"chunk_pos_encoding dim must be even; got {dim}")
    if np.isscalar(chunk_idx):
        idx = np.array([chunk_idx], dtype=np.float32)
        single = True
    else:
        idx = np.asarray(chunk_idx, dtype=np.float32)
        single = False
    pos = idx / max(max_idx, 1)
    div = np.exp(-np.arange(0, dim, 2, dtype=np.float32) * (math.log(10_000.0) / dim))
    angles = pos[..., None] * div[None, :] * (2 * math.pi)
    enc = np.empty(pos.shape + (dim,), dtype=np.float32)
    enc[..., 0::2] = np.sin(angles)
    enc[..., 1::2] = np.cos(angles)
    return enc[0] if single else enc


def latency_log_encoding(
    latency_ms: np.ndarray | float,
    dim: int,
    lo: float = 1.0,
    hi: float = 1000.0,
) -> np.ndarray:
    """Log-scale sinusoidal encoding for latency in [lo, hi] ms.

    Latency clamped into [lo, hi] then mapped to [0, 1] in log space, then encoded
    sinusoidally so the head can attend to broad bands (10-30 ms vs 80-200 ms).
    """
    if dim % 2 != 0:
        raise ValueError(f"latency_log_encoding dim must be even; got {dim}")
    if np.isscalar(latency_ms):
        v = np.array([latency_ms], dtype=np.float32)
        single = True
    else:
        v = np.asarray(latency_ms, dtype=np.float32)
        single = False
    v = np.clip(v, lo, hi)
    log_lo = math.log(max(lo, 1e-6))
    log_hi = math.log(hi)
    norm = (np.log(np.maximum(v, 1e-6)) - log_lo) / max(log_hi - log_lo, 1e-6)
    div = np.exp(-np.arange(0, dim, 2, dtype=np.float32) * (math.log(10_000.0) / dim))
    angles = norm[..., None] * div[None, :] * (2 * math.pi)
    enc = np.empty(norm.shape + (dim,), dtype=np.float32)
    enc[..., 0::2] = np.sin(angles)
    enc[..., 1::2] = np.cos(angles)
    return enc[0] if single else enc


def build_a2c2_input(
    base_action: np.ndarray,
    obs_features: np.ndarray,
    chunk_idx: np.ndarray | int,
    latency_ms: np.ndarray | float,
    cfg: A2C2Config,
) -> np.ndarray:
    """Concatenate the standard A2C2 input vector.

    base_action:  (..., action_dim)
    obs_features: (..., obs_dim)        — VLM prefix pool, image features, etc.
    chunk_idx:    (...) or scalar       — index in chunk [0, max_chunk_idx)
    latency_ms:   (...) or scalar       — observed/estimated inference delay

    Returns: (..., input_dim)
    """
    base_action = np.asarray(base_action, dtype=np.float32)
    obs_features = np.asarray(obs_features, dtype=np.float32)
    if base_action.shape[-1] != cfg.action_dim:
        raise ValueError(
            f"base_action last dim {base_action.shape[-1]} != cfg.action_dim {cfg.action_dim}"
        )
    if obs_features.shape[-1] != cfg.obs_dim:
        raise ValueError(
            f"obs_features last dim {obs_features.shape[-1]} != cfg.obs_dim {cfg.obs_dim}"
        )
    chunk_enc = chunk_pos_encoding(chunk_idx, cfg.chunk_pos_dim, cfg.max_chunk_idx)
    lat_enc = latency_log_encoding(latency_ms, cfg.latency_dim, cfg.latency_min_ms, cfg.latency_max_ms)
    if base_action.ndim > 1 and chunk_enc.ndim < base_action.ndim:
        broadcast_shape = base_action.shape[:-1] + (cfg.chunk_pos_dim,)
        chunk_enc = np.broadcast_to(chunk_enc, broadcast_shape).copy()
        lat_enc = np.broadcast_to(lat_enc, base_action.shape[:-1] + (cfg.latency_dim,)).copy()
    return np.concatenate([base_action, obs_features, chunk_enc, lat_enc], axis=-1)


if _HAS_TORCH:

    class A2C2Head(nn.Module):
        """Forward-only correction head. Training in scripts/train_a2c2.py.

        Forward signature: (input_vec) -> correction
            input_vec shape:  (B, input_dim)  or  (input_dim,)
            correction shape: (B, action_dim) or (action_dim,)

        Use build_a2c2_input(...) to assemble input_vec from raw tensors.
        """

        def __init__(self, cfg: A2C2Config):
            super().__init__()
            self.cfg = cfg
            act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.activation]
            layers: list[nn.Module] = []
            prev = cfg.input_dim
            for h in cfg.hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(act_cls())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                prev = h
            layers.append(nn.Linear(prev, cfg.action_dim))
            self.net = nn.Sequential(*layers)
            with torch.no_grad():
                self.net[-1].weight.zero_()
                self.net[-1].bias.zero_()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)

        def correct(
            self,
            base_action: np.ndarray,
            obs_features: np.ndarray,
            chunk_idx: int,
            latency_ms: float,
        ) -> np.ndarray:
            """One-shot inference helper: returns base_action + correction (numpy)."""
            self.eval()
            inp = build_a2c2_input(base_action, obs_features, chunk_idx, latency_ms, self.cfg)
            with torch.no_grad():
                tensor = torch.from_numpy(inp).float()
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                    correction = self.forward(tensor).squeeze(0).cpu().numpy()
                else:
                    correction = self.forward(tensor).cpu().numpy()
            return base_action + correction

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def size_bytes(self, fp16: bool = True) -> int:
            return self.param_count() * (2 if fp16 else 4)

else:

    class A2C2Head:  # type: ignore[no-redef]
        """Stub raised when torch is missing — A2C2 requires PyTorch."""

        def __init__(self, *_: object, **__: object):
            raise ImportError(
                "A2C2Head requires torch. Install with: pip install 'reflex-vla[correction]'"
            )
