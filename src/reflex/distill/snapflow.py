"""SnapFlow — 1-step self-distillation for flow-matching VLAs.

Reimplementation from Luan et al. "SnapFlow" (arxiv 2604.05656, Apr 2026).
The paper showed 98.75% LIBERO task success at 1-NFE vs 97.75% at the
10-step teacher — i.e. distillation IMPROVED task success on pi0.5.

Code not yet public (as of 2026-04-20); reflex is the first reproducible
implementation. This module is PURE MATH + modeling — unit-testable
without GPU. Backend glue (teacher loader, dataloader, checkpoint save)
lives in `src/reflex/finetune/backends/snapflow_backend.py`.

## The three loss components

1. **Flow-matching term**: standard flow-matching MSE on velocity.
   Student matches the true analytic velocity at random time t.
2. **Consistency term**: student at target_time=1 (one-step generation)
   matches a 2-step Euler shortcut from the TEACHER. The teacher generates
   `x_t` → `x_mid` → `v_shortcut`; the student learns to produce
   `v_shortcut` directly from `x_t` in one shot.
3. **Zero-init target-time embedding**: the student adds a new learnable
   embedding for `target_time` that's initialized to produce zero output.
   At `target_time=None` it behaves identically to the teacher (normal
   velocity prediction); at `target_time=1.0` it activates the one-step
   path. This is the SnapFlow trick — one network, dual modes.

## What this module does NOT do

- No teacher loading (lerobot policy classes → `teacher_loader.py`)
- No dataloader construction (LeRobotDataset wrapping → backend)
- No checkpoint save format (lerobot `pretrained_model/` layout → backend)
- No ONNX export (handled by `reflex.exporters.monolithic` post-training)

Reference: architecture doc Section C.1-C.3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default mixing coefficient for the consistency term.
# Paper uses alpha=1.0; we parameterize in case we want to anneal it.
DEFAULT_CONSISTENCY_ALPHA: float = 1.0


@dataclass
class SnapFlowLosses:
    """Per-step loss breakdown. Backend logs these to the training log
    for observability + later VERIFICATION.md rendering."""

    flow_matching: float
    consistency: float
    total: float


def flow_matching_interp(
    noise: "torch.Tensor",
    action: "torch.Tensor",
    t: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Linear flow-matching interpolation.

    Returns (x_t, v_target) where:
      x_t      = (1-t) * noise + t * action
      v_target = action - noise     (the analytic velocity)

    Shapes:
      noise, action: (B, chunk, action_dim)
      t:             (B,) — random per-sample time
      x_t, v_target: (B, chunk, action_dim)

    `t` is broadcast along chunk + action_dim axes. Caller should have
    sampled t ~ Uniform(0, 1) per batch element.
    """
    import torch

    assert noise.shape == action.shape, f"noise {noise.shape} vs action {action.shape}"
    # Broadcast t to (B, 1, 1) so it multiplies per-sample
    while t.ndim < noise.ndim:
        t = t.unsqueeze(-1)
    x_t = (1.0 - t) * noise + t * action
    v_target = action - noise  # independent of t (linear interp)
    return x_t, v_target


def two_step_euler_shortcut(
    teacher_velocity_fn: Callable[..., "torch.Tensor"],
    x_t: "torch.Tensor",
    t: "torch.Tensor",
    *,
    obs_kwargs: dict,
) -> "torch.Tensor":
    """Compute SnapFlow's 2-step Euler shortcut velocity from the teacher.

    The shortcut estimates "what one-step velocity moves x_t to action"
    using TWO small Euler steps with the teacher velocity field:

        v_t         = teacher(x_t,   t)
        x_mid       = x_t + 0.5*(1-t) * v_t                         # Euler half-step
        v_mid       = teacher(x_mid, t + 0.5*(1-t))
        v_shortcut  = 0.5 * (v_t + v_mid)                           # avg

    Expects `teacher_velocity_fn(x, t, **obs_kwargs) -> velocity`.
    Runs under `torch.no_grad()` inside — caller doesn't need to wrap.

    Returns a tensor shaped like x_t.
    """
    import torch

    with torch.no_grad():
        v_t = teacher_velocity_fn(x_t, t, **obs_kwargs)
        # Broadcast t to match v_t dims for the half-step addition
        t_broadcast = t
        while t_broadcast.ndim < x_t.ndim:
            t_broadcast = t_broadcast.unsqueeze(-1)
        half_step = 0.5 * (1.0 - t_broadcast)
        x_mid = x_t + half_step * v_t
        t_mid = t + 0.5 * (1.0 - t)
        v_mid = teacher_velocity_fn(x_mid, t_mid, **obs_kwargs)
        v_shortcut = 0.5 * (v_t + v_mid)
    return v_shortcut


def snapflow_loss_step(
    student_velocity_fn: Callable[..., "torch.Tensor"],
    teacher_velocity_fn: Callable[..., "torch.Tensor"],
    *,
    action: "torch.Tensor",
    noise: "torch.Tensor",
    t: "torch.Tensor",
    obs_kwargs: dict,
    consistency_alpha: float = DEFAULT_CONSISTENCY_ALPHA,
) -> tuple["torch.Tensor", SnapFlowLosses]:
    """Compute one SnapFlow training-step loss.

    Args:
      student_velocity_fn: callable `(x, t, target_time=None, **obs_kwargs) -> v`.
        When `target_time=None`, behaves as standard velocity prediction.
        When `target_time=1.0`, produces the one-step generation output.
      teacher_velocity_fn: callable `(x, t, **obs_kwargs) -> v` (frozen).
      action: ground-truth action chunk (B, chunk, action_dim).
      noise: Gaussian noise, same shape as action.
      t: random time per sample (B,), sampled Uniform(0, 1).
      obs_kwargs: VLM conditioning inputs (image, state, language) that
        both student and teacher forward passes need.
      consistency_alpha: mix weight for the consistency term.

    Returns (loss_tensor, SnapFlowLosses snapshot). Caller does
    loss.backward() and optimizer.step().
    """
    import torch
    import torch.nn.functional as F

    # (a) Flow-matching term: student must match the analytic velocity
    # at the random interpolation point.
    x_t, v_target = flow_matching_interp(noise, action, t)
    v_student = student_velocity_fn(x_t, t, target_time=None, **obs_kwargs)
    fm_loss = F.mse_loss(v_student, v_target)

    # (b) Consistency term: student at target_time=1 should match the
    # teacher's 2-step Euler shortcut from x_t.
    v_shortcut = two_step_euler_shortcut(
        teacher_velocity_fn, x_t, t, obs_kwargs=obs_kwargs,
    )
    target_time_one = torch.ones_like(t)
    v_student_one = student_velocity_fn(
        x_t, t, target_time=target_time_one, **obs_kwargs,
    )
    consistency_loss = F.mse_loss(v_student_one, v_shortcut)

    total = fm_loss + consistency_alpha * consistency_loss
    snapshot = SnapFlowLosses(
        flow_matching=float(fm_loss.item()),
        consistency=float(consistency_loss.item()),
        total=float(total.item()),
    )
    return total, snapshot


class ZeroInitTargetTimeEmbedding:
    """The "zero-init target_time" trick from SnapFlow.

    The student model is a COPY of the teacher, but with a new learnable
    embedding that conditions on `target_time` (the time we want to
    generate AT, distinct from `t` the interpolation point). This
    embedding is initialized to output exactly zero, so early in
    training the student behaves identically to the teacher. Over
    training, the embedding learns to produce the one-step shortcut when
    target_time=1.

    ## How it's wired into the velocity head

    Original teacher: `velocity = head(h_vlm, h_action, time_embed(t))`
    SnapFlow student: `velocity = head(h_vlm, h_action, time_embed(t) + target_time_embed(target_time))`

    When `target_time_embed` is zero-init, the student == teacher at init.
    When training drives the consistency loss down, `target_time_embed(1.0)`
    learns the shortcut-velocity offset.

    ## This class is a CONTRACT, not an implementation

    The actual embedding layer varies per-VLA (SmolVLA uses an MLP on
    sinusoidal-time; pi0 uses a different stack). Each adapter in
    `src/reflex/finetune/backends/snapflow_backend.py` builds the
    right shape for its target model; this class is the common shape
    they all conform to.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
    ):
        import torch
        import torch.nn as nn

        # Two-layer MLP on sinusoidal-time, zero-initialized output.
        # Matches the pattern in SmolVLA's time_embedder.
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Zero-init the OUTPUT layer so the whole thing produces zero
        # initially. Weights zero; bias zero. (Hidden layer kept random
        # so gradients flow.)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def __call__(
        self,
        target_time_sinusoidal: "torch.Tensor",
    ) -> "torch.Tensor":
        """Apply the embedding.

        Args:
          target_time_sinusoidal: sinusoidal embedding of target_time
            with shape (B, embedding_dim). Caller computes the
            sinusoidal encoding so we don't replicate lerobot's utility.

        Returns (B, embedding_dim) embedding, zero at init.
        """
        return self.mlp(target_time_sinusoidal)


def sinusoidal_time_embedding(
    t: "torch.Tensor",
    embedding_dim: int,
    *,
    min_freq: float = 1.0,
    max_freq: float = 1000.0,
) -> "torch.Tensor":
    """Standard sinusoidal encoding for scalar time, as used by
    flow-matching VLAs.

    Args:
      t: (B,) time values in [0, 1].
      embedding_dim: output dim (must be even; half sin + half cos).
      min_freq / max_freq: frequency range; logspace between them.

    Returns (B, embedding_dim).
    """
    import math

    import torch

    assert embedding_dim % 2 == 0, f"embedding_dim {embedding_dim} must be even"
    half = embedding_dim // 2
    freqs = torch.logspace(
        math.log10(min_freq),
        math.log10(max_freq),
        steps=half,
        device=t.device,
        dtype=t.dtype,
    )  # (half,)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 2.0 * math.pi  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
    return emb


__all__ = [
    "DEFAULT_CONSISTENCY_ALPHA",
    "SnapFlowLosses",
    "ZeroInitTargetTimeEmbedding",
    "flow_matching_interp",
    "sinusoidal_time_embedding",
    "snapflow_loss_step",
    "two_step_euler_shortcut",
]
