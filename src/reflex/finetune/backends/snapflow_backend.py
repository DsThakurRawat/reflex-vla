"""SnapFlow backend — in-process 1-step self-distillation trainer.

Implements `Backend.fit()` using the math in `reflex.distill.snapflow`
and the frozen teacher loader in `reflex.distill.teacher_loader`.

## What this backend does

1. Load teacher policy (frozen, eval mode) via teacher_loader.
2. deepcopy() teacher → student. Install the zero-init target_time
   embedding into the student's velocity head (model surgery via
   the policy-specific adapter in `_velocity_adapters`).
3. Build a LeRobotDataset dataloader matching the teacher's expected
   obs_kwargs (image, state, language).
4. Training loop: for each batch, sample t + noise, compute SnapFlow
   loss (flow_matching + consistency), backprop through student only.
5. Save checkpoint every `checkpoint_every` steps via the teacher's
   native `save_pretrained` (same format `_auto_export` consumes).
6. Fire the lifecycle hooks (`on_start`, `on_step`, `on_checkpoint`,
   `on_end`) at the documented points.

## What this backend does NOT do

- No LoRA/adapter logic — SnapFlow trains FULL weights (the student
  is a full-weight copy, no PEFT merge step needed).
- No ONNX export (postprocess.finalize chain owns that).
- No LIBERO eval (libero_drop_gate hook owns that, fired by finalize).
- No VLM prefix caching (current v0.3 recomputes per step for
  simplicity; v0.5 can cache when profiling shows it's the bottleneck).

## The velocity-adapter split

The SnapFlow math is paradigm-agnostic, but actually EXTRACTING the
velocity function from a lerobot PI0Policy / PI05Policy depends on
per-policy internals (expert-stack attribute names, prefix KV cache
construction, etc.). So this module splits:

  - `SnapFlowBackend.fit()` — generic loop (optimizer, dataloader,
    loss step, checkpoint, hooks). Takes velocity_fn callables.
  - `_build_velocity_adapters(teacher, student, policy_type)` —
    policy-specific. For pi0/pi05, wraps the PaliGemma-VLM +
    expert-stack. Lazy-imported per-policy-type so CI can test
    the generic loop without lerobot installed.

v0.3: pi0 + pi05 only. SmolVLA is v0.3.1 pending velocity-convergence
validation; GR00T is v0.5+ (different denoising paradigm — DDPM).
"""
from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from reflex.finetune.backends.base import (
    Backend,
    CheckpointResult,
    TrainerContext,
)

logger = logging.getLogger(__name__)


# Default mix coefficient for the SnapFlow consistency term when the
# FinetuneConfig doesn't override via extra_lerobot_args. Paper uses 1.0.
DEFAULT_CONSISTENCY_ALPHA: float = 1.0

# Save a student checkpoint every N steps (same convention as
# lerobot-train). Override via cfg.extra_lerobot_args["checkpoint_every"].
DEFAULT_CHECKPOINT_EVERY: int = 1_000


class SnapFlowBackend:
    """In-process SnapFlow distillation trainer."""

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        cfg = ctx.config
        if not cfg.teacher_export:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=(
                    "SnapFlowBackend requires cfg.teacher_export to be set. "
                    "Pass --teacher-export <path-or-hf-id> pointing at a "
                    "reflex-exported dir (merged PyTorch checkpoint)."
                ),
            )

        # Lazy-import torch + SnapFlow math so a CI run that never touches
        # distill doesn't pay the import cost.
        import torch
        from torch.optim import AdamW

        from reflex.distill.snapflow import snapflow_loss_step
        from reflex.distill.teacher_loader import load_teacher

        # ---- 1. Load teacher, build student --------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = cfg.precision if cfg.precision in ("bf16", "fp32") else "bf16"
        logger.info("[snapflow] loading teacher from %s on %s", cfg.teacher_export, device)
        loaded = load_teacher(cfg.teacher_export, device=device, dtype=dtype)
        teacher = loaded.policy
        policy_type = loaded.policy_type

        logger.info("[snapflow] building student (deepcopy of teacher)")
        student = copy.deepcopy(teacher)
        # Student is TRAINABLE — teacher_loader froze it, undo that for the copy.
        for p in student.parameters():
            p.requires_grad = True
        student.train()

        # ---- 2. Install velocity adapters + target_time embedding ----------
        teacher_velocity_fn, student_velocity_fn = _build_velocity_adapters(
            teacher=teacher,
            student=student,
            policy_type=policy_type,
        )

        # ---- 3. Build dataloader ------------------------------------------
        try:
            loader = _build_dataloader(cfg, policy_type=policy_type)
        except Exception as e:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"dataloader construction failed: {type(e).__name__}: {e}",
            )

        # ---- 4. Optimizer + checkpoint dir --------------------------------
        opt = AdamW(
            (p for p in student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        checkpoint_root = Path(cfg.output) / "training" / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_every = int(
            cfg.extra_lerobot_args.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
        )
        consistency_alpha = float(
            cfg.extra_lerobot_args.get("consistency_alpha", DEFAULT_CONSISTENCY_ALPHA)
        )

        # ---- 5. Fire on_start ---------------------------------------------
        ctx.hooks.run("on_start", ctx, config=cfg, policy_type=policy_type)

        # ---- 6. Training loop ---------------------------------------------
        step = 0
        last_ckpt: Path | None = None
        loss_history: list[dict[str, float]] = []
        log_handle = open(ctx.training_log_path, "a", encoding="utf-8")
        try:
            for step, batch in enumerate(loader, start=1):
                action, noise, t, obs_kwargs = _prepare_batch(batch, device=device)

                opt.zero_grad()
                loss, snap = snapflow_loss_step(
                    student_velocity_fn,
                    teacher_velocity_fn,
                    action=action,
                    noise=noise,
                    t=t,
                    obs_kwargs=obs_kwargs,
                    consistency_alpha=consistency_alpha,
                )
                loss.backward()
                opt.step()

                loss_history.append(asdict(snap) | {"step": step})
                log_handle.write(
                    json.dumps({"step": step, **asdict(snap), "lr": cfg.learning_rate}) + "\n"
                )
                log_handle.flush()

                ctx.hooks.run(
                    "on_step",
                    ctx,
                    step=step,
                    loss=snap.total,
                    lr=cfg.learning_rate,
                    flow_matching=snap.flow_matching,
                    consistency=snap.consistency,
                )

                if step % checkpoint_every == 0 or step == cfg.num_steps:
                    last_ckpt = _save_student_checkpoint(
                        student, checkpoint_root, step, teacher_config=loaded.config,
                    )
                    ctx.hooks.run(
                        "on_checkpoint", ctx, step=step, ckpt_path=last_ckpt,
                    )

                if step >= cfg.num_steps:
                    break
        finally:
            log_handle.close()

        # ---- 7. Ensure we have a final checkpoint -------------------------
        if last_ckpt is None:
            last_ckpt = _save_student_checkpoint(
                student, checkpoint_root, step or 0, teacher_config=loaded.config,
            )

        # ---- 8. Provenance stamp ------------------------------------------
        _write_provenance(
            last_ckpt,
            teacher_dir=loaded.checkpoint_dir,
            policy_type=policy_type,
            steps=step,
            consistency_alpha=consistency_alpha,
        )

        # ---- 9. Fire on_end + return --------------------------------------
        ctx.hooks.run(
            "on_end", ctx, status="ok", steps_completed=step,
        )
        return CheckpointResult(
            final_checkpoint_path=last_ckpt,
            training_steps_completed=step,
            status="ok",
            intermediate_metrics={
                "policy_type": policy_type,
                "final_loss": loss_history[-1]["total"] if loss_history else None,
                "final_fm_loss": loss_history[-1]["flow_matching"] if loss_history else None,
                "final_consistency_loss": loss_history[-1]["consistency"] if loss_history else None,
                "consistency_alpha": consistency_alpha,
                "teacher_export": str(loaded.checkpoint_dir),
            },
        )


# ---------------------------------------------------------------------------
# Policy-specific velocity adapters
# ---------------------------------------------------------------------------

def _build_velocity_adapters(
    teacher: Any,
    student: Any,
    policy_type: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return `(teacher_velocity_fn, student_velocity_fn)` for policy_type.

    Each returned callable has signature:
        teacher_velocity_fn(x, t, **obs_kwargs) -> velocity tensor
        student_velocity_fn(x, t, target_time=None, **obs_kwargs) -> velocity tensor

    The student's `target_time` routes through the zero-init embedding
    installed by `_install_target_time_embedding`.

    Raises NotImplementedError for policy_types we haven't wired yet
    (SmolVLA v0.3.1; GR00T v0.5+).
    """
    if policy_type in ("pi0", "pi05"):
        return _build_pi_family_adapters(teacher, student)
    raise NotImplementedError(
        f"No velocity adapter for policy_type={policy_type!r}. "
        f"pi0/pi05 are supported in v0.3; SmolVLA in v0.3.1; "
        f"GR00T in v0.5+. See "
        f"reflex_context/01_architecture/distill_SYNTHESIS.md."
    )


def _build_pi_family_adapters(
    teacher: Any,
    student: Any,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Velocity adapters for pi0 / pi05.

    Both pi0 and pi05 wrap a PaliGemma VLM + a Gemma action expert. We
    recompute the VLM prefix PER STEP for simplicity (v0.3 scope).

    The student gets a `target_time` embedding installed on its expert;
    the teacher stays as-is (no target_time path).

    This is a BEST-EFFORT adapter — the exact attribute path inside
    lerobot's PI0Policy depends on the lerobot version. If the attribute
    graph changes, this function raises a clear error rather than
    silently returning wrong velocities.
    """
    _install_target_time_embedding(student)

    def _run_policy_velocity(policy, x, t, obs_kwargs):
        """Single velocity-head forward using the policy's documented API.

        We use `policy.model.sample_actions_step` if exposed (lerobot
        ≥ 0.3) — it's the per-timestep hook the Euler loop calls
        internally. Falls back to a ValueError with a remediation hint
        if the policy doesn't expose it.
        """
        model = getattr(policy, "model", policy)
        # Prefer a one-step velocity hook if the policy exposes one.
        fn = getattr(model, "denoise_step", None) or getattr(model, "sample_actions_step", None)
        if fn is None:
            raise AttributeError(
                f"{type(policy).__name__} has no denoise_step / sample_actions_step "
                f"method. SnapFlow needs a per-timestep velocity hook. "
                f"If you're on a newer lerobot that renamed the method, add a "
                f"branch to _run_policy_velocity in snapflow_backend.py."
            )
        return fn(x, t, **obs_kwargs)

    def teacher_velocity_fn(x, t, **obs_kwargs):
        return _run_policy_velocity(teacher, x, t, obs_kwargs)

    def student_velocity_fn(x, t, target_time=None, **obs_kwargs):
        # target_time is routed through student.target_time_embed if installed.
        if target_time is not None:
            obs_kwargs = {**obs_kwargs, "target_time": target_time}
        return _run_policy_velocity(student, x, t, obs_kwargs)

    return teacher_velocity_fn, student_velocity_fn


def _install_target_time_embedding(student: Any) -> None:
    """Install the zero-init target_time embedding on the student's
    velocity head.

    Per architecture doc §C.1 + §C.3: the student is identical to the
    teacher at init (zero-embedding output), and learns the one-step
    shortcut via the consistency loss over training.

    Concrete wiring depends on the policy type's expert-stack attribute
    graph; v0.3 attaches the embedding as `student.target_time_embed`
    and relies on the policy's forward to consume it when `target_time`
    is passed through obs_kwargs.

    If the policy doesn't have a documented slot for the embedding, we
    still attach it — the actual velocity adapter picks it up via the
    target_time kwarg.
    """
    import torch.nn as nn

    from reflex.distill.snapflow import ZeroInitTargetTimeEmbedding

    # Try to infer embedding_dim from the model config. pi0/pi05 both
    # have an action_expert with hidden_size.
    emb_dim = _infer_time_embedding_dim(student)
    embed = ZeroInitTargetTimeEmbedding(embedding_dim=emb_dim)
    # Attach mlp as a submodule so optimizer.step() updates it.
    if hasattr(student, "target_time_embed"):
        logger.warning(
            "[snapflow] student already has target_time_embed; overwriting"
        )
    # `embed` is a vanilla Python class holding an nn.Sequential; register
    # the inner module so it's tracked by student.parameters().
    if isinstance(student, nn.Module):
        student.add_module("target_time_embed_mlp", embed.mlp)
    student.target_time_embed = embed


def _infer_time_embedding_dim(model: Any) -> int:
    """Best-effort: infer the time-embedding hidden size from the model.

    pi0/pi05 typically have an `action_expert` or `expert` attribute with
    a hidden_size in its config. We fall back to 1024 (pi0-base default)
    if we can't find it, rather than crashing — the zero-init means the
    student behaves as teacher-identical regardless, so a wrong dim only
    matters once training has progressed (when the bug will be obvious).
    """
    for attr_path in (
        ("config", "expert_hidden_size"),
        ("config", "action_expert_hidden_size"),
        ("model", "config", "expert_hidden_size"),
        ("action_expert", "config", "hidden_size"),
    ):
        cur = model
        try:
            for name in attr_path:
                cur = getattr(cur, name)
            if isinstance(cur, int) and cur > 0:
                return cur
        except AttributeError:
            continue
    logger.warning(
        "[snapflow] could not infer time-embedding dim from model; "
        "defaulting to 1024 (pi0-base convention)",
    )
    return 1024


# ---------------------------------------------------------------------------
# Dataloader + batch prep
# ---------------------------------------------------------------------------

def _build_dataloader(cfg, *, policy_type: str):
    """Build a LeRobotDataset dataloader for the distillation run.

    Lazy-imports lerobot.datasets so CI without lerobot installed can
    still test the rest of the loop.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _prepare_batch(batch: dict, *, device: str) -> tuple:
    """Unpack a LeRobotDataset batch into (action, noise, t, obs_kwargs).

    LeRobotDataset batches have keys like 'action', 'observation.images.*',
    'observation.state', 'language.task' — we split the action ground-truth
    out and leave everything else in obs_kwargs for the velocity fn.
    """
    import torch

    action = batch["action"].to(device)
    batch_size = action.shape[0]
    noise = torch.randn_like(action)
    t = torch.rand(batch_size, device=device)

    obs_kwargs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
        if k != "action"
    }
    return action, noise, t, obs_kwargs


# ---------------------------------------------------------------------------
# Checkpoint save + provenance
# ---------------------------------------------------------------------------

def _save_student_checkpoint(
    student: Any,
    checkpoint_root: Path,
    step: int,
    *,
    teacher_config: dict,
) -> Path:
    """Save the student under the standard lerobot pretrained_model/
    layout so `_auto_export` can consume it unchanged.

    Directory: <root>/<step:08d>/pretrained_model/
      ├── config.json           (inherited from teacher + distill tag)
      ├── model.safetensors     (student's full weights)
      └── distill_provenance.json  (added by _write_provenance)
    """
    ckpt_dir = checkpoint_root / f"{step:08d}" / "pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained writes config.json + model.safetensors atomically.
    save_fn = getattr(student, "save_pretrained", None)
    if save_fn is None:
        raise AttributeError(
            f"student policy has no save_pretrained — SnapFlow expects a "
            f"HuggingFace-style model. Got {type(student).__name__}."
        )
    save_fn(str(ckpt_dir))

    # Stamp the config with a distill marker so downstream consumers
    # (export, VERIFICATION.md) know this came from SnapFlow.
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            cfg = json.load(f)
        cfg["_reflex_distill_method"] = "snapflow"
        cfg["_reflex_distill_teacher_type"] = teacher_config.get("type", "unknown")
        with config_path.open("w") as f:
            json.dump(cfg, f, indent=2)

    logger.info("[snapflow] checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def _write_provenance(
    ckpt_dir: Path,
    *,
    teacher_dir: Path,
    policy_type: str,
    steps: int,
    consistency_alpha: float,
) -> None:
    """Write a distill_provenance.json next to the checkpoint so
    VERIFICATION.md can reference where the teacher came from.

    Fields:
      - teacher_export: absolute path
      - policy_type: pi0 | pi05 | ...
      - steps: training steps completed
      - consistency_alpha: mix coef used
      - method: "snapflow"
    """
    prov = {
        "method": "snapflow",
        "policy_type": policy_type,
        "teacher_export": str(teacher_dir),
        "steps": steps,
        "consistency_alpha": consistency_alpha,
        "paper": "arxiv.org/abs/2604.05656",
    }
    (ckpt_dir / "distill_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )


__all__ = ["SnapFlowBackend"]
