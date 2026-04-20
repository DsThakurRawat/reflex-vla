"""Tests for Phase B 2/3 — SnapFlow backend + teacher_loader + libero_drop_gate.

These are the three NEW modules the distill architecture needs before
CLI wiring (Phase B 3/3). The tests here pin:
  - teacher_loader.resolve_policy_type (config-field inference)
  - teacher_loader.load_teacher (allowlist + error paths; mock lerobot)
  - SnapFlowBackend.fit (happy path + missing teacher_export + checkpoint format)
  - libero_drop_gate (skip, abort, pass, LIBERO-missing)

No GPU required. lerobot + real LIBERO are mocked throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from reflex.distill.teacher_loader import (
    V03_TEACHER_ALLOWLIST,
    LoadedTeacher,
    load_teacher,
    resolve_policy_type,
)
from reflex.finetune.backends import TrainerContext
from reflex.finetune.backends.base import CheckpointResult
from reflex.finetune.backends.snapflow_backend import (
    DEFAULT_CONSISTENCY_ALPHA,
    SnapFlowBackend,
    _build_velocity_adapters,
    _infer_time_embedding_dim,
    _prepare_batch,
    _save_student_checkpoint,
    _write_provenance,
)
from reflex.finetune.config import FinetuneConfig
from reflex.finetune.hooks import HookRegistry
from reflex.finetune.hooks.libero_drop_gate import (
    DEFAULT_GATE_THRESHOLD_PP,
    _LiberoUnavailable,
    attach_to,
    libero_drop_gate,
)


# ---------------------------------------------------------------------------
# teacher_loader
# ---------------------------------------------------------------------------


class TestResolvePolicyType:
    def test_explicit_type_field_wins(self):
        assert resolve_policy_type({"type": "pi0"}) == "pi0"
        assert resolve_policy_type({"type": "pi05"}) == "pi05"
        assert resolve_policy_type({"type": "smolvla"}) == "smolvla"

    def test_smolvla_inference_from_keys(self):
        cfg = {"load_vlm_weights": True, "chunk_size": 50}
        assert resolve_policy_type(cfg) == "smolvla"

    def test_pi05_inferred_from_num_expert_layers(self):
        cfg = {"prefix_length": 48, "chunk_size": 50, "num_steps": 10, "num_expert_layers": 12}
        assert resolve_policy_type(cfg) == "pi05"

    def test_pi0_default_fallback(self):
        cfg = {"prefix_length": 48, "chunk_size": 50, "num_steps": 10}
        assert resolve_policy_type(cfg) == "pi0"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="could not infer"):
            resolve_policy_type({"random_field": 42})


class TestLoadTeacher:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="teacher export not found"):
            load_teacher(tmp_path / "nope")

    def test_missing_config_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing config.json"):
            load_teacher(tmp_path)

    def test_disallowed_policy_type_raises(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"type": "smolvla"}))
        with pytest.raises(ValueError, match="v0.3"):
            load_teacher(tmp_path, allowlist=frozenset({"pi0", "pi05"}))

    def test_happy_path_returns_loaded_teacher(self, tmp_path):
        """Mocks the lerobot import and verifies eval() + frozen params."""
        (tmp_path / "config.json").write_text(json.dumps({"type": "pi0"}))

        fake_policy = MagicMock()
        fake_policy.parameters.return_value = [
            MagicMock(requires_grad=True) for _ in range(3)
        ]

        with patch(
            "reflex.distill.teacher_loader._load_policy_for_type",
            return_value=fake_policy,
        ):
            loaded = load_teacher(tmp_path, device="cpu", dtype="fp32")
        assert isinstance(loaded, LoadedTeacher)
        assert loaded.policy_type == "pi0"
        assert loaded.checkpoint_dir == tmp_path
        fake_policy.eval.assert_called_once()
        # Every param had requires_grad=False set after load:
        for p in fake_policy.parameters.return_value:
            assert p.requires_grad is False

    def test_v03_allowlist_is_pi_family_only(self):
        assert V03_TEACHER_ALLOWLIST == frozenset({"pi0", "pi05"})


# ---------------------------------------------------------------------------
# snapflow_backend utilities
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    def test_splits_action_from_obs(self):
        batch = {
            "action": torch.randn(4, 50, 7),
            "observation.images.top": torch.randn(4, 3, 224, 224),
            "observation.state": torch.randn(4, 7),
            "language.task": ["pick up block"] * 4,
        }
        action, noise, t, obs = _prepare_batch(batch, device="cpu")
        assert action.shape == (4, 50, 7)
        assert noise.shape == action.shape
        assert t.shape == (4,)
        assert (t >= 0).all() and (t <= 1).all()
        assert "action" not in obs
        assert "observation.images.top" in obs
        assert "language.task" in obs  # non-tensor passed through

    def test_noise_is_fresh_per_call(self):
        batch = {"action": torch.randn(2, 3, 4)}
        _, n1, _, _ = _prepare_batch(batch, device="cpu")
        _, n2, _, _ = _prepare_batch(batch, device="cpu")
        assert not torch.allclose(n1, n2)


class TestInferTimeEmbeddingDim:
    def test_via_config_expert_hidden_size(self):
        model = MagicMock(spec=["config"])
        model.config = MagicMock(spec=["expert_hidden_size"])
        model.config.expert_hidden_size = 768
        assert _infer_time_embedding_dim(model) == 768

    def test_fallback_to_1024_when_unknown(self):
        model = object()  # no attrs
        assert _infer_time_embedding_dim(model) == 1024


class TestBuildVelocityAdapters:
    def test_unknown_policy_type_raises(self):
        with pytest.raises(NotImplementedError, match="No velocity adapter"):
            _build_velocity_adapters(
                teacher=MagicMock(), student=MagicMock(), policy_type="gr00t_n1_5",
            )

    def test_pi_family_wraps_denoise_step(self):
        """The adapter should route (x, t, obs) through denoise_step if present."""
        import torch.nn as nn

        # Teacher + student both expose .model.denoise_step
        def make_policy(val: float):
            m = nn.Module()
            def fn(x, t, **kw):
                return torch.full_like(x, val)
            m.denoise_step = fn
            p = nn.Module()
            p.model = m
            p.config = nn.Module()
            p.config.expert_hidden_size = 64
            return p

        teacher = make_policy(0.1)
        student = make_policy(0.2)

        teacher_fn, student_fn = _build_velocity_adapters(teacher, student, "pi0")

        x = torch.zeros(1, 3, 4)
        t = torch.rand(1)
        vt = teacher_fn(x, t)
        vs = student_fn(x, t)
        assert torch.allclose(vt, torch.full_like(x, 0.1))
        assert torch.allclose(vs, torch.full_like(x, 0.2))

        # Student accepts target_time — the adapter routes it into obs_kwargs
        received = {}
        def probe(x, t, **kw):
            received.update(kw)
            return torch.zeros_like(x)
        student.model.denoise_step = probe
        student_fn(x, t, target_time=torch.ones(1))
        assert "target_time" in received


class TestSaveCheckpoint:
    def test_writes_pretrained_model_dir(self, tmp_path):
        fake = MagicMock()
        def save(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text(json.dumps({"type": "pi0"}))
            (Path(path) / "model.safetensors").write_bytes(b"fake-weights")
        fake.save_pretrained.side_effect = save

        ckpt = _save_student_checkpoint(
            fake, tmp_path / "training" / "checkpoints", step=100,
            teacher_config={"type": "pi0"},
        )
        assert ckpt == tmp_path / "training" / "checkpoints" / "00000100" / "pretrained_model"
        assert (ckpt / "model.safetensors").exists()
        cfg = json.loads((ckpt / "config.json").read_text())
        assert cfg["_reflex_distill_method"] == "snapflow"
        assert cfg["_reflex_distill_teacher_type"] == "pi0"


class TestWriteProvenance:
    def test_writes_expected_fields(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        _write_provenance(
            tmp_path, teacher_dir=Path("/teachers/pi0"),
            policy_type="pi0", steps=5000, consistency_alpha=1.0,
        )
        prov = json.loads((tmp_path / "distill_provenance.json").read_text())
        assert prov["method"] == "snapflow"
        assert prov["policy_type"] == "pi0"
        assert prov["steps"] == 5000
        assert "arxiv.org" in prov["paper"]


class TestSnapFlowBackendFit:
    def test_missing_teacher_export_returns_training_failed(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/pi0", dataset="lerobot/libero",
            output=tmp_path, num_steps=10, phase="distill", teacher_export=None,
        )
        log_path = tmp_path / "training_log.jsonl"
        log_path.touch()
        ctx = TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log_path)

        result = SnapFlowBackend().fit(ctx)
        assert result.status == "training_failed"
        assert "teacher_export" in result.error

    def test_dataloader_failure_returns_training_failed(self, tmp_path):
        """Missing lerobot → dataloader build raises → backend returns error."""
        cfg = FinetuneConfig(
            base="lerobot/pi0", dataset="lerobot/libero",
            output=tmp_path, num_steps=2, phase="distill",
            teacher_export=str(tmp_path / "teacher"),
        )
        (tmp_path / "teacher").mkdir()
        (tmp_path / "teacher" / "config.json").write_text(json.dumps({"type": "pi0"}))
        log_path = tmp_path / "training_log.jsonl"
        log_path.touch()
        ctx = TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log_path)

        # Mock load_teacher to succeed, but let dataloader build blow up
        # (simulates lerobot.datasets import failure).
        fake_policy = MagicMock()
        fake_policy.parameters.return_value = [
            MagicMock(requires_grad=False) for _ in range(2)
        ]
        fake_loaded = LoadedTeacher(
            policy=fake_policy, config={"type": "pi0"},
            policy_type="pi0", checkpoint_dir=tmp_path / "teacher",
        )
        # load_teacher + _build_dataloader are LAZY-imported inside fit(),
        # so we patch them at their source modules (not at snapflow_backend).
        with patch(
            "reflex.distill.teacher_loader.load_teacher",
            return_value=fake_loaded,
        ), patch(
            "reflex.finetune.backends.snapflow_backend._build_velocity_adapters",
            return_value=(lambda *a, **kw: None, lambda *a, **kw: None),
        ), patch(
            "copy.deepcopy",
            return_value=fake_policy,
        ), patch(
            "reflex.finetune.backends.snapflow_backend._build_dataloader",
            side_effect=ImportError("lerobot.datasets not available"),
        ):
            result = SnapFlowBackend().fit(ctx)
        assert result.status == "training_failed"
        assert "dataloader" in result.error.lower()


# ---------------------------------------------------------------------------
# libero_drop_gate
# ---------------------------------------------------------------------------


def _make_distill_ctx(tmp_path, threshold_pp: float = DEFAULT_GATE_THRESHOLD_PP,
                     skip: bool = False, phase: str = "distill"):
    extra = {"libero_gate_threshold_pp": threshold_pp}
    if skip:
        extra["libero_gate_skip"] = True
    cfg = FinetuneConfig(
        base="lerobot/pi0", dataset="lerobot/libero",
        output=tmp_path, num_steps=10, phase=phase,
        teacher_export=str(tmp_path / "teacher"),
        extra_lerobot_args=extra,
    )
    log = tmp_path / "training_log.jsonl"
    log.touch()
    return TrainerContext(config=cfg, hooks=HookRegistry(), training_log_path=log)


class TestLiberoDropGate:
    def test_skip_via_config(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, skip=True)
        libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_non_distill_phase_skips(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, phase="train")
        libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_libero_unavailable_silently_passes(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path)
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            side_effect=_LiberoUnavailable("no libero"),
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert "force_abort" not in ctx.extra

    def test_gate_passes_when_within_threshold(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, threshold_pp=5.0)
        report = MagicMock()
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.90, 0.87),  # 3 pp drop — within threshold
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=report)
        assert "force_abort" not in ctx.extra
        assert report.libero_drop_pp == pytest.approx(3.0, abs=0.01)

    def test_gate_vetoes_when_drop_exceeds_threshold(self, tmp_path):
        ctx = _make_distill_ctx(tmp_path, threshold_pp=5.0)
        report = MagicMock()
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.92, 0.80),  # 12 pp drop — over threshold
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=report)
        assert ctx.extra["force_abort"] is True
        assert "12.0" in ctx.extra["abort_reason"] or "12" in ctx.extra["abort_reason"]
        assert report.libero_drop_pp == pytest.approx(12.0, abs=0.01)

    def test_rollout_crash_aborts(self, tmp_path):
        """Unexpected errors abort — it's safer than silently shipping."""
        ctx = _make_distill_ctx(tmp_path)
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            side_effect=RuntimeError("sim crashed"),
        ):
            libero_drop_gate(ctx, final_checkpoint_path=tmp_path, report=MagicMock())
        assert ctx.extra["force_abort"] is True
        assert "crashed" in ctx.extra["abort_reason"]


class TestAttachTo:
    def test_registers_default_handler(self, tmp_path):
        reg = HookRegistry()
        attach_to(reg)
        assert "on_postprocess" in reg

    def test_threshold_override_reaches_handler(self, tmp_path):
        reg = HookRegistry()
        attach_to(reg, threshold_pp=2.0)
        ctx = _make_distill_ctx(tmp_path)
        report = MagicMock()
        # With threshold=2.0 and 3pp drop, we expect veto
        with patch(
            "reflex.finetune.hooks.libero_drop_gate._run_teacher_student_rollouts",
            return_value=(0.90, 0.87),
        ):
            reg.run("on_postprocess", ctx, final_checkpoint_path=tmp_path, report=report)
        assert ctx.extra["force_abort"] is True
