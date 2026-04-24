"""Unit tests for the A2C2 correction head + input encoders.

B.4 transfer-validation gate — head module tests. Pure-PyTorch; no Modal,
no LIBERO. ~80 ms total.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from reflex.correction.a2c2_head import (
    A2C2Config,
    A2C2Head,
    build_a2c2_input,
    chunk_pos_encoding,
    latency_log_encoding,
)


class TestA2C2Config:
    def test_input_dim_sums_components(self):
        cfg = A2C2Config(action_dim=7, obs_dim=256, chunk_pos_dim=32, latency_dim=32)
        assert cfg.input_dim == 7 + 256 + 32 + 32

    def test_estimated_param_count_matches_layer_arithmetic(self):
        cfg = A2C2Config(
            action_dim=7, obs_dim=256, chunk_pos_dim=32, latency_dim=32,
            hidden_dims=(128, 96),
        )
        # 327 -> 128: 327*128 + 128 = 42_x; 128 -> 96: 128*96 + 96; 96 -> 7: 96*7 + 7
        expected = (327 * 128 + 128) + (128 * 96 + 96) + (96 * 7 + 7)
        assert cfg.estimated_param_count() == expected

    def test_size_under_paper_envelope_default_config(self):
        # Paper says ~100 KB; we accept anything under 200 KB FP16 (50% margin)
        cfg = A2C2Config()
        assert cfg.estimated_size_bytes(fp16=True) < 200 * 1024


class TestEncoders:
    def test_chunk_pos_encoding_scalar_shape(self):
        enc = chunk_pos_encoding(7, dim=32, max_idx=50)
        assert enc.shape == (32,)
        assert enc.dtype == np.float32

    def test_chunk_pos_encoding_array_shape(self):
        idx = np.array([0, 5, 10, 49])
        enc = chunk_pos_encoding(idx, dim=16, max_idx=50)
        assert enc.shape == (4, 16)

    def test_chunk_pos_encoding_in_range(self):
        enc = chunk_pos_encoding(np.arange(50), dim=32)
        assert np.all(np.abs(enc) <= 1.0 + 1e-6)

    def test_chunk_pos_encoding_odd_dim_raises(self):
        with pytest.raises(ValueError, match="must be even"):
            chunk_pos_encoding(0, dim=33)

    def test_latency_log_encoding_clamping(self):
        # Below lo and above hi should clamp into range
        below = latency_log_encoding(0.001, dim=16, lo=1.0, hi=1000.0)
        at_lo = latency_log_encoding(1.0, dim=16, lo=1.0, hi=1000.0)
        np.testing.assert_allclose(below, at_lo, atol=1e-6)
        above = latency_log_encoding(1e6, dim=16, lo=1.0, hi=1000.0)
        at_hi = latency_log_encoding(1000.0, dim=16, lo=1.0, hi=1000.0)
        np.testing.assert_allclose(above, at_hi, atol=1e-6)

    def test_latency_log_encoding_monotonicity_proxy(self):
        # Encodings at distinct latencies should differ (loose; sinusoids can collide)
        a = latency_log_encoding(20.0, dim=32)
        b = latency_log_encoding(200.0, dim=32)
        assert not np.allclose(a, b)


class TestBuildInput:
    def test_single_sample_shape(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, chunk_pos_dim=8, latency_dim=8)
        x = build_a2c2_input(
            base_action=np.zeros(7, dtype=np.float32),
            obs_features=np.zeros(10, dtype=np.float32),
            chunk_idx=3,
            latency_ms=50.0,
            cfg=cfg,
        )
        assert x.shape == (cfg.input_dim,)

    def test_batch_shape(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, chunk_pos_dim=8, latency_dim=8)
        B = 4
        x = build_a2c2_input(
            base_action=np.zeros((B, 7), dtype=np.float32),
            obs_features=np.zeros((B, 10), dtype=np.float32),
            chunk_idx=np.array([0, 1, 2, 3]),
            latency_ms=np.array([20.0, 40.0, 80.0, 100.0]),
            cfg=cfg,
        )
        assert x.shape == (B, cfg.input_dim)

    def test_action_dim_mismatch_raises(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, chunk_pos_dim=8, latency_dim=8)
        with pytest.raises(ValueError, match="action_dim"):
            build_a2c2_input(np.zeros(6), np.zeros(10), 0, 50.0, cfg)

    def test_obs_dim_mismatch_raises(self):
        cfg = A2C2Config(action_dim=7, obs_dim=10, chunk_pos_dim=8, latency_dim=8)
        with pytest.raises(ValueError, match="obs_dim"):
            build_a2c2_input(np.zeros(7), np.zeros(11), 0, 50.0, cfg)


class TestA2C2Head:
    def test_param_count_matches_estimate(self):
        cfg = A2C2Config()
        head = A2C2Head(cfg)
        assert head.param_count() == cfg.estimated_param_count()

    def test_size_under_150kb_fp16_default_config(self):
        cfg = A2C2Config()
        head = A2C2Head(cfg)
        assert head.size_bytes(fp16=True) < 150 * 1024

    def test_output_shape_matches_action_dim(self):
        cfg = A2C2Config(action_dim=6, obs_dim=64, chunk_pos_dim=16, latency_dim=16)
        head = A2C2Head(cfg)
        x = torch.zeros(1, cfg.input_dim)
        out = head(x)
        assert out.shape == (1, cfg.action_dim)

    def test_zero_init_residual_returns_zero(self):
        # Final layer is zero-init so an untrained head produces correction=0
        cfg = A2C2Config()
        head = A2C2Head(cfg)
        x = torch.randn(2, cfg.input_dim)
        out = head(x)
        np.testing.assert_array_equal(out.detach().numpy(), np.zeros((2, cfg.action_dim), dtype=np.float32))

    def test_forward_deterministic(self):
        cfg = A2C2Config()
        head = A2C2Head(cfg)
        x = torch.randn(4, cfg.input_dim)
        head.eval()
        a = head(x).detach().numpy()
        b = head(x).detach().numpy()
        np.testing.assert_array_equal(a, b)

    def test_correct_helper_returns_base_plus_zero_correction(self):
        cfg = A2C2Config(action_dim=6, obs_dim=8, chunk_pos_dim=8, latency_dim=8)
        head = A2C2Head(cfg)
        base = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.0], dtype=np.float32)
        obs = np.zeros(8, dtype=np.float32)
        out = head.correct(base, obs, chunk_idx=2, latency_ms=50.0)
        np.testing.assert_allclose(out, base, atol=1e-6)

    def test_correct_helper_batch_shape(self):
        cfg = A2C2Config(action_dim=7, obs_dim=8, chunk_pos_dim=8, latency_dim=8)
        head = A2C2Head(cfg)
        B = 3
        base = np.zeros((B, 7), dtype=np.float32)
        obs = np.zeros((B, 8), dtype=np.float32)
        out = head.correct(base, obs, np.array([0, 1, 2]), np.array([20.0, 50.0, 80.0]))
        assert out.shape == (B, 7)
