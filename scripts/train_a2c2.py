"""Train an A2C2 correction head on offline action chunks.

B.4 A2C2 transfer-validation gate — training stream (Stream 1 in the plan).

Usage (local synthetic data — for plumbing verification, no Modal cost):
    python scripts/train_a2c2.py \\
        --data synthetic \\
        --out outputs/a2c2_synthetic.pt \\
        --epochs 3 \\
        --batch-size 32

Usage (real LIBERO traces from `reflex serve --record <dir>`):
    python scripts/train_a2c2.py \\
        --data 'data/libero_traces/*.jsonl' \\
        --out outputs/a2c2_lerobot_trained.pt \\
        --epochs 5 \\
        --batch-size 32 \\
        --action-dim 7 \\
        --obs-dim 256

Modal launch wrapper (TBD as `scripts/modal_train_a2c2.py`) reuses this script
inside a Modal function with an A10G GPU.

Inputs assumed in each JSONL record (record/replay v1 schema):
    - actions: list[list[float]]  — chunk of (chunk_size, action_dim)
    - state: list[float]          — robot state at request time
    - latency_ms: float           — observed inference latency (with --inject)

Currently supports only 'state' as the obs_features source. To use a richer
obs_features (e.g., VLM prefix output), pre-process the JSONL into a separate
.npz before calling this script — keeps the script obs-agnostic.

Loss: MSE between (base_action + correction) and the recorded executed action.
For the synthetic-data path we use the recorded action as ground truth (residual
target = 0); for real data the same applies because the base policy IS what the
robot saw, and we want corrections to approach zero unless the head can
discover residuals worth predicting.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("a2c2.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[dict] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(rec)
    return out


def _flatten_traces(records: list[dict], action_dim: int, obs_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw JSONL records into per-step training tensors.

    Returns (base_actions, obs_features, chunk_indices, latency_ms_per_step).
    Each row corresponds to one (chunk_idx, action) pair.
    """
    base_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    chunk_idx_rows: list[int] = []
    latency_rows: list[float] = []
    for rec in records:
        if rec.get("kind") != "request" and rec.get("type") != "request":
            actions = rec.get("actions") or rec.get("response", {}).get("actions")
            state = rec.get("state") or rec.get("request", {}).get("state")
        else:
            actions = rec.get("actions")
            state = rec.get("state")
        if not actions or not state:
            continue
        latency_ms = float(rec.get("latency_ms") or rec.get("latency_total_ms") or 0.0)
        injected = float(rec.get("injected_latency_ms") or 0.0)
        observed_latency = latency_ms + injected
        for chunk_idx, action in enumerate(actions):
            a = np.asarray(action, dtype=np.float32)
            if a.shape[0] != action_dim:
                continue
            obs_pad = np.zeros(obs_dim, dtype=np.float32)
            s = np.asarray(state[: obs_dim], dtype=np.float32)
            obs_pad[: s.shape[0]] = s
            base_rows.append(a)
            obs_rows.append(obs_pad)
            chunk_idx_rows.append(chunk_idx)
            latency_rows.append(observed_latency)
    return (
        np.asarray(base_rows, dtype=np.float32),
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(chunk_idx_rows, dtype=np.int64),
        np.asarray(latency_rows, dtype=np.float32),
    )


def _generate_synthetic(
    n_episodes: int,
    action_dim: int,
    obs_dim: int,
    chunk_size: int = 16,
    seed: int = 0,
    target_noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic traces for plumbing verification. Deterministic given seed.

    Returns base, obs, chunk_idx, latency, target_residual.
    target_residual simulates the "true correction" the head should learn —
    structured noise dependent on latency + chunk_idx so the head has a
    learnable signal (not just random labels).
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_episodes * chunk_size, action_dim)).astype(np.float32) * 0.3
    obs = rng.standard_normal((n_episodes * chunk_size, obs_dim)).astype(np.float32) * 0.5
    chunk_idx = np.tile(np.arange(chunk_size, dtype=np.int64), n_episodes)
    latency = rng.uniform(20, 80, size=n_episodes * chunk_size).astype(np.float32)
    # Latency-conditioned residual: bigger correction at higher latency, modulated
    # by chunk_idx (later chunks need more correction = staleness compounds).
    lat_norm = (latency - 20) / 60.0
    chunk_norm = chunk_idx.astype(np.float32) / max(chunk_size, 1)
    scale = (lat_norm * chunk_norm)[:, None]  # (N, 1)
    target_residual = (
        rng.standard_normal((n_episodes * chunk_size, action_dim)).astype(np.float32)
        * target_noise_std
        * scale
    )
    return base, obs, chunk_idx, latency, target_residual


def _gather_paths(spec: str) -> list[Path]:
    if spec == "synthetic":
        return []
    p = Path(spec)
    if p.is_dir():
        return sorted(list(p.glob("*.jsonl")) + list(p.glob("*.jsonl.gz")))
    if any(ch in spec for ch in "*?["):
        from glob import glob
        return [Path(s) for s in sorted(glob(spec))]
    return [p]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an A2C2 correction head.")
    parser.add_argument("--data", required=True, help="JSONL glob/dir/file or 'synthetic'")
    parser.add_argument("--out", required=True, help="Path to write checkpoint .pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--obs-dim", type=int, default=256)
    parser.add_argument("--chunk-pos-dim", type=int, default=32)
    parser.add_argument("--latency-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", default="128,96", help="Comma-separated hidden dims")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--n-synthetic-episodes", type=int, default=200)
    parser.add_argument("--target-noise-std", type=float, default=0.05,
                        help="Synthetic-mode only: stddev of structured target residuals "
                             "(latency-and-chunk-idx-conditioned). 0 = degenerate gate path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default="", help="Optional JSON path for training metrics")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("torch is required; install reflex-vla[correction]")
        return 2

    from reflex.correction.a2c2_head import A2C2Config, A2C2Head, build_a2c2_input

    cfg = A2C2Config(
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        chunk_pos_dim=args.chunk_pos_dim,
        latency_dim=args.latency_dim,
        hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",") if x),
    )
    logger.info(
        "A2C2Config: input=%d hidden=%s output=%d → ~%.1f KB FP16",
        cfg.input_dim, cfg.hidden_dims, cfg.action_dim, cfg.estimated_size_bytes() / 1024,
    )

    target_residual: np.ndarray | None = None
    if args.data == "synthetic":
        logger.info("Generating synthetic data: %d episodes (target_noise_std=%.3f)",
                    args.n_synthetic_episodes, args.target_noise_std)
        base, obs, chunk_idx, latency, target_residual = _generate_synthetic(
            args.n_synthetic_episodes, cfg.action_dim, cfg.obs_dim,
            seed=args.seed, target_noise_std=args.target_noise_std,
        )
    else:
        paths = _gather_paths(args.data)
        if not paths:
            logger.error("No JSONL files found at %s", args.data)
            return 2
        all_records: list[dict] = []
        for p in paths:
            recs = _load_jsonl(p)
            logger.info("loaded %s: %d records", p.name, len(recs))
            all_records.extend(recs)
        base, obs, chunk_idx, latency = _flatten_traces(all_records, cfg.action_dim, cfg.obs_dim)
        if base.size == 0:
            logger.error("No usable training rows extracted from JSONL")
            return 2
        # Real-data target: zero residual (no separate executed_action recorded).
        # Magnitude-of-correction is the proxy. Update record/replay v2 to add
        # executed_action if a sharper gate is needed.
        target_residual = np.zeros_like(base)
    logger.info("Training tensor: base=%s obs=%s chunk=%s latency=%s",
                base.shape, obs.shape, chunk_idx.shape, latency.shape)

    rng = np.random.default_rng(args.seed)
    n = base.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    head = A2C2Head(cfg)
    logger.info("A2C2Head params: %d → ~%.1f KB FP16", head.param_count(), head.size_bytes() / 1024)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def _build_batch(idx: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor"]:
        x = build_a2c2_input(base[idx], obs[idx], chunk_idx[idx], latency[idx], cfg)
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(target_residual[idx]).float()  # type: ignore[index]
        return x_t, y_t

    metrics: dict = {"epochs": [], "config": {
        "action_dim": cfg.action_dim, "obs_dim": cfg.obs_dim,
        "hidden_dims": list(cfg.hidden_dims), "params": head.param_count(),
    }}

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        head.train()
        rng.shuffle(train_idx)
        train_losses: list[float] = []
        for batch_start in range(0, train_idx.shape[0], args.batch_size):
            batch = train_idx[batch_start : batch_start + args.batch_size]
            x_t, y_t = _build_batch(batch)
            pred = head(x_t)
            loss = loss_fn(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            x_v, y_v = _build_batch(val_idx)
            val_loss = float(loss_fn(head(x_v), y_v).item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_metric = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        metrics["epochs"].append(epoch_metric)
        logger.info("epoch %d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)
    metrics["wall_time_s"] = time.perf_counter() - t0
    metrics["final_val_mse"] = metrics["epochs"][-1]["val_loss"] if metrics["epochs"] else float("nan")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "config": vars(args), "metrics": metrics}, out_path)
    logger.info("checkpoint written: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
        logger.info("metrics written: %s", args.metrics_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
