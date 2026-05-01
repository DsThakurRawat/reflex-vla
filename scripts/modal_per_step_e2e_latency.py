"""Modal A100: per-step E2E chunk-latency bench (gate 5).

Question: through the FULL Pi05DecomposedInference pipeline (vlm_prefix
+ expert + Python glue), does the per-step path add unacceptable
chunk-latency overhead vs the baked path?

Gate 4 measured per-step ORT-call overhead in isolation (+13.3% via
IOBinding). Gate 5 measures the same thing through production runtime —
catches any glue overhead, IOBinding integration bugs, or vlm_prefix
interaction issues that gate 4 missed.

Methodology:
  - Construct two Pi05DecomposedInference instances pointing at the
    gate-3 v4 ONNX exports on the volume (baked + per-step)
  - cache_level='none' to ensure every call runs full VLM + expert
  - Generate dummy preprocessed inputs once (3 images + 3 masks +
    lang tokens + lang masks + noise)
  - Warm up each instance, then bench N=50 chunks each
  - Report median + p95 + p99 chunk latency, overhead, p99 ratio

Acceptance (matches gate 4):
  - per-step median chunk latency ≤ 1.20 × baked
  - per-step p99 ≤ 1.30 × baked

Cost: ~$1.50 Modal (A100-80GB, ~5 min wall once cached).

Usage:
    HF_TOKEN=<token> modal run --detach scripts/modal_per_step_e2e_latency.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("reflex-per-step-e2e-latency")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


# Same _BUST as gate 4 v4 → image cache hit.
_BUST = "20260501-per-step-gates-curand-eager-dlopen"

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20,<1.24",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-curand-cu12>=10.0,<12.0",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cufft-cu12>=11.0,<12.0",
    )
    .add_local_dir(
        os.path.join(REPO_ROOT, "src"),
        remote_path="/root/reflex-vla/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(os.path.join(REPO_ROOT, "pyproject.toml"), remote_path="/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file(os.path.join(REPO_ROOT, "README.md"), remote_path="/root/reflex-vla/README.md", copy=True)
    .add_local_file(os.path.join(REPO_ROOT, "LICENSE"), remote_path="/root/reflex-vla/LICENSE", copy=True)
    .run_commands(
        f'echo "build_bust={_BUST}"',
        'pip install -e "/root/reflex-vla[monolithic]"',
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


N_WARMUP = 10
N_BENCH = 200  # bumped from 50 for tighter p99 + median stability


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def e2e_bench() -> dict:
    import logging
    import time
    import numpy as np

    from reflex.runtime.pi05_decomposed_server import Pi05DecomposedInference

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    baked_dir = Path(ONNX_OUT) / "per_step_parity/pi05_teacher_n10_baked"
    per_step_dir = Path(ONNX_OUT) / "per_step_parity/pi05_teacher_n10_per_step"

    if not (baked_dir / "expert_denoise.onnx").exists():
        raise FileNotFoundError(f"Baked export missing at {baked_dir}. Run gate 3 first.")
    if not (per_step_dir / "expert_denoise.onnx").exists():
        raise FileNotFoundError(f"Per-step export missing at {per_step_dir}. Run gate 3 first.")

    log.info("Building Pi05DecomposedInference (baked) from %s", baked_dir)
    inf_baked = Pi05DecomposedInference(export_dir=str(baked_dir), cache_level="none")
    log.info("Building Pi05DecomposedInference (per-step) from %s", per_step_dir)
    inf_per_step = Pi05DecomposedInference(export_dir=str(per_step_dir), cache_level="none")

    # Dummy preprocessed inputs matching predict_action_chunk's signature.
    # pi0.5 LIBERO finetune: 3 cameras × 224×224, 200 lang tokens, chunk=50, action_dim=32.
    rng = np.random.default_rng(seed=42)
    B = 1
    img_h, img_w = 224, 224
    chunk = 50
    action_dim = 32
    lang_len = 200

    def make_inputs(seed: int):
        r = np.random.default_rng(seed)
        return {
            "img_base": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "img_wrist_l": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "img_wrist_r": r.uniform(0.0, 1.0, size=(B, 3, img_h, img_w)).astype(np.float32),
            "mask_base": np.ones((B,), dtype=bool),
            "mask_wrist_l": np.ones((B,), dtype=bool),
            "mask_wrist_r": np.ones((B,), dtype=bool),
            "lang_tokens": r.integers(0, 100, size=(B, lang_len), dtype=np.int64),
            "lang_masks": np.ones((B, lang_len), dtype=bool),
            "noise": r.standard_normal(size=(B, chunk, action_dim)).astype(np.float32),
        }

    # Phase-level timing helper. Calls _get_or_run_prefix + _run_expert
    # directly to attribute wall-time to each phase. Mirrors the public
    # predict_action_chunk path for cache_level='none' (skips episode/action
    # cache short-circuits, runs both phases every call).
    def time_one_chunk(inf, inputs):
        # Phase 1: vlm_prefix (the 38 past_kv + prefix_pad_masks computation)
        t0 = time.perf_counter()
        past_kv, prefix_pad = inf._get_or_run_prefix(
            img_base=inputs["img_base"],
            img_wrist_l=inputs["img_wrist_l"],
            img_wrist_r=inputs["img_wrist_r"],
            mask_base=inputs["mask_base"],
            mask_wrist_l=inputs["mask_wrist_l"],
            mask_wrist_r=inputs["mask_wrist_r"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            image_phashes=(b"x" * 8, b"x" * 8, b"x" * 8),  # cache disabled at level=none
            lang_hash="x",
        )
        t_vlm = (time.perf_counter() - t0) * 1000

        # Phase 2: expert (the per-step Euler loop or baked single call)
        expert_feed_base = {nm: past_kv[i] for i, nm in enumerate(inf._past_kv_names)}
        expert_feed_base["prefix_pad_masks"] = prefix_pad
        t1 = time.perf_counter()
        _ = inf._run_expert(expert_feed_base, inputs["noise"])
        t_expert = (time.perf_counter() - t1) * 1000

        return t_vlm + t_expert, t_vlm, t_expert

    log.info("Warming up baked (%d iters)", N_WARMUP)
    for i in range(N_WARMUP):
        _ = inf_baked.predict_action_chunk(**make_inputs(seed=i))

    log.info("Benching baked (N=%d) with phase breakdown", N_BENCH)
    baked_times, baked_vlm, baked_expert = [], [], []
    for i in range(N_BENCH):
        inputs = make_inputs(seed=1000 + i)
        total, vlm, exp = time_one_chunk(inf_baked, inputs)
        baked_times.append(total)
        baked_vlm.append(vlm)
        baked_expert.append(exp)

    log.info("Warming up per-step (%d iters)", N_WARMUP)
    for i in range(N_WARMUP):
        _ = inf_per_step.predict_action_chunk(**make_inputs(seed=i))

    log.info("Benching per-step (N=%d) with phase breakdown", N_BENCH)
    per_step_times, per_step_vlm, per_step_expert = [], [], []
    for i in range(N_BENCH):
        inputs = make_inputs(seed=2000 + i)
        total, vlm, exp = time_one_chunk(inf_per_step, inputs)
        per_step_times.append(total)
        per_step_vlm.append(vlm)
        per_step_expert.append(exp)

    def stats(arr):
        a = np.array(arr)
        return {
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(np.mean(a)),
            "std_ms": float(np.std(a)),
            "min_ms": float(np.min(a)),
            "max_ms": float(np.max(a)),
        }

    baked_stats = stats(baked_times)
    per_step_stats = stats(per_step_times)
    baked_vlm_stats = stats(baked_vlm)
    baked_expert_stats = stats(baked_expert)
    per_step_vlm_stats = stats(per_step_vlm)
    per_step_expert_stats = stats(per_step_expert)
    median_pct = (per_step_stats["median_ms"] / baked_stats["median_ms"]) - 1.0
    p99_ratio = per_step_stats["p99_ms"] / baked_stats["p99_ms"]
    passes_median = median_pct <= 0.20
    passes_p99 = p99_ratio <= 1.30
    passes_overall = passes_median and passes_p99

    log.info("--- E2E ---")
    log.info("BAKED      median=%.2fms p99=%.2fms", baked_stats["median_ms"], baked_stats["p99_ms"])
    log.info("PER-STEP   median=%.2fms p99=%.2fms", per_step_stats["median_ms"], per_step_stats["p99_ms"])
    log.info("Overhead: median %+.1f%% / p99 %.2fx — gate %s",
             median_pct * 100, p99_ratio, "PASS" if passes_overall else "FAIL")
    log.info("--- VLM PREFIX phase (should be ~equal across baked/per-step) ---")
    log.info("BAKED      vlm median=%.2fms p99=%.2fms", baked_vlm_stats["median_ms"], baked_vlm_stats["p99_ms"])
    log.info("PER-STEP   vlm median=%.2fms p99=%.2fms", per_step_vlm_stats["median_ms"], per_step_vlm_stats["p99_ms"])
    log.info("--- EXPERT phase ---")
    log.info("BAKED      exp median=%.2fms p99=%.2fms", baked_expert_stats["median_ms"], baked_expert_stats["p99_ms"])
    log.info("PER-STEP   exp median=%.2fms p99=%.2fms", per_step_expert_stats["median_ms"], per_step_expert_stats["p99_ms"])
    expert_median_pct = (per_step_expert_stats["median_ms"] / baked_expert_stats["median_ms"]) - 1.0
    log.info("Expert overhead median: %+.1f%%", expert_median_pct * 100)

    return {
        "n_warmup": N_WARMUP,
        "n_bench": N_BENCH,
        "baked": baked_stats,
        "per_step": per_step_stats,
        "baked_vlm": baked_vlm_stats,
        "baked_expert": baked_expert_stats,
        "per_step_vlm": per_step_vlm_stats,
        "per_step_expert": per_step_expert_stats,
        "median_overhead_pct": median_pct,
        "p99_ratio": p99_ratio,
        "expert_median_overhead_pct": expert_median_pct,
        "passes_median_gate": passes_median,
        "passes_p99_gate": passes_p99,
        "passes_overall": passes_overall,
        "thresholds": {"median_overhead_pct_max": 0.20, "p99_ratio_max": 1.30},
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Per-step E2E chunk-latency bench (gate 5)")
    print("=" * 60)
    result = e2e_bench.remote()

    receipt_path = Path(REPO_ROOT) / ".." / "reflex_context" / "per_step_e2e_latency_last_run.json"
    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2))
    print(f"\nReceipt: {receipt_path}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    b, p = result["baked"], result["per_step"]
    print(f"  baked    median={b['median_ms']:.2f}ms  p95={b['p95_ms']:.2f}ms  p99={b['p99_ms']:.2f}ms")
    print(f"  per-step median={p['median_ms']:.2f}ms  p95={p['p95_ms']:.2f}ms  p99={p['p99_ms']:.2f}ms")
    print(f"\n  overhead median {result['median_overhead_pct'] * 100:+.1f}%  p99 {result['p99_ratio']:.2f}x")
    print(f"  Overall: {'PASS' if result['passes_overall'] else 'FAIL'} "
          f"(thresholds: median≤+20%, p99≤1.30x)")
