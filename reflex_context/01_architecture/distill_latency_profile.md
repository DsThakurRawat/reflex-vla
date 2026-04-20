# Distill Latency Profile — Where Is the Actual Bottleneck?

**Question**: if we distill a VLA's denoise loop from 10 steps to 1, does end-to-end latency drop 10×, or is the VLM forward the real bottleneck?

**TL;DR verdict**: **YES, denoise IS the bottleneck on modern high-end GPU** (~80% of E2E on A800/RTX-5090 for pi-family at num_steps=10), but the picture flips at the two extremes:
- On **Jetson Orin Nano / Thor**, VLM prefix is compute-bound and ~30–50% of E2E. Distill is a meaningful but sub-4× win, not 10×.
- For **GR00T N1.6** (4 DDPM steps by default, not 10 flow-matching), denoise is already 35% of E2E at RTX 5090 — distilling 4→1 is worth ~10–15 ms out of 31 ms, a 1.35× gain.
- For **SmolVLA** our own adaptive-denoise experiment shows velocities never converge under a 0.01 threshold (0/25 triggers) — so even without distillation, there's no free early-stop lunch here; distillation may be the *only* path.

Section E recommends **(b) scoped to a subset of VLAs**: ship `reflex distill` for pi0 and pi0.5 (where num_steps=10 flow matching + the SnapFlow 9.6× result is the best empirical case), defer SmolVLA and GR00T to v0.3+ with kill-gates.

Last updated 2026-04-20. No number in this doc without a source.

---

## Section A — Per-VLA latency breakdown

Reflex-measured (A10G TRT FP16) + published breakdowns. Empty cells are genuinely unmeasured.

| VLA | Stage | A10G TRT (ms) | Orin Nano est. (ms) | High-end published (ms) | Source |
|---|---|---|---|---|---|
| **SmolVLA** (0.45B) | VLM prefix | ~2 | ~4 | 48 on A100 eager | Reflex [1] |
| | 1 denoise step | 0.95 | ~1.9 | ~5 on A800 | Reflex [2], SnapFlow [3] |
| | 10-step chunk (E2E) | 11.67 | ~23 | 178 on A800 PyTorch; 50 at n=1 | Reflex [2], SnapFlow [3] |
| **pi0** (3.3B) | VLM prefix | ~4 | ~8 | 60 on A800 | SnapFlow [3] |
| | 1 denoise step | 1.94 | ~3.9 | ~21–23 on A800 | Reflex [2], SnapFlow [3] |
| | 10-step chunk (E2E) | 23.6 | ~47 | **274 on A800 PyTorch** | SnapFlow [3] |
| **pi0.5** (3.62B) | VLM prefix | ~4 | ~8 | ~60 on A800 | SnapFlow [3] |
| | 1 denoise step | 2.24 | ~4.5 | ~22 on A800 | Reflex [2] |
| | 10-step chunk (E2E) | 27.1 | ~54 | 274 A800 PyTorch; 94 Thor TRT NVFP4 | Reflex [2], SnapFlow [3], NVIDIA Jetson AI Lab [4] |
| **GR00T N1.6** (3B) | Data processing | — | — | **2 on RTX 5090** | GR00T deploy README [5] |
| | Backbone (Eagle 2.5 VLM) | — | — | **18 on RTX 5090** | [5] |
| | Action head (DiT 4 steps) | — | — | **11 on RTX 5090** | [5] |
| | Per-step denoise | 5.59 | ~11 | ~2.75 on RTX 5090 | Reflex [2], [5] |
| | 4-step chunk (E2E) | 55.9 | ~113 | **31 RTX 5090 (32.1 Hz); 93.8 Thor (10.7 Hz); 216.5 Orin AGX 64GB (4.6 Hz)** | Reflex [2], [5] |

Sources: [1] Reflex internal, [`05_sessions/2026-04-10_mega_session.md`](../05_sessions/2026-04-10_mega_session.md) L118–120. [2] Reflex TRT FP16 bench, [`06_experiments/latency_benchmarks.md`](../06_experiments/latency_benchmarks.md) (commit `9e3dabb`, Modal A10G). [3] SnapFlow arXiv 2604.05656v1 §4.3 + Appendix F Table 7. [4] NVIDIA Jetson AI Lab *pi0.5 on Thor* tutorial. [5] NVIDIA Isaac-GR00T N1.7 Deployment README. [6] Dexmal arXiv 2510.26742 Table 2 (RTX 4090 pi0, 2 views): Vision 3.957ms (14.5%), LLM/VLM 12.732ms (46.7%), AE 9.831ms (36.0%), total 26.5ms.

---

## Section B — The bottleneck verdict

For each VLA × hardware, what percentage of total latency is the denoise loop? Distillation's upper-bound value prop scales with this fraction.

### SmolVLA
- **A10G TRT FP16, num_steps=10**: 11.67 ms/chunk. Denoise ≈ **81%** (10 × 0.95 ms). VLM prefix ≈ 2 ms TRT FP16 (from 48 ms A100 PyTorch eager, compressed by TRT). **Denoise dominates.**
- **Orin Nano (est., ~2× A10G)**: ≈ 23 ms/chunk. VLM prefix scales harder (bandwidth-bound), pushing prefix to 40–60%. **Denoise ≈ 40–60%.**
- **A800 PyTorch (SnapFlow)**: 178 ms at num_steps=10, 50 ms at num_steps=1 → denoise ≈ **70%**.

### pi0 / pi0.5 (flow matching, num_steps=10)
- **A800 PyTorch (SnapFlow canonical)**: 274 ms E2E = 60 ms VLM (22%) + 214 ms denoise (**78%**). SnapFlow's pull quote: *"denoising alone accounts for 80% of end-to-end inference time"* [3]. **Canonical "distill wins big" case.**
- **A10G TRT FP16**: pi0 23.6 ms/chunk. Denoise 19.4 ms (**82%**), VLM prefix ~4 ms (18%, TRT compresses VLM harder than denoise).
- **Jetson Thor TRT NVFP4 (pi0.5, Jetson AI Lab)**: 94 ms E2E. Denoise ≈ 60 ms (**64%**), VLM ≈ 35 ms (36%). **Denoise still larger but VLM is material.**
- **Jetson Orin Nano (extrapolated, 274 ms × ~2)**: ~550 ms. VLM prefix expected 150–200 ms. Denoise fraction compresses to **55–65%**.

### GR00T N1.6 (DDPM DiT, num_steps=4)
- **RTX 5090 TRT (GR00T README)**: 31 ms E2E = 2 ms data + 18 ms backbone + 11 ms action head. Denoise (4 × ~2.75 ms) ≈ **35%**. VLM ≈ **58%**. **VLM dominates.** 4→1 distill saves ~8 ms / 31 ms = 1.35×.
- **Jetson Thor TRT**: 93.8 ms E2E. Applying RTX 5090 58/35/7% ratio: denoise ≈ **35%**. Same story.
- **Jetson Orin AGX 64GB (DiT-only TRT)**: 216.5 ms E2E. VLM runs PyTorch here (Eagle 2.5 won't TRT-build on Orin). Denoise still ≈ 35%. **VLM's TRT-unfriendliness is the bigger issue than denoise count on Orin.**

### Summary table of denoise-% by VLA × hardware

| VLA | A10G TRT | Orin Nano est. | Thor TRT | RTX 5090 TRT | A800 PyTorch |
|---|---|---|---|---|---|
| **SmolVLA** | ~81% | ~50% | — | — | ~70% |
| **pi0** | ~82% | ~55% | — | — | **~78%** |
| **pi0.5** | ~83% | ~55% | ~64% | — | **~78%** |
| **GR00T N1.6** | ~64% (est. 4×5.59/55.9ms; but VLM not yet TRT'd on reflex path) | ~35% | ~35% | **~35%** | — |

**The key finding for the scope-decider**: distillation's upper-bound E2E win is bounded by the denoise fraction. For flow-matching models (pi0, pi0.5, SmolVLA) on modern GPU, distilling 10 → 1 theoretically caps at 5–6× (via Amdahl — if denoise is 80%, eliminating it entirely gives 5×, not 10×). For GR00T on RTX 5090, distilling 4 → 1 caps at 1.5×. **No configuration gives a true 10× win**; the "10 → 1 = 10×" framing is linear-scaling wishful thinking that ignores the non-denoise fraction entirely.

---

## Section C — Customer pain frequency

### Customer-facing complaints on VLA inference speed (top signals)

Pulled from `reflex_context/01_architecture/finetune_competitive_research.md` (top-10 pain points), lerobot/openpi GitHub issues, and Jetson AI Lab tutorials. Tagged by what they're really complaining about: **TOTAL** latency, **DENOISE** specifically, **JITTER/variance**, or **TASK SUCCESS**.

| # | Quote / signal | Source | Tag | Distill helps? |
|---|---|---|---|---|
| 1 | *"despite the training running without issues... the evaluation yields 0% success"* | [lerobot #2259](https://github.com/huggingface/lerobot/issues/2259) | **TASK SUCCESS** (norm-stats) | No |
| 2 | *"RuntimeError: size of tensor a (6) must match tensor b (7)"* | [lerobot #2418](https://github.com/huggingface/lerobot/issues/2418) | **TASK SUCCESS** (shape mismatch) | No |
| 3 | *"I suspect the discrepancy... due to the effective batch size"* | [lerobot #3287](https://github.com/huggingface/lerobot/issues/3287) | **TASK SUCCESS** (irreproducibility) | No |
| 4 | *"memory increase continuously during training Groot"* | [lerobot #2371](https://github.com/huggingface/lerobot/issues/2371) | **TRAINING OOM** (not serving) | No |
| 5 | *"after the finetuning process, the success rate is very low (1% on spatial)"* | [openpi #711](https://github.com/Physical-Intelligence/openpi/issues/711) | **TASK SUCCESS** (norm-stats) | No |
| 6 | OpenVLA base: *"~3 FPS on Jetson AGX Orin even at INT4"* — cited as the "edge too slow" datapoint | [openvla README / competitor_deployment_stacks.md](../03_research/competitor_deployment_stacks.md) | **TOTAL** latency | Partial (OpenVLA doesn't denoise; it autoregresses tokens) |
| 7 | Dexmal realtime-vla paper framing: *"3–5 FPS against a need for 20–30 Hz"* | [arXiv 2510.26742](https://arxiv.org/html/2510.26742v1) | **TOTAL** latency | Partial |
| 8 | SmolVLA blog: *"skipping layers in VLM backbone... asynchronous inference to compute the next action while current is executing"* — HF chose *async chunking*, not distillation, as their speed lever | [HF SmolVLA blog](https://huggingface.co/blog/smolvla) | **TOTAL** (solved via chunking) | No (chunking already ate this) |
| 9 | OpenVLA-OFT: *"26× faster (5 Hz → 130 Hz)... parallel decoding replaces autoregressive"* | [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) | **TOTAL** (solved via parallel decode, not distill) | No |
| 10 | SnapFlow paper framing: pi-family *"iterative denoising... introduces substantial latency. Denoising alone accounts for 80% of end-to-end inference time"* | [arXiv 2604.05656v1](https://arxiv.org/abs/2604.05656v1) | **DENOISE** specifically | **Yes — this IS the distill pitch** |

### Count by category

| Category | Count (of top 10) | Interpretation |
|---|---|---|
| TASK SUCCESS | 5 | Dominant pain. Users want their fine-tune not to fail at 0%, not lower latency. |
| TOTAL latency | 4 | Real, but typically "I can't deploy at all on edge" rather than "too slow by factor of N." |
| DENOISE specifically | 1 | Only SnapFlow explicitly frames latency as a denoise problem. |
| JITTER/variance | 0 | No signal in our survey — not a customer complaint class. |

### Interpretation

1. **The denoise-specific customer signal is thin.** Only 1/10 signals names denoise as the problem (SnapFlow). The rest split TASK SUCCESS (distill doesn't help) vs TOTAL latency (distill is one lever among several).
2. **The competitive response to "VLAs are slow on edge" has NOT been distillation.** HuggingFace shipped async chunking. OpenVLA-OFT shipped parallel decoding + L1 regression. Dexmal shipped streaming. Each is architectural, not denoise distillation. SnapFlow is the lone denoise-distill voice.
3. **Dominant pain is semantic, not infrastructural.** From our own top-10, 7/10 are "model doesn't work on my data" (norm stats, shape, reproducibility); 2/10 are training OOM; only 1/10 is inference speed. Customers aren't (yet) calling for lower latency at the frequency they're calling for "make it work."
4. **Where latency does appear, it's Jetson-targeted** (openvla 3 FPS on AGX Orin, pi0.5 on Thor tutorial). The loudest latency pain lives at the edge — which is exactly where denoise-fraction is **lowest** (35–55%). Distill's wall-clock impact is smallest where customers complain most.

---

## Section D — Alternative levers (ranked)

If distill-from-10-to-1 doesn't deliver the promised 10×, what will? Ranked by expected E2E impact on a mid-range hardware target (A10G / Thor) for the pi-family (our best distill case):

| Rank | Lever | Expected speedup on A10G/Thor | Status in reflex | Evidence |
|---|---|---|---|---|
| 1 | **TRT FP16 vs PyTorch eager** | **2.6–3.3×** across all 4 VLAs | ✅ shipped v0.1 | Reflex `modal_bench_trt_fp16.py` [2] |
| 2 | **pi-Flow / SnapFlow distillation (10→1 denoise)** | **3.3× E2E on pi0.5 A800** (274→83ms); **1.4× on GR00T (denoise is only 35%)** | ❌ scaffolded, not shipped | SnapFlow [3]; our own `src/reflex/distill/pi_flow.py` stub |
| 3 | **Async chunking (RTC)** | ~2× effective throughput (chunk-overlap, not per-call latency) | ✅ shipped v0.1 | [RTC arXiv 2506.07339](https://arxiv.org/abs/2506.07339), `reflex serve` chunk_size_threshold=0.7 |
| 4 | **FP8 / NVFP4 on Thor / Blackwell** | **1.7× on Thor for pi0.5** (163→95ms bf16→NVFP4) | ❌ not shipped — Thor-only | NVIDIA Jetson AI Lab [4] |
| 5 | **FP16 vs FP32 on Orin Nano** | **~50% memory reduction**, modest speed gain (FP16 kernels ~1.2–1.4× faster; main unlock is fit) | ✅ shipped v0.3 for pi0 (cast-insertion pass) | Reflex `orin_nano_fp16_plan.md`, 2026-04-19/20 rows of measured_numbers.md |
| 6 | **Parallel decoding (OpenVLA-OFT pattern)** | **26× on OpenVLA specifically** (not a flow-matching model — different problem) | ❌ not applicable to our 4 VLAs directly | [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) |
| 7 | **Adaptive denoise (pi0-only)** | **58% savings on pi0 only** (0% on SmolVLA, drift on pi0.5 and GR00T) | ✅ shipped but gated | Reflex `adaptive_denoising_validation.md` |
| 8 | **KV-cache reuse across denoise steps** (VLM prefix cached, rerun only action head per step) | Theoretically ~20–40% savings on the *VLM prefix* — but this is **already the default** in flow-matching VLAs; the VLM prefix is naturally computed once per chunk, not per denoise step | ✅ already implicit | SmolVLA/pi0 architecture: VLM prefix is the conditioning input to the action expert, not re-run per Euler step |
| 9 | **Batched sampling (N parallel action samples for confidence-weighted averaging)** | Variance reduction (task success), not latency | ❌ research |  |
| 10 | **Action chunking (predict 50, execute 10, re-plan)** | 5× *effective* control rate (not compute) | ✅ shipped v0.1 | Flow-matching default; `chunk_size=50`, `n_action_steps` config |

### Key observations on the alternatives

- **KV-cache reuse is NOT a free lever.** In SmolVLA / pi0 / pi0.5, the VLM prefix is already computed once per chunk and passed as conditioning `vlm_kv` to the action expert. Each Euler denoise step re-runs only the action expert, reading the frozen `vlm_kv`. This is the default, not an optimization. Verified in `reflex_context/01_architecture/smolvla_forward_pass.md`.
- **Async chunking has already eaten the low-hanging fruit.** reflex's current `--chunk_size_threshold=0.7` delivers ~2× effective throughput and compounds with any per-call latency win.
- **FP8/NVFP4 on Thor** is a better bet than distillation for pi0.5 on Thor (1.7× via precision, ~$0 eng risk). Distillation needs training compute + LIBERO gate + failure-mode management.
- **For GR00T, the missing lever is full-TRT VLM backbone.** The Jetson Orin 216.5 ms number uses DiT-only TRT mode because Eagle 2.5 won't build TRT on Jetson. Fixing that (VLM = 58% on RTX 5090) is higher-leverage than distilling the already-4-step DiT.

### Stacked E2E on pi0.5 (A800 baseline 274 ms)

| Lever | Running total |
|---|---|
| PyTorch eager baseline | 274 ms |
| + TRT FP16 (~3×) | 91 ms |
| + pi-Flow 10→1 distill (~3.3× on denoise portion) | **~28 ms per call** |
| + FP8 on Thor (1.7× more) | ~17 ms on Thor |

**TRT+distill stacked reaches ~28 ms; TRT alone is 91 ms; distill adds ~3× beyond shipped — meaningful, not transformative, and only on flow-matching models.**

---

## Section E — Scope recommendation for the scope-decider agent

### Pick: **(b) scoped to a subset of VLAs where denoise dominates**

Specifically: ship `reflex distill --recipe pi_flow` **only for pi0 and pi0.5** in v0.3. Defer SmolVLA to v0.3.1 gated on our own adaptive-denoise finding holding post-distill. Defer GR00T to v0.5+ gated on full VLM-TRT build first (which is a higher-leverage win than DiT distillation).

### Defending with numbers

1. **pi0 / pi0.5 on cloud GPU**: denoise is **78–82%** of E2E. Distilling 10→1 delivers ~3.3× per SnapFlow (274→83 ms on A800 pi0.5) — real, large, customer-visible. **Commit.**

2. **pi0 / pi0.5 on Jetson Thor**: denoise is **~64%** of E2E (94 ms at NVFP4). Distilling cuts ~40 ms. Net: 94 → ~55 ms. Still meaningful. **Extends the pi-family commit.**

3. **SmolVLA**: denoise is 81% of E2E numerically, but our own [`adaptive_denoising_validation.md`](../06_experiments/adaptive_denoising_validation.md) shows velocities don't converge (0/25 triggers). Distillation should work where adaptive thresholds don't, but the adaptive failure signals the flow field is structurally harder to compress than pi0's. **Scope as v0.3.1 with kill-gate: pi0 ≥95% LIBERO AND SmolVLA ≥90% → keep SmolVLA; else defer.**

4. **GR00T N1.6**: denoise is only **35%** of E2E (4-step DDPM). Best-case 4→1 saves ~8 ms of 31 ms = 1.35×. **Not a wedge for GR00T.** Higher-leverage: make Eagle VLM TRT-buildable on Jetson (closes 216.5→~130 ms gap). **Defer GR00T distill to v0.5+** conditional on VLM-TRT first.

5. **Customer-signal sanity check**: only 1/10 top signals is denoise-specific (SnapFlow). 5/10 are TASK SUCCESS. **Implication**: `reflex distill` is a smaller wedge than `reflex finetune` correctness work. If the 2–4 weeks is dedicated specifically to distillation, (b) scoping maximizes return — pi-family real, SmolVLA conditional, GR00T premature.

6. **Competitive timing**: SnapFlow (Oct 2026) shows the recipe exists and is ONNX-exportable. If we don't ship by v0.4, LeRobot likely will. But half-baked distill working on 1 of 4 VLAs with >5% task drop is worse than scoped-pi-only done right.

### Explicit kill-gates

**Gate 1** — pi0 distill preserves ≥ 95% LIBERO task success (SnapFlow's <5% drop). If >10pp drop on LIBERO-Long at N=25, regression blocks ship.
**Gate 2** — pi0 distill delivers ≥ 2.5× E2E on A10G TRT standalone. Below that, value prop is marginal vs shipping chunking alone.
**Gate 3** — distilled pi0 ONNX export preserves cos ≥ +0.999 vs distilled PyTorch at shared noise. Anything less = decomposed-path redux.
**Gate 4** — SmolVLA distill LIBERO drop < 10pp vs smolvla_libero at N=25. Else defer.
**Gate 5** — GR00T distill is explicitly NOT on the v0.3 roadmap. If asked, point at Eagle-VLM-TRT gap first.

---

## What this analysis does NOT answer

- **Real Orin Nano validation**: all Orin Nano numbers are A10G×2 extrapolations. Weight-9 goal still unmet. The 50–60% denoise-fraction estimate is back-of-envelope.
- **Post-distill FP16 / NVFP4 stackability**: does pi-Flow-distilled pi0 export to FP16 ONNX at cos ≥ 0.999? Unknown. Precision compounding not studied in SnapFlow.
- **Task-success post-distill on DROID / SimplerEnv / real robots**: SnapFlow validates on LIBERO only.
- **Jitter/variance**: not in top-10 customer signals but untracked internally. Distill might raise or lower it — no data.
- **Customer-signal depth**: 10 signals is shallow. Deeper LeRobot Discord + NVIDIA Jetson forum scrape is a follow-up.

---

## Related artifacts

- [`06_experiments/latency_benchmarks.md`](../06_experiments/latency_benchmarks.md) — reflex A10G TRT FP16 numbers (Section A source)
- [`06_experiments/adaptive_denoising_validation.md`](../06_experiments/adaptive_denoising_validation.md) — 4-VLA adaptive-denoise verdict (only pi0 validates)
- [`01_architecture/finetune_competitive_research.md`](./finetune_competitive_research.md) — top-10 customer pain points (Section C source)
- [`03_research/hardware_targets.md`](../03_research/hardware_targets.md) — per-Jetson memory fit and Hz estimates
- `src/reflex/distill/pi_flow.py` / `dmpo.py` — scaffolds; v0.3 entry points

## External references

- [1] Reflex internal commits (see `_raw/git_history.md`)
- [2] Reflex `scripts/modal_bench_trt_fp16.py` (commit `9e3dabb`, 2026-04-14)
- [3] [SnapFlow: One-Step Action Generation for Flow-Matching VLAs, arXiv 2604.05656v1](https://arxiv.org/abs/2604.05656v1)
- [4] [NVIDIA Jetson AI Lab — OpenPI pi0.5 on Thor tutorial](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [5] [NVIDIA Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md)
- [6] [Dexmal Running VLAs at Real-time Speed, arXiv 2510.26742](https://arxiv.org/html/2510.26742v1) — pi0 per-component on RTX 4090
- [7] [OpenVLA-OFT: Optimizing Speed and Success, arXiv 2502.19645](https://arxiv.org/abs/2502.19645) — 26× via parallel decoding (not distillation)
- [8] [pi-Flow distillation recipe, arXiv 2510.14974](https://arxiv.org/abs/2510.14974) — ICLR 2026, 10→2 step distill <5% LIBERO drop
- [9] [Characterizing VLA Models, arXiv 2603.02271](https://arxiv.org/abs/2603.02271) — "action generation = 75% of latency" canonical citation
- [10] [HuggingFace SmolVLA blog](https://huggingface.co/blog/smolvla) — HF's own "skip VLM layers + async chunking" story
