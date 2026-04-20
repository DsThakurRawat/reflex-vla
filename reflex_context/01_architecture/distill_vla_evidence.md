# VLA Distillation — Empirical Evidence Review

*Research memo: Does distillation actually work on real VLAs (SmolVLA, pi0, pi0.5, GR00T), or only on toy diffusion policies?*

*Date: 2026-04-19. Scope: published papers, GitHub repos, industry blog posts. Excludes methodology (sibling agent).*

---

## TL;DR

Exactly **one** public paper (SnapFlow, 2604.05656, Apr 2026) actually distills a real flow-matching VLA (pi0.5 + SmolVLA) to 1-step and reports LIBERO numbers — with a ~1-point improvement (97.75% → 98.75%). Everything else is either (a) toy CNN/DiT policies distilled to 1-step (OneDP, Consistency Policy, FlowPolicy — parity to +2pp, not on VLAs), (b) "from-scratch 1-step" methods sidestepping distillation (MeanFlow: MP1, DM1, OFP, DBP, DMPO, OMP, HybridFlow), or (c) orthogonal distillation (VITA-VLA, FD-VLA — capability/force transfer, not step reduction).

Industry signal is near-silent. Physical Intelligence's pi0-FAST (autoregressive tokenizer, 4–5× *slower* than flow) moves opposite to 1-step. NVIDIA GR00T ships ONNX/TensorRT, not distillation. OpenPI has zero issues/PRs mentioning distillation.

**Bottom line:** VLA distillation is unsolved-but-tractable, with one promising 2026 data point. Reflex should budget research-grade risk, not commodity-engineering risk.

---

## Section A — Published VLA distillation track record

Rows are ordered from "most directly applicable to Reflex" down to adjacent work. "VLA" = the *actual* target is pi0/pi0.5/SmolVLA/GR00T/OpenVLA. "Toy" = a small CNN/ResNet18 diffusion policy, ~100M params, no VLM prefix.

| Paper | arxiv | Model distilled | Method | Teacher → student task success | Training cost | Reproducible? |
|---|---|---|---|---|---|---|
| **SnapFlow** (Luan et al., Apr 2026) | [2604.05656](https://arxiv.org/abs/2604.05656) | **pi0.5 (3B) + SmolVLA (500M)** — the real thing | Progressive self-distillation, 2-step Euler shortcut targets, zero-init time embed | pi0.5 LIBERO: 97.75% (10-step) → **98.75% (1-step)**; per-suite Spatial 97→99, Object 100→100, Goal 96→99, Long-10 89→91. SmolVLA PushT MSE −8.3%. | ~12 h on 1 GPU | **No code released yet (as of Apr 2026)**. Single-paper, no independent reproduction. |
| **OFP / One-Step Flow Policy** (Li et al., Mar 2026) | [2603.12480](https://arxiv.org/abs/2603.12480) | Toy (56 sim tasks) + **pi0.5 on RoboTwin 2.0** | From-scratch self-distillation with self-consistency loss, not true teacher→student | Toy: 1-step OFP matches or beats 100-step DP/Flow. pi0.5 on RoboTwin: **"1-step OFP surpasses original 10-step"** (94.7% avg across 4 tasks, no per-task baseline numbers given). | Not disclosed | No GitHub link in paper. |
| **OneDP** (Wang et al., NVIDIA, Oct 2024, ICML 2025) | [2410.21257](https://arxiv.org/abs/2410.21257) | Toy (ResNet18 + 1D-CNN U-Net DP) — **not a VLA** | KL-divergence distillation along diffusion chain | Avg: DP 82.9% → OneDP 84.3% (6 sim tasks). Real-world avg: DP 83% → OneDP 98%. Max single-task drop: PushT −4.7pp. | "2–10% additional pretraining cost"; 20 epochs vs 1000 for teacher | [Project page exists](https://research.nvidia.com/labs/dir/onedp/); GitHub status: "code coming soon" — **not yet released as of Apr 2026**. |
| **Consistency Policy** (Prasad et al., Stanford+CMU, May 2024) | [2405.07503](https://arxiv.org/abs/2405.07503) | Toy DDPM diffusion policy — **not a VLA** | Consistency distillation enforcing self-consistency along teacher trajectories | Robomimic: Lift 1.00→1.00, Can 0.97→0.98, Square 0.93→0.92, **Tool Hang 0.79→0.70 (−9pp)**, PushT 0.87→0.82 (−5pp), Franka Kitchen 0.98→0.93. Real-world 3 tasks: neutral to +10pp. | Authors: "needs more training time to reach the same perf"; no GPU-hours disclosed | Code stated available; reproduction not documented. |
| **FlowPolicy** (Zhang et al., AAAI 2025) | [2412.04987](https://arxiv.org/abs/2412.04987) | Toy 3D point-cloud policy — **not a VLA**. Not real distillation (trains from scratch). | Consistency flow matching from scratch | Adroit: Hammer 100%, Door 58%, Pen 53%; Metaworld avg 52.9%. Claims "+1.3% over DP3" despite 1 step. | Not disclosed | [zql-kk/FlowPolicy](https://github.com/zql-kk/FlowPolicy), 153★. Single-author fork network. No VLA ports visible. |
| **ManiFlow** (Yan et al., Sep 2025, UW+NVIDIA+Allen AI) | [2509.01819](https://arxiv.org/abs/2509.01819) | DiT-X transformer policy — **not a VLA** (no VLM prefix). Single-stage, no teacher. | Consistency flow training from scratch (not distillation) | RoboTwin 5 bimanual: 1-step 63.7% vs 3D-DP 10-step 42.7% and Flow 10-step 48.1%. Fails on tactile-required contact-rich tasks. | Not disclosed | No GitHub URL in paper. |
| **MP1 / DM1 / OMP** MeanFlow family (Sheng/Zou/Fang et al., 2025) | [2507.10543](https://arxiv.org/abs/2507.10543), [2510.07865](https://arxiv.org/abs/2510.07865), [2512.19347](https://arxiv.org/abs/2512.19347) | Toy 3D policies — **not a VLA**. No teacher. | MeanFlow identity, from scratch | MP1: +10pp over DP3; DM1: Lift 85→99% +10–20pp Robomimic. OMP: Adroit/MetaWorld SOTA at 1-step. | Not disclosed | [MP1 code](https://github.com/LogSSim/MP1). |
| **DBP / DBPO** (Gao et al., Apr 2026) | [2604.03540](https://arxiv.org/abs/2604.03540) | Toy. Drift-based 1-step policy, no teacher. | Fixed-point drift, internalizes refinement into weights | Matches or beats multi-step DP; 100× faster; deployed on dual-arm at 105.2 Hz | Not disclosed | None cited. |
| **DMPO** Dispersive MeanFlow (Zou et al., Jan 2026) | [2601.20701](https://arxiv.org/abs/2601.20701) | Toy. Lightweight, RoboMimic+Gym. | MeanFlow + dispersive reg + RL, no distillation | 5–20× speedup, >120 Hz Franka | Not disclosed | None cited. |
| **HybridFlow** (Dong et al., Feb 2026) | [2602.13718](https://arxiv.org/abs/2602.13718) | Toy. 2-NFE hybrid. | MeanFlow + ReNoise + ReFlow refine | Beats 16-step DP by **15–25pp** (unusual claim); 8× faster to 52 Hz | Not disclosed | None cited. |
| **VITA-VLA** (Dong et al., Oct 2025) | [2510.09607](https://arxiv.org/abs/2510.09607) | VLA — but this is *capability* distillation (small action model → VLM), not **step distillation** for speed. | Two-stage alignment + SFT | LIBERO 97.3% avg, real-world 82% (+17pp over teacher). Not a relevant comparison for Reflex. | N/A | Orthogonal purpose. |
| **FD-VLA** (Zhao et al., Feb 2026) | [2602.02142](https://arxiv.org/abs/2602.02142) | VLA — but this is *force-sensing* distillation, not speed. | Learnable query token → force token | Outperforms direct F/T sensor baselines. | N/A | Orthogonal purpose. |
| **OneDP adjacent — NVIDIA Multi-Student Diffusion Distillation** (2025) | ([NV research](https://research.nvidia.com/publication/2025-03_multi-student-diffusion-distillation-better-one-step-generators)) | Image-gen, not robot policies. | Multi-student distillation | Image FID improvements. | N/A | Not VLA. |

**Summary of Section A:** Excluding from-scratch 1-step methods and orthogonal-purpose distillation, **only three papers** address "distill a 10-step flow-matching policy to a 1-step student": OneDP (toy), Consistency Policy (toy), SnapFlow (real VLAs). Only SnapFlow targets real VLAs. That's a massive finding in itself.

---

## Section B — Industry leaks

**Physical Intelligence:** Public engineering has focused on *not needing* distillation. pi0-FAST ([2501.09747](https://arxiv.org/abs/2501.09747)) is a tokenizer-based autoregressive VLA that makes training 5× faster but inference **4–5× slower** than pi0's flow matching — opposite direction from 1-step. Their [Knowledge Insulation post](https://www.pi.website/research/knowledge_insulation) emphasizes "fast inference speed" for pi0.5+KI via flow matching, no Hz number, never mentions distillation. [OpenPI](https://github.com/Physical-Intelligence/openpi) has **zero open issues/PRs mentioning "distill" or "consistency"** as of Apr 2026 — strong negative signal.

**NVIDIA GR00T (N1–N1.7):** NVIDIA solves real-time control via **ONNX/TensorRT export + Jetson Orin/Thor**, not distillation. [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and [GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/) document full pipeline export but no distilled variants. GR00T still runs a 4-step DDPM loop. OneDP authors are NVIDIA research — they know the technique but haven't applied it to GR00T publicly.

**Hugging Face / lerobot / SmolVLA:** Efficiency strategy is **architectural** (layer skipping, half-depth VLM, token pruning, async inference), not distillation. SmolVLA paper/blog claims "30% faster, 2× throughput" via async inference. LeRobot issue search for "distill" returns only orthogonal hits. No SmolVLA-distill on HF Hub as of Apr 2026.

**Skild AI, 1X, Figure, Gemini Robotics:** No public signals on distillation.

**NVIDIA Nemotron:** NVIDIA ships LLM compression+distillation (Nemotron-Nano-9B-v2, 2508.14444) — proving they have the pipeline. **Absence of equivalent for GR00T is conspicuous.**

**Interpretation:** The parsimonious read is that big labs have not solved VLA distillation to their own standards. Physical Intelligence moving *away* from fast-inference flow (toward slower autoregressive) is telling — if they could have distilled pi0 to 1-step losslessly, they presumably would have. Alternative: distilled checkpoints exist as competitive moat, kept closed.

---

## Section C — Failures and cautionary tales

### C.1 Documented in-paper failures

**SnapFlow's own honesty** ([2604.05656](https://arxiv.org/abs/2604.05656)):
- **"Naively reducing the step count is unreliable, degrading success on most tasks due to the velocity field being uncalibrated for single-step jumps."** This is the naive baseline they ablate against — pi0.5 at 1-step without distillation hits 96.75% vs 97.75% at 10 steps, but with **high per-task variance** (e.g., libero_spatial Task 6 dropped 10pp; libero_10 Task 8 ranged 50–100% across runs).
- **End-to-end bottleneck shifts, not disappears:** "With denoising compressed to one step, the VLM prefix (60 ms) becomes the new bottleneck (72% of E2E)." Distillation of the action head alone caps out below the full-pipeline speedup one might hope for.
- Only evaluated on LIBERO simulation — **no real-robot validation**. Authors explicitly flag this.

**Consistency Policy** ([2405.07503](https://arxiv.org/abs/2405.07503)):
- **Tool Hang 0.79 → 0.70 (−9pp) in single-step**, recovered only partially at 3-step chained inference (0.77). Authors note: "harder tasks show larger single-step degradation since subsequent chaining steps may not have much room to improve outputs." So for precise/tight-tolerance tasks, expect degradation.
- PushT −5pp at 1-step.
- Authors admit "needs more training time to reach the same performance" — training is non-trivial.

**ManiFlow** ([2509.01819](https://arxiv.org/abs/2509.01819)):
- Explicit failure category: **"fails in tasks that require detailed contact information and precise force feedback, such as delicate assembly operations or compliant insertion tasks"** — independent of step count. For contact-rich tasks with no tactile input, 1-step flow policies underperform regardless of distillation quality.

**HybridFlow** ([2602.13718](https://arxiv.org/abs/2602.13718)):
- The paper's finding that "direct application of MeanFlow to robotics is impeded by critical theoretical pathologies, specifically spectral bias and gradient starvation in low-velocity regimes" is a cautionary tale against assuming image-gen 1-step techniques transfer cleanly. They need a 2-NFE hybrid to rescue precision.

### C.2 GitHub / community negative signals

- **No one has publicly distilled SmolVLA and reported failure.** No blog/issue/paper exists. Most likely read: nobody has tried publicly. SmolVLA is ~1 year old (Jun 2025), small community, everyone's still fine-tuning baseline. Empty space = "not yet attempted" more than "tried and failed silently." LeRobot community files noisy issues for everything, so silent failures are unlikely.
- **OpenPI issues #724, #559** are only inference-related active issues, both deployment bugs (transformers version, notebook errors), not distillation.
- **No SmolVLA/pi0 fork of Consistency Policy or OneDP in public code search.** OneDP code not released yet.

---

## Section D — GR00T DDPM-specific evidence

GR00T (N1 through N1.7) uses a **diffusion transformer** with flow-matching-style velocity prediction — see our local note `gr00t_ddpm_dit_vs_flow_matching.md`. It runs 4 denoising steps today. The question is whether the distillation literature for flow-matching transfers.

**Direct evidence:** None. No paper in the search distills GR00T specifically. NVIDIA's own OneDP ([2410.21257](https://arxiv.org/abs/2410.21257)) is from a sibling research group (DIR lab) but has not been applied to GR00T in any public artifact.

**Transferability evidence (mixed):**
- Progressive distillation works on DDPM combinatorial optimization with 16× speedup and 0.019% degradation ([2308.06644](https://arxiv.org/abs/2308.06644)) — but that's TSP, not embodied control.
- One-Step DMD ([2311.18828](https://arxiv.org/abs/2311.18828)) and EM Distillation ([2405.16852](https://arxiv.org/abs/2405.16852)) are solved for image-gen DDPM. Gap between "ImageNet FID still looks fine" and "robot task success doesn't drop 15pp" is enormous and not bridged by these papers.
- SnapFlow's framing ([2604.05656](https://arxiv.org/abs/2604.05656)) is explicitly for **flow-matching** VLAs. GR00T is flow-matching-adjacent (velocity prediction with Gaussian prior) but not identical — SnapFlow's "2-step Euler shortcut velocity" trick *should* generalize, but has not been publicly tested on GR00T's 4-step DDPM formulation.
- Consistency Models work on both DDPM and flow matching in principle, but Consistency Policy ([2405.07503](https://arxiv.org/abs/2405.07503)) only tests DDPM (Diffusion Policy teacher). The 5–9pp drops observed on hard tasks (Tool Hang, PushT) are a lower bound on what could happen with a GR00T student.

**Most likely GR00T outcome given literature:** Applying SnapFlow-style self-distillation to GR00T's 4-step DDPM *should* work, with risk of 2–5pp task-success drop for a 4× inference speedup. Progressive distillation 4→2→1 is the safer path than direct 4→1, consistent with the HybridFlow finding that 2-NFE is often the sweet spot. But this is extrapolation from adjacent literature — no one has published it.

---

## Section E — Realistic expectations for a distilled VLA

Synthesizing the three data points most relevant to Reflex (SnapFlow, OneDP, Consistency Policy):

**Task success floor vs teacher, 1-step:**
- **Best case (SnapFlow on pi0.5 / OneDP real-world):** +1 to +15pp gain. Likely inflated by evaluation noise (10 episodes/task) and on tasks close to saturation.
- **Median case:** Parity to −2pp average, with per-task variance ±5pp. This is what the mean of Consistency Policy, OneDP, SnapFlow ablations shows on standard suites.
- **Worst case on harder tasks:** −5 to −10pp, specifically on (a) tight-tolerance manipulation (Tool Hang-style), (b) long-horizon with precise trajectories (PushT-style), (c) contact-rich without tactile sensing. If reflex's teacher is at 40%, budget student floor at **30–35%**, not 38%.

**Latency speedup (end-to-end, not just denoise step):**
- **Denoise-alone:** 9–20× (SnapFlow: 9.6×; Consistency Policy: ~10×; OneDP: ~40× from 1.5 Hz → 62 Hz).
- **End-to-end including VLM prefix:** 3.3–3.6× on SmolVLA (SnapFlow: 178 ms → 50 ms, 3.56×). **For pi0.5: SnapFlow reports 274 ms → 83 ms ≈ 3.3×.** The VLM prefix becomes the new bottleneck (60 ms ≈ 72% of post-distill E2E per SnapFlow).
- **Reaching >100 Hz real-time** will require *both* distillation *and* VLM acceleration (layer pruning, token drop, quantization). SnapFlow explicitly flags this: "orthogonal to layer-distillation and token-pruning approaches, enabling compositional speedups."

**Training time per-VLA:**
- **SnapFlow:** ~12 h on 1 GPU. Best case.
- **OneDP:** 20 epochs vs 1000 teacher = ~2% of teacher training (~hours).
- **Consistency Policy:** "Needs more training time" — likely days.
- **Realistic Reflex budget:** 1–3 days per VLA on 1–2 A100s for a first attempt; multi-week if method needs iteration. Not weeks per deployment, but weeks per method refinement.

**Honest pitch for Reflex distill:**
> "For flow-matching VLAs (pi0, pi0.5, SmolVLA), expect end-to-end inference speedup of 3–4× (to ~50–80 ms on A100, ~12–25 Hz control) and task-success delta in the range [−5pp, +1pp] vs the 10-step teacher on in-distribution tasks. Expect 5–10pp drops on precise/contact-rich tasks. Training is ~1 day on a single A100 per model. Real-robot validation of the distilled policy remains unpublished as of Apr 2026; Reflex would be among the first to report it."

Do **not** pitch "no loss" or "100 Hz guaranteed." Those are not supported by the literature.

---

## Section F — Honest assessment

**Verdict: Unsolved-but-tractable research-grade problem, leaning toward tractable for pi0.5/SmolVLA, genuine research for GR00T.**

Defense:

1. **Not solved.** Zero code-released, independently-reproduced, real-robot-validated examples of a distilled VLA exist as of Apr 2026. The one paper that targets real VLAs (SnapFlow) is a single Chinese-academic preprint from 2 weeks ago with no code and no real-robot numbers. A solved problem would have an HF Hub model, a reproduction thread, and industry buy-in. None of those exist.

2. **Not a research gamble either.** The adjacent literature (OneDP, Consistency Policy, ManiFlow, MeanFlow family) establishes that 1-step flow/diffusion policies are *feasible* on toy tasks with parity-to-small-loss. SnapFlow gives one existence proof that the technique scales to 3B-parameter real VLAs with apparent success. This is materially stronger evidence than "someone speculated it should work."

3. **The risk profile is specific, not general:**
   - **Low risk:** Match 10-step pi0.5 / SmolVLA teacher within ±2pp on benchmark tasks (LIBERO-Spatial, Object, Goal).
   - **Medium risk:** Match teacher on LIBERO-Long-10 (long-horizon) and on RoboTwin (bimanual). SnapFlow hints +2pp on Long-10 but small sample.
   - **High risk:** Real-robot zero-shot deployment without task-success regression. **Nobody has reported this.** SnapFlow authors explicitly flag it as future work.
   - **Very high risk:** Distilling GR00T specifically (DDPM vs flow matching, 4-step vs 10-step, different VLM backbone). Transferability is plausible but unverified.

4. **Why this is the right problem for Reflex:** The gap between "arxiv claims 1-step VLA works" and "user can `reflex distill pi0.5` and get a production artifact with benchmarked success" is exactly a deployment-toolchain opportunity. Reflex is not research *into* distillation — it's making distillation a reproducible one-command product once the method stabilizes. SnapFlow's 12-hour single-GPU footprint is deployable-scale, not research-scale.

5. **Caveats the pitch must include:**
   - First-party task-success benchmarking is mandatory. Do not ship a distilled VLA without running it against the user's actual tasks (not LIBERO). The gap between sim and real is unbounded in this literature.
   - Expose teacher/student/NFE as user-visible parameters. Let the user trade off.
   - Budget for 5–10pp degradation on precise/contact-rich tasks as the documented failure mode. Warn the user.
   - Do not promise GR00T distillation Day 1. Support SmolVLA and pi0.5 first where SnapFlow gives direct evidence.

**Empty-space finding worth flagging:** The absence of distilled-SmolVLA / distilled-pi0 on HF Hub or GitHub is not ambiguous. Given that (a) SmolVLA is specifically marketed as consumer-deployable, (b) lerobot is maximally open, (c) the technique has been in the open-source literature for 2+ years (Consistency Policy, 2024), the empty space means the community has not yet tried, not that it failed. **Reflex can be first.** But first-mover means taking real-robot validation on the chin — the literature does not yet tell us whether a distilled pi0.5 can fold laundry in the kitchen as well as a 10-step pi0.5.

---

## Citation appendix

Primary (directly relevant to VLA distillation-for-speed):
- SnapFlow: [arxiv 2604.05656](https://arxiv.org/abs/2604.05656) — Luan et al., Apr 2026
- OFP: [arxiv 2603.12480](https://arxiv.org/abs/2603.12480) — Li et al., Mar 2026
- OneDP: [arxiv 2410.21257](https://arxiv.org/abs/2410.21257) — Wang et al., NVIDIA, ICML 2025
- Consistency Policy: [arxiv 2405.07503](https://arxiv.org/abs/2405.07503) — Prasad et al., May 2024

Secondary (from-scratch 1-step, not distillation but relevant):
- ManiFlow: [2509.01819](https://arxiv.org/abs/2509.01819)
- MP1: [2507.10543](https://arxiv.org/abs/2507.10543)
- DM1: [2510.07865](https://arxiv.org/abs/2510.07865)
- OMP: [2512.19347](https://arxiv.org/abs/2512.19347)
- DBP: [2604.03540](https://arxiv.org/abs/2604.03540)
- DMPO: [2601.20701](https://arxiv.org/abs/2601.20701)
- HybridFlow: [2602.13718](https://arxiv.org/abs/2602.13718)
- FlowPolicy: [2412.04987](https://arxiv.org/abs/2412.04987)
- Streaming Flow: [2505.21851](https://arxiv.org/abs/2505.21851)

Orthogonal distillation (for completeness):
- VITA-VLA (capability distill, not speed): [2510.09607](https://arxiv.org/abs/2510.09607)
- FD-VLA (force distill, not speed): [2602.02142](https://arxiv.org/abs/2602.02142)
- PRISM (LLM distill for planning): [2506.17486](https://arxiv.org/abs/2506.17486)

Industry anchors:
- pi0: [2410.24164](https://arxiv.org/abs/2410.24164)
- pi0-FAST: [2501.09747](https://arxiv.org/abs/2501.09747)
- SmolVLA: [2506.01844](https://arxiv.org/abs/2506.01844)
- GR00T N1: [2503.14734](https://arxiv.org/abs/2503.14734)
- OpenVLA-OFT: [2502.19645](https://arxiv.org/abs/2502.19645)
- OpenPI repo: [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- Isaac-GR00T repo: [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- lerobot repo: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

Diffusion-distillation foundations (ImageNet-era, for GR00T DDPM transferability):
- DMD: [2311.18828](https://arxiv.org/abs/2311.18828)
- EM Distillation: [2405.16852](https://arxiv.org/abs/2405.16852)
- VarDiU: [2508.20646](https://arxiv.org/abs/2508.20646)
- Progressive Distillation for combinatorial / DDPM: [2308.06644](https://arxiv.org/abs/2308.06644)
