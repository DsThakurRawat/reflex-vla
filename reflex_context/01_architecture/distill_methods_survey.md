# Distillation Methods Survey — for `reflex distill`

**Scope.** Opinionated survey of techniques that compress multi-step generative robot policies (flow-matching, DDPM) into 1-step or few-step generators. Feeds the scope and architecture decision for the new `reflex distill` module. Target: >100 Hz reactive control (<10 ms per action) on Jetson-class edge hardware, down from the current 10-50 ms multi-step regime.

**Paradigms in scope.**
- Flow-matching VLAs: SmolVLA (arXiv 2506.01844), pi0, pi0.5 — 10 Euler steps of a learned velocity field.
- DDPM DiT VLAs: GR00T N1.6 (arXiv 2503.14734) — 4 DDIM steps over a 32-layer DiT with AdaLN.

Paradigm note: the actual computational object being integrated differs. In flow matching, the network predicts an instantaneous velocity `v(x_t, t)` and `x_{t+dt} = x_t + dt · v`. In DDPM, the network predicts noise or x0 and the reverse step uses a variance-preserving SDE discretization. Distillation methods that operate on "the ODE trajectory" apply to both; methods that assume score-based parameterization require more care on flow-matching models.

---

## Section A — Method matrix

| Method | Paradigm | 1-step quality | Training stability | Needs teacher | GitHub ref |
|---|---|---|---|---|---|
| Progressive Distillation (Salimans & Ho, 2202.00512) | DDPM-native, works on flow via v-parameterization | 4-step matches teacher; 1-step degrades | Stable; each stage is an MSE regression | Yes (halved each round) | `google-research/google-research/tree/master/diffusion_distillation` |
| Consistency Models (Song et al., 2303.01469) | DDPM-native; consistency-training variant also works on flow | FID 3.55 CIFAR-10 / 6.20 ImageNet-64 at 1 NFE | Training-from-scratch unstable, EMA-sensitive; distillation variant stable | Distillation: yes. CT: no | `openai/consistency_models` |
| Consistency Policy (Prasad et al., 2405.07503) | DDPM (Diffusion Policy); flow variant exists | Within 2-5% success of teacher at 1-NFE on 6 sim tasks | Stable; robust to teacher quality | Yes | `Aaditya-Prasad/Consistency-Policy` |
| Rectified Flow / InstaFlow (Liu et al., 2209.03003; 2309.06380) | Flow-matching-native; k-Reflow iterates | 1-Reflow: FID 23 SD-1.5 MS-COCO at 1 NFE | Stable for k=1; reflow round 2+ diverges for action chunks | Yes (for reflow) | `gnobitab/InstaFlow`, `gnobitab/RectifiedFlow` |
| Shortcut Models (Frans et al., 2410.12557) | Flow-matching-native | Matches 1-NFE baselines; 128-step quality at 4 NFE | Stable; one model, one training phase | No (self-distillation) | `kvfrans/shortcut-models` |
| SnapFlow (Luan et al., 2604.05656) | Flow-matching-native, VLA-specific | 98.75% LIBERO on pi0.5 at 1 NFE, matches 10-step teacher | Stable; self-distillation, no teacher copy | No | Paper-only as of 2026-04-07; code not yet public |
| pi-Flow (arXiv 2510.14974) | Flow-matching-native, image-gen-native, applied to VLA | 1-NFE FID 2.85 ImageNet-256; <5% LIBERO drop at 2-NFE on pi0.5 | Stable; imitation-distillation of velocity along policy trajectory | Yes (frozen 10-step) | Scaffolded in `reflex`: `src/reflex/distill/pi_flow.py` |
| OneDP (Wang et al., 2410.21257) | DDPM (Diffusion Policy) | 1.5 Hz → 62 Hz; SOTA success on Robomimic | KL-divergence loss along diffusion chain; 2-10% additional pretrain | Yes | `research.nvidia.com/labs/dir/onedp` |
| DMD (Yin et al., 2311.18828) | DDPM-native; flow adaptation works | FID 2.62 ImageNet-64; 11.49 zero-shot COCO | Unstable (GAN-like); two auxiliary score models required | Yes (score) | `tianweiy/DMD` |
| DPPO (Ren et al., 2409.00588) | DDPM-native; RL fine-tune, not distillation | N/A (not a distillation method) | Stable for diffusion parameterization | No (uses RL reward) | `allenzren/DPPO` |
| Score Distillation Sampling (Poole et al., 2209.14988) | Image-diffusion asset optimization | Limited — targets 3D NeRFs, not 1-step generator | Mode-seeking; low diversity | Yes | `ashawkey/stable-dreamfusion` |
| MeanFlow (Geng et al., 2505.13447) | Flow-matching, one-step from scratch | FID 3.43 ImageNet-256 at 1 NFE | Stable; trained from scratch, no distillation, no curriculum | No | `Gsunshine/meanflow` (unofficial PyTorch) |
| MVP (Mean Velocity Policy, Zhan et al., 2602.13810) | Flow, RL policy | SOTA Robomimic / OGBench at 1 NFE | Stable; IVC regularization prevents degeneracy | No | Paper-only, pre-print 2026-02 |
| DMPO (Zou et al., 2601.20701) | Flow/MeanFlow-native, for policies | >120 Hz control; competitive on RoboMimic | Stable (MeanFlow + RL); no distillation stage | No | Scaffolded in `reflex`: `src/reflex/distill/dmpo.py` |

Row "Paradigm" distinguishes native fit from adapted fit. "Needs teacher" is the operationally meaningful question for our training-compute budget.

---

## Section B — Top 3 candidates

Selection criteria: (a) proven on VLA-scale transformer architectures, not just UNets/DiTs at image-gen scale; (b) stable training without adversarial dynamics; (c) small-data friendly — LeRobot/LIBERO/DROID offer <100k episodes per setup, not ImageNet-21k.

### 1. SnapFlow (arXiv 2604.05656, 2026-04-07) — PRIMARY CANDIDATE

**Why it fits flow-matching VLAs.** Designed for exactly our setting. Published 12 days ago. Validated on pi0.5 (3B) and SmolVLA (500M) — the exact two architectures Reflex ships export/serve/validate for. Headline result on pi0.5 LIBERO: **98.75% average success at 1 NFE vs 97.75% at 10 NFE teacher**, with end-to-end latency 274 ms → 83 ms (3.3x) and denoise speedup 9.6x. On SmolVLA: 8.3% MSE reduction, 3.56x end-to-end speedup.

**Formulation.** Mixes standard flow-matching samples with "consistency samples" whose targets are two-step Euler shortcut velocities computed from the model's own marginal velocity predictions. Zero-initialized target-time embedding lets one network switch between local velocity estimation and global one-step generation.

**Does it cover DDPM?** Not directly. SnapFlow's core insight — computing two-step Euler shortcut velocities from the model's marginal velocity field — assumes flow-matching parameterization. For GR00T N1.6's DDPM DiT we would need to port the idea (the consistency-loss shape translates, but noise-prediction vs velocity-prediction changes the math) or pair it with Consistency Policy below.

**Reference implementation.** Code not yet released as of 2026-04-19. Paper describes the training loop in enough detail to reimplement in ~300 lines of PyTorch. Loss has three components: (i) standard flow-matching MSE on velocity; (ii) consistency loss on two-step Euler shortcut target; (iii) zero-init embedding.

**Hard requirements.**
- Teacher is the frozen base VLA at 10 NFE (same weights used for `flow-matching` base inference).
- No classifier-free guidance needed — VLA conditioning is image+state+language, not CFG scale.
- Same backbone architecture as teacher (no replacement).
- Requires real observation data `(image, state, language) → (action_chunk_noise, target_velocity)`. Noise-only training does NOT work because action head is conditioned on VLM features — the observation distribution matters. This is the same correction the `2026-04-14-ship-distill-first` pre-mortem applied to pi-Flow.

**Expected sample-quality floor.** On pi0.5 LIBERO: paper-claimed 1% gain at 1 NFE. On SmolVLA LIBERO: paper-claimed MSE reduction, task-success not published. Practical floor assumption for Reflex: 2-5% task-success drop vs teacher on LIBERO-10, consistent with pi-Flow's earlier <5% claim.

**GPU-hours.** Paper: ~12 h on a single GPU for pi0.5 (3B) scale. Adjusted for our Modal A10G rate (~$0.50/hour) and expected data-loading overhead: ~200-400 A10G-hours, $100-200 Modal per distilled model.

### 2. pi-Flow (arXiv 2510.14974) — SECONDARY CANDIDATE

**Why it fits.** Currently the scaffolded recipe in `src/reflex/distill/pi_flow.py`. Velocity-field matching loss — student predicts the teacher's velocity along the teacher's own trajectory, not just the endpoint. For VLAs, this preserves the stochastic expressivity that matters for multimodal action distributions (key for manipulation tasks where multiple grasps are valid).

**Formulation.** At each training step, sample `t ∈ [0, 1]`, integrate teacher to `x_t`, compute teacher velocity `v_T(x_t, t)`, compute student velocity `v_S(x_t, t)`, minimize `‖v_S - v_T‖²`. At inference, student runs 2-4 Euler steps instead of 10.

**Does it cover DDPM?** Somewhat. For GR00T's DDPM DiT, the noise-prediction network can be reinterpreted as a velocity field via `v = -σ_t · ε` (standard DDPM→flow equivalence). Training code would need a paradigm switch, so it's NOT the same training path. Expected ~20% extra engineering to cover both.

**Reference implementation.** Scaffolded already (`ed8157c`). Full implementation referenced in `2026-04-14-ship-distill-first` ADR at ~200 lines of PyTorch plus a Modal script.

**Hard requirements.**
- Teacher deterministic? Yes — teacher runs in eval mode with fixed noise seed for trajectory caching to work efficiently.
- No CFG.
- Real observation data, same as SnapFlow.

**Expected sample-quality floor.** Paper claim: <5% LIBERO drop at 2 NFE (this is what the `reflex distill --recipe pi_flow` marketing claim is based on). Actual 1-NFE for pi-Flow is not claimed in the original paper; for 1-NFE you need something closer to SnapFlow or MeanFlow-style mean-velocity learning.

**GPU-hours.** ~200-500 A10G-hours per model; $200-500 Modal budget (as revised in the pre-mortem).

### 3. Consistency Policy (arXiv 2405.07503) — DDPM COVERAGE CANDIDATE

**Why it fits.** Explicit robotics distillation work, validated on 6 sim tasks + 3 real-world tasks on laptop GPU. Authors specifically designed it for "mobile manipulators or quadrotors that cannot carry high-end GPUs" — our deployment target exactly. Robust to pretrained teacher quality (no need for extensive teacher validation before distilling).

**Does it cover flow-matching?** Directly usable with a change of base model. Consistency Policy was trained on top of Diffusion Policy (Chi et al.) which uses DDPM. For flow-matching VLAs, Consistency Training (CT) instead of Consistency Distillation (CD) is theoretically possible but less validated in robotics. SnapFlow is the cleaner flow-matching answer; Consistency Policy is the cleaner DDPM answer.

**Formulation.** Self-consistency enforcement along Heun-style ODE integration. `f_θ(x_t, t) = f_θ(x_{t-Δt}, t-Δt)` at distillation, constrained along the teacher's trajectory.

**Hard requirements.**
- Teacher is a pretrained Diffusion Policy (for CD variant).
- Needs teacher ODE integrator — Heun/DPM-Solver. GR00T already uses DDIM which is compatible.
- Small initial sample variance is a key design decision (paper §5).

**Expected sample-quality floor.** Paper: 8-15% success drop at 1-NFE on some tasks; order-of-magnitude inference speedup. Better alternative for DDPM than forcing SnapFlow onto it.

**Reference implementation.** `github.com/Aaditya-Prasad/Consistency-Policy` — canonical PyTorch reference.

**GPU-hours.** Paper: 1-2 hours on A6000 per task. Scaled to our setting: ~50-150 A10G-hours per distilled model.

---

## Section C — Contrarian case

**The real argument: distillation is sunk cost if flow-matching VLAs are a 2-year bridge.**

OpenVLA-OFT (arXiv 2502.19645) changed the terms of the conversation. Kim, Finn, and Liang showed that **L1 regression beats diffusion for VLA action heads**. Their OFT recipe — parallel decoding + action chunking + continuous action representation + **L1 regression loss** — raised OpenVLA's LIBERO average from 76.5% to 97.1% and increased action-generation throughput **26×** vs the diffusion baseline. The L1 head runs in a single forward pass, with no denoising loop at all. No distillation needed. Authors explicitly compared against pi0 and RDT-1B using their default recipes and beat them.

VLM2VLA (arXiv 2509.22195) pushes further: represent actions as language tokens and fine-tune with LoRA. No action head whatsoever. The VLM autoregressively generates action tokens. This approach avoids the entire "how do we accelerate a generative model" question by not having one. Generalization to novel tasks with open-world semantic reasoning — the thing diffusion-based VLAs struggle with when fine-tuned — is preserved.

SmolVLA's own async-inference stack (2506.01844 §4) decouples perception from action prediction and reaches higher control rates through **chunking and async**, not single-step denoising. The Dexmal real-time VLA paper (2510.26742) reinforces this: the 3-5 FPS → 20-30 Hz gap closes primarily via **VLM prefix KV-cache** optimization (which `reflex export v2` handles), not by collapsing the 10-step denoising loop.

**What this means.** If a Reflex customer's real constraint is task-success-at-speed, the decision tree in 2026 may look less like "distill the diffusion" and more like "swap the diffusion head for an L1 regression head and retrain." That shift:
- Needs ~same amount of training compute (fine-tune with L1 instead of flow-matching loss).
- Gives you 1-NFE inherently — no loss of stochastic expressivity because the OFT paper shows multimodality is captured by action chunking + ensemble.
- Removes the entire class of "velocity field calibration" bugs our distillation output inherits.
- Doesn't generalize to models where the teacher's stochastic policy is actually needed (bimanual dexterous manipulation, contact-rich tasks — but OFT paper shows even these work with L1 on ALOHA).

**Where the contrarian case is weakest.** (a) Reflex's wedge is *deploying the VLAs that exist*, not retraining them. Customers who already have a pi0.5 or GR00T N1.6 checkpoint in production need inference-time acceleration, not a paradigm switch. (b) GR00T N1.6's DDPM DiT was shipped by NVIDIA in March 2025 and is not getting an L1-regression variant before 2027. (c) VLM2VLA has been validated in 800 real-robot experiments — but only on ~3 robot embodiments, and its generalization story rests on LoRA preserving VLM foundation knowledge; Reflex has no evidence this holds across the embodiments customers deploy on.

**Net recommendation.** Distillation is NOT misprioritized in 2026 *if* it's treated as a 12-18-month bridge capability for the installed base of flow-matching VLAs. It IS misprioritized if presented as the long-term strategic moat. The `reflex distill` wedge should be scoped to the installed base (SmolVLA, pi0, pi0.5, GR00T N1.6) and the roadmap should include an `--approach l1_head_replacement` recipe as a Phase 3 addition once OFT-style heads are more production-proven.

---

## Section D — Cross-paradigm coverage

For each top-3 method, whether one training loop covers both flow-matching AND DDPM, or whether we need two paths:

**SnapFlow (top candidate).** Flow-matching only in its published form. For DDPM coverage, the "two-step Euler shortcut velocity" computation has no direct analogue in noise-prediction DDPM; we'd need to reformulate using the DDPM → flow equivalence (`v = -σ_t · ε`). Engineering estimate: **separate training file for DDPM**, ~150 LOC delta from the flow-matching file. Loss function and data-loader can be shared; only the velocity-computation shim changes.

**pi-Flow.** Velocity-field matching is conceptually paradigm-neutral if we define "teacher velocity" as either (a) the direct flow-matching velocity output or (b) the velocity implied by DDPM's noise-prediction via the score-field equivalence. Engineering estimate: **one training file with a `parameterization` switch**, ~50 LOC delta.

**Consistency Policy.** Native DDPM (via Heun solver). For flow-matching, Consistency Training (CT) variants exist but are less validated for robotics specifically. Engineering estimate: **separate training file for flow-matching**, ~200 LOC delta.

**Scope recommendation.** Ship SnapFlow as `reflex distill --recipe snapflow --paradigm flow` for the flagship case (SmolVLA, pi0, pi0.5). Ship Consistency Policy as `reflex distill --recipe consistency --paradigm ddpm` for GR00T. This keeps each trainer clean and idiomatic to its paradigm; the tradeoff is two trainers to maintain. Do NOT attempt a unified trainer — the paradigm abstraction would leak and introduce the class of bugs we fought through in `pi0_monolithic_wrap_pattern.md`.

---

## Section E — Honest unknowns

1. **SnapFlow code is not yet public.** Paper posted 2026-04-07. We are 12 days in. Our reimplementation has to work from the paper alone; we do not have reference hyperparameters or any published LIBERO task-wise breakdown. Expect 2-3 weeks of "our loss curve doesn't match the paper" debugging before the first clean result.

2. **Small-data regime for robot observations.** Image-generation distillation methods are validated on ImageNet-21k / LAION scale. Our training set for a LIBERO-distilled student is at most 10k episodes ≈ 500k (obs, action-chunk) pairs. Whether SnapFlow's consistency-sample loss is well-conditioned at this scale is not tested in the paper.

3. **VLM backbone frozen vs fine-tuned during distillation.** The published pi-Flow and SnapFlow papers are unclear about whether the VLM backbone is frozen or gets the same gradient as the action expert. For robot policies conditioned on VLM features, backbone drift during distillation would corrupt the feature distribution the action head is conditioned on. Default: freeze. But we haven't seen this verified.

4. **Does distillation compose with TRT FP16 export?** Our serve path depends on TRT FP16 for sub-15ms inference. The distilled student's velocity field has sharper curvature than the teacher's (by construction — it takes larger effective steps). FP16 quantization error on steep gradients is an open question; we currently validate teacher parity at 1e-4 tolerance and have never run this check on a distilled student.

5. **Task-success delta vs MSE delta.** Paper-reported numbers in this survey are either MSE on validation actions or task-success on LIBERO. Our own `validate` pipeline measures Wasserstein on sampled action chunks (arXiv 2603.13966 wrap). No literature paper connects MSE-on-action to Wasserstein-on-chunks, so we will discover the relationship empirically during the first distilled-student run.

6. **Does 1-NFE break chunking semantics?** Real-Time Chunking (arXiv 2506.07339) assumes the server can infer the next chunk while the current is being consumed. If 1-NFE inference runs in 3 ms but the VLM prefix encoder takes 8 ms, the chunking threshold logic `chunk_size_threshold=0.7` may be wrong in the new regime. This is an interaction bug nobody has published on.

7. **Paradigm equivalence for GR00T's specific DDPM.** GR00T N1.6 runs at 4 DDIM steps. A 1-NFE distilled student of a 4-step DDPM policy is a different problem from a 1-NFE distilled student of a 10-step flow-matching policy — the compression ratio is 4× not 10×, so the sample-quality floor may be higher. Literature (Consistency Policy paper) only validates on longer-horizon diffusion (100 steps → 1 step), not this short-chain regime.

8. **RL-free vs RL-based distillation.** DMPO (arXiv 2601.20701) argues RL during distillation lets the student exceed teacher performance. SnapFlow and pi-Flow are supervised. Whether we want "match teacher cheaply" (supervised, 1 training stage) or "exceed teacher at cost of instability" (RL, 2-3 stages) is a scope call, not a technical unknown — but the literature leans supervised for reproducibility.

---

## Citations summary

Primary papers referenced:
- 2202.00512 — Salimans & Ho, Progressive Distillation
- 2209.03003 — Liu et al., Rectified Flow
- 2209.14988 — Poole et al., DreamFusion / SDS
- 2210.03142 — Meng et al., Distillation of Guided Diffusion Models
- 2303.01469 — Song et al., Consistency Models
- 2309.06380 — Liu et al., InstaFlow
- 2310.14189 — Song & Dhariwal, Improved Techniques for Consistency Models
- 2311.18828 — Yin et al., Distribution Matching Distillation (DMD)
- 2405.07503 — Prasad et al., Consistency Policy
- 2409.00588 — Ren et al., Diffusion Policy Policy Optimization (DPPO)
- 2410.12557 — Frans et al., One Step Diffusion via Shortcut Models
- 2410.21257 — Wang et al., One-Step Diffusion Policy (OneDP)
- 2502.19645 — Kim, Finn & Liang, OpenVLA-OFT
- 2503.14734 — NVIDIA, GR00T N1
- 2505.13447 — Geng et al., Mean Flows for One-step Generative Modeling
- 2506.01844 — Shukor et al., SmolVLA
- 2506.07339 — Real-Time Chunking (for chunking interaction)
- 2509.22195 — Hancock et al., VLM2VLA (Actions as Language)
- 2510.14974 — pi-Flow
- 2510.17858 — Cai et al., Shortcut distillation for pretrained flow models
- 2510.26742 — Dexmal Real-Time VLA
- 2602.13810 — Zhan et al., Mean Velocity Policy (MVP)
- 2601.20701 — Zou et al., Dispersive MeanFlow Policy Optimization (DMPO)
- 2604.05656 — Luan et al., SnapFlow

Reflex-internal references:
- `reflex_context/01_architecture/gr00t_ddpm_dit_vs_flow_matching.md` — paradigm distinction
- `reflex_context/03_research/papers_referenced.md` — running index
- `reflex_context/01_decisions/2026-04-14-ship-distill-first.md` — SUPERSEDED but relevant
- `reflex_context/01_decisions/2026-04-16-council-reprioritization.md` — current v0.2 priority
- `src/reflex/distill/pi_flow.py`, `src/reflex/distill/dmpo.py` — existing scaffolds
