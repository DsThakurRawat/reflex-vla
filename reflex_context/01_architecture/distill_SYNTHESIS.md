# `reflex distill` — Synthesis & Build Plan

**Date**: 2026-04-20
**Status**: Research complete. Scope committed. Architecture locked. Ready to build v0.3.
**Parent goal**: `distill-dmpo` in GOALS.yaml (weight 9) — will be renamed `distill-snapflow` on v0.3 cut.

This doc is the one-page decision; read it first. The five reports below provide the evidence trail if you want to pressure-test any claim.

---

## Research + design docs (read in order if going deep)

| Doc | What it answers |
|---|---|
| [distill_methods_survey.md](distill_methods_survey.md) | Technique SOTA (SnapFlow, Consistency Policy, DMPO, Rectified Flow, Score Distillation, adversarial). Top 3 candidates. Contrarian case. 3065 words. |
| [distill_vla_evidence.md](distill_vla_evidence.md) | Empirical evidence on VLAs specifically. Industry silence. Failure modes. 2939 words. |
| [distill_latency_profile.md](distill_latency_profile.md) | Per-stage latency breakdown by VLA × hardware. Customer pain frequency. 3050 words. |
| [distill_scope_decision.md](distill_scope_decision.md) | The commit — which VLAs, which method, which tier, kill-gates. 2100 words. |
| [distill_architecture.md](distill_architecture.md) | File-level implementation plan. 7 axioms, 6 kill-gates, module tree. 3091 words. |

---

## One-paragraph summary

`reflex distill` reimplements SnapFlow (arxiv 2604.05656, April 2026) to turn 10-step flow-matching VLAs into 1-step generators. Empirical pick is forced: SnapFlow is the only published method that successfully distilled real VLAs (pi0.5 + SmolVLA) with measured task success — 98.75% on LIBERO at 1-NFE vs 97.75% at the 10-step teacher. No code public yet; reflex becomes the first reproducible implementation (~300 LOC). v0.3 scope is **pi-family only** (pi0 + pi0.5) — GR00T's denoise is only 35% of its end-to-end latency, so DDPM distillation is deferred to v0.5+. Architecturally it plugs into the existing `reflex finetune` trainer registry as `phase=distill`, reusing ~80% of the fine-tune substrate (preflight, parity gate, export chain, VERIFICATION.md). **Pro-gated** trainer; free-tier consumers can deploy community-distilled checkpoints via `reflex export --from-distilled`. Realistic expectation: **3-4× E2E speedup** (not 10×), task-success delta **[−5pp, +1pp]** on benchmark tasks (contact-rich tasks at risk of −5 to −10pp). Positioned as a 12-18 month bridge, not a long-term moat.

---

## The decision tree

### What `reflex distill` v0.3 IS

- A SnapFlow-only trainer for flow-matching VLAs (pi0 + pi0.5 in v0.3)
- A `phase=distill` variant of `reflex finetune` — same CLI pattern, same preflight, same parity + calibration gates, same auto-export
- A Pro-gated trainer (hard paywall) with free-tier deployment of distilled checkpoints
- A kill-gate-enforced shipping bar (LIBERO task success ≥ teacher − 5pp, cos-parity ≥ +0.999)

### What `reflex distill` v0.3 is NOT

- Not a DDPM distiller (Consistency Policy for GR00T deferred to v0.5+)
- Not a SmolVLA distiller yet (v0.3.1 conditional on velocity-convergence validation)
- Not a universal multi-paradigm abstraction (explicit: one trainer per paradigm, no leaky glue)
- Not a replacement for the fine-tune pipeline — it IS the pipeline with a different phase

---

## Why each design call was made

### 1. SnapFlow (not pi-Flow, not DMPO, not Consistency Policy)

**Decision**: v0.3 ships exactly one distillation method: SnapFlow.
**Evidence**: methods survey Section B — SnapFlow is the only paper validated on real VLAs with reported LIBERO numbers (+1pp vs teacher at 1-NFE). pi-Flow is secondary (2-4 NFE, lower ceiling). DMPO turned out to be "Dispersive MeanFlow Policy Optimization" — RL + MeanFlow, not knowledge distillation. Consistency Policy is DDPM-native; our v0.3 customer is flow-matching only.
**Trade-off accepted**: we're deprecating the pre-existing `src/reflex/distill/{pi_flow,dmpo}.py` scaffolds. Moving to `archive/v0.2/`. Single-method focus lowers implementation risk at the cost of paradigm coverage.

### 2. pi-family v0.3, SmolVLA v0.3.1, GR00T v0.5+

**Decision**: ship pi0 + pi0.5 only in v0.3.
**Evidence**: latency profile Section B — denoise fraction (the upper bound on distill's wins) varies wildly:
- pi-family on cloud GPU: **78-82%** denoise → distill wins big
- SmolVLA: middle ground + velocity-convergence concerns flagged in our own adaptive-denoise data (0/25 triggers in LIBERO tests)
- GR00T on RTX 5090: denoise only **35%**, VLM is **58%** → distillation caps at 1.35× E2E speedup
**Trade-off accepted**: we compromise on the "all 4 VLAs covered" pitch. The scope-discipline is defensible because GR00T's real latency win is elsewhere (Eagle VLM TRT engine, already goal-tracked).

### 3. Extend `reflex finetune`, don't fork

**Decision**: `reflex distill` is a `phase="distill"` variant of `run_finetune()`, NOT a separate subsystem.
**Evidence**: scope decision Section B — 80% of the distillation orchestration is identical to fine-tune (preflight schema check, backend invocation, checkpoint locate, LoRA merge, auto-export, VERIFICATION.md). Forking would duplicate this.
**Trade-off accepted**: the fine-tune module grows a `phase` enum + distill-specific hooks. Architecture doc Section B shows the exact diff.

### 4. Pro-gate the trainer, free-gate the consumer

**Decision**: `reflex distill` requires Pro license. `reflex export --from-distilled <hf_id>` stays free so anyone can deploy a community-distilled checkpoint.
**Evidence**: scope decision Section E — distillation compute is expensive ($100-400 per model), expertise is high (paper has no public code, 2-3 weeks of reimpl debugging), and a hard paywall here preserves the fine-tune free tier's goodwill.
**Trade-off accepted**: narrower top-of-funnel for distill. Elevates `stripe-license-gating` goal from weight 6 → weight 9 (becomes load-bearing for this feature).

### 5. Task-success gate, not just cos-parity

**Decision**: v0.3 shipping bar includes a `libero_drop_gate` hook that runs a LIBERO rollout comparison (teacher vs student) and blocks if student task success drops > 5pp.
**Evidence**: architecture axiom D3 (derived from VLA evidence Section C) — cos-parity at machine precision does NOT guarantee task success. Documented failures in Consistency Policy (5-9pp on Tool Hang, PushT) and ManiFlow (catastrophic on contact-rich without tactile) show real-task regression can coexist with clean numerical parity.
**Trade-off accepted**: shipping bar is higher than fine-tune's. Takes longer to reach v0.3 but the claim "distilled pi0 preserves task success" is load-bearing when made.

### 6. 3 prerequisite infra files before distill can ship

**Decision**: architecture requires `src/reflex/finetune/backends/base.py` + `hooks/__init__.py` + `postprocess.py` to land FIRST. These were called out in the fine-tune architecture (v0.5 shape) but haven't been implemented yet.
**Evidence**: architecture Section B — without `Backend` protocol + hook registry + postprocess chain, we can't extend fine-tune with `phase=distill` cleanly. Either we build them now as part of distill, or we back-fill fine-tune v0.5 first.
**Trade-off accepted**: distill PR train becomes 2-phase. Phase A: land the shared-infra refactor (adds value to fine-tune independently). Phase B: SnapFlow trainer + hooks.

### 7. Deprecate `pi_flow.py` + `dmpo.py` scaffolds

**Decision**: move existing `src/reflex/distill/{pi_flow,dmpo}.py` to `archive/v0.2/` on v0.3 cut.
**Evidence**: scope decision rejects pi-Flow + DMPO as v0.3 methods. Keeping scaffolds around creates maintenance drift and implies they're supported.
**Trade-off accepted**: git history preserves them if anyone needs to resurrect. Rename `GOALS.yaml::distill-dmpo` → `distill-snapflow`.

---

## Realistic expectations (honest pitch material)

From the VLA evidence + latency profile:

| Claim | Honest truth |
|---|---|
| Speedup | **3-4× end-to-end**, not 10× (VLM becomes new bottleneck at ~72% post-distill) |
| Denoise-only speedup | 10× (matches the 10→1 step reduction) |
| Task success delta (generic) | **[−5pp, +1pp]** range. SnapFlow's +1pp outlier is possible but not guaranteed |
| Contact-rich / precise manipulation | **−5 to −10pp** per documented failure modes. Warn customers |
| GPU cost to produce a distilled checkpoint | ~12 GPU-hours on A100 (~$30-60 Modal) |
| Training data needed | same as fine-tune dataset; no separate corpus |
| Real-robot (not sim) validation | **unpublished as of 2026-04-20**. reflex can't make real-robot claims until we measure |

---

## Skeptic rebuttals (straight from scope doc)

**Q: "OpenVLA-OFT / VLM2VLA are moving past flow-matching. You're distilling dying tech."**
A: True for 2027+. But in 2026-2027, the installed base (SmolVLA, pi0, pi0.5) is all flow-matching. Customers who bought Orin Nanos in 2026 want reactive control on pi0 NOW, not in 18 months when they migrate to VLM2VLA. reflex is a 12-18 month bridge, not a long-term moat. Scope decision Section C.

**Q: "If VLM becomes the new bottleneck (72% of E2E post-distill), you're optimizing for a future where customers will just pick a smaller VLM."**
A: Also partially true. But SmolVLA2 / tiny VLMs don't exist publicly yet. Current customers have pi0.5 + SmolVLM2 locked in by training cost. Distillation gives them a latency win NOW without retraining. When tiny-VLM-base models arrive, distillation remains additive on top.

**Q: "Physical Intelligence hasn't distilled pi0 publicly. Why do you think you know better?"**
A: Physical Intelligence has business reasons to keep distillation internal (licensing, enterprise pricing). Their lack of public release doesn't mean "didn't work." reflex's product-wedge is *public reproducibility* — our value add is the parity-gate + calibration + Pro distribution, not the math.

---

## What "done" looks like — per horizon

### v0.3 MVP (4-8 weeks from start)

**Prerequisite PR train (Phase A)** — land shared finetune infra:
- `src/reflex/finetune/backends/base.py` — Backend Protocol
- `src/reflex/finetune/hooks/__init__.py` — hook registry
- `src/reflex/finetune/postprocess.py` — finalize() chain

**Distill PR train (Phase B)** — actual distillation:
- `src/reflex/distill/snapflow.py` — pure math (~300 LOC, unit-testable without GPU)
- `src/reflex/finetune/backends/snapflow_backend.py` — glue to the Backend protocol
- `src/reflex/finetune/hooks/libero_drop_gate.py` — task-success kill-gate
- `src/reflex/cli.py` — rewrite existing `reflex distill` command (which currently imports DMPO)
- Modal script for end-to-end distill runs
- Tests: unit (math) + integration (LIBERO rollout comparison)

**Customer can do**:
```bash
reflex distill \
    --base lerobot/pi0_base \
    --teacher-steps 10 \
    --method snapflow \
    --dataset lerobot/libero \
    --output ./distilled_pi0 \
    --api-key $REFLEX_API_KEY   # Pro-gate
# → distilled_pi0/model.onnx (1-step) + VERIFICATION.md with task-success delta
```

**Shipping bar**: LIBERO task success ≥ teacher − 5pp AND cos-parity first-action ≥ +0.999. Measured via the `libero_drop_gate` hook. If either gate fails, status=`task_success_gate_failed`, no ONNX produced.

### v0.3.1 (conditional, 2-4 weeks after v0.3)

- SmolVLA support (requires velocity-convergence ablation first — our own adaptive-denoise data shows concerns)
- If SmolVLA fails the ablation → drop it, remain pi-family-only

### v0.5 (conditional, 3-6 months out)

- Consistency Policy trainer for GR00T DDPM (only if Eagle-VLM-TRT-on-Jetson lands first — otherwise GR00T's 58% VLM cost makes distillation's 35% denoise win pointless)
- `--distillation_method` enum extensibility to accept `consistency` alongside `snapflow`

---

## Kill-gates (committed, enforceable in code)

Per architecture Section I — six kill-gates mapped to concrete files and severities:

| Gate | When it fires | Enforcement | Override |
|---|---|---|---|
| Task success drop > 5pp | End of training, via `libero_drop_gate` hook | Hard block. No ONNX produced. | `--force-ship` Pro flag |
| cos-parity < 0.999 | Post-export, via existing parity gate | Hard block. | `--force-ship` Pro flag |
| Teacher checkpoint not exportable | Pre-flight | Hard block. Preflight fails. | None — fix the teacher first |
| Contact-rich benchmark drop > 10pp | End of training (v0.3.1) | Warning; ships with disclaimer | `--ack-contact-regression` |
| Student GPU OOM during training | Train-time | Auto-reduce batch size; if still fails, abort | None |
| Student inference slower than teacher on target hardware | Post-export latency check | Warning; ships anyway | None (signals methodology bug) |

---

## Monetization roadmap

| Tier | What they get | Landed in v0.3? |
|---|---|---|
| Free | `reflex export --from-distilled <hf_id>` — download + run community-distilled checkpoints | ✅ |
| Pro ($500/mo recommended, 3 distillations included) | `reflex distill` trainer + `--force-ship` override + priority Discord | ✅ (requires `stripe-license-gating` to ship first) |
| Enterprise ($2k-10k/mo) | Custom distillation recipes, on-call support, self-hosted license server | v1.0 |

**Load-bearing prerequisite**: `stripe-license-gating` (GOALS.yaml weight 6) must be elevated to weight 9 and shipped BEFORE `reflex distill` v0.3 launches. Without a working license system, we can't Pro-gate the trainer.

---

## Open product calls (defaults picked)

From architecture Section H, 5-10 A/B questions with defaults:

1. **Student init**: from teacher weights (default) vs fresh random. Teacher-init is faster convergence.
2. **Teacher input format**: HF model id (loads PyTorch) vs exported ONNX path. Default: HF id; ONNX adds export-to-training complexity.
3. **LoRA or full-weight student**: **full-weight**. LoRA adapters complicate the 1-step inference path.
4. **Preprocessor reuse**: reuse teacher's policy_preprocessor unchanged. Don't re-learn normalization.
5. **Auto-export after distillation**: **yes** (same pattern as fine-tune).
6. **LIBERO drop gate default threshold**: **5pp**. Override via `--task-success-tolerance`.
7. **Training data source**: same dataset as teacher's fine-tune. No separate corpus.
8. **Checkpoint save frequency**: every 2000 steps + final. Matches fine-tune.
9. **FP16 post-distill**: auto-run `fp16_convert` at export time. FP16 Orin Nano fit preserved.
10. **Distilled checkpoint hub**: publish to `reflex-vla/<base>-distilled` under a community Hugging Face org (v0.5+).

---

## What this does NOT include (defer or kill)

- No DDPM support v0.3 (GR00T → v0.5+ conditional)
- No pi-Flow or DMPO alternatives (deprecated on v0.3 cut)
- No real-robot validation (unpublished as of 2026-04-20 per literature)
- No task-specific distillation recipes (one-size-fits-all SnapFlow hyperparams)
- No multi-teacher distillation (ensemble of teachers → one student) — v1.0+
- No RL-fine-tune on top of distilled policy (DPPO-style) — out of scope

---

## Ready to build

Next step: implement the Phase A shared-infra files (Backend protocol + hooks + postprocess) in `src/reflex/finetune/`. This is independently useful for fine-tune v0.5 and unblocks distill. ~1-2 weeks.

After Phase A, Phase B (actual SnapFlow reimpl + hooks + CLI + Modal + tests): ~3-4 weeks.

**Total 4-8 weeks for v0.3 MVP** including the prerequisite refactor. Pro-gated from day 1. stripe-license-gating must ship in parallel.

Related:
- Full module plan: `distill_architecture.md` Section D
- Scope rationale: `distill_scope_decision.md`
- Evidence trail: `distill_methods_survey.md`, `distill_vla_evidence.md`, `distill_latency_profile.md`
- Inheritance: `finetune_architecture.md` Proposal 2, `finetune_SYNTHESIS.md`
