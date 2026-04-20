# `reflex distill` — Scope Decision

**Date:** 2026-04-19
**Author:** scope-decision agent
**Status:** COMMIT. Not a survey. Downstream architecture agent reads this and implements.
**Supersedes:** `2026-04-14-ship-distill-first` ADR (which predates the three research reports consumed here).

---

## Section A — The Decision

**Ship `reflex distill` in v0.3, scoped to pi0 and pi0.5 only, using a reimplemented SnapFlow trainer (flow-matching native, single paradigm). SmolVLA enters in v0.3.1 behind a kill-gate. GR00T is explicitly NOT in v0.3 and is gated behind the Eagle-VLM-TRT lever first. The `reflex distill` command reuses the `reflex finetune` Proposal 2 substrate (preflight, backend registry, hook layer, postprocess parity-gate) — it ships as a new backend phase, not a new subsystem. Monetization: Pro-gated at the trainer level (compute-heavy, expertise-dependent, publishable-quality-output moat), with the distilled ONNX artifact free to deploy and benchmark.**

One sentence: we are building a one-command SnapFlow reimplementation for the pi-family in v0.3 because denoise is 78-82% of E2E there, the paper gives a ~1pp LIBERO improvement as a proof point, and the engineering is ~300 LOC on top of infrastructure we already shipped last week.

---

## Section B — Evidence Table

| # | Question | Decision | Rationale + citation |
|---|---|---|---|
| 1 | Build at all? | **YES, conditional on kill-gates.** | pi0/pi0.5 denoise = 78-82% of E2E on cloud GPU (latency_profile Section B) → 3.3× E2E real, not wishful. SnapFlow (evidence Section A) gives the existence proof on real 3B VLAs with a +1pp LIBERO delta. Customer signal is thin (1/10 top pains is denoise-specific, latency_profile Section C), but competitive timing matters — LeRobot or OpenPI could ship this in a future release and collapse the wedge. The kill-gates (Section D) protect against the "unsolved research problem" scenario. |
| 2 | Which VLAs in v0.3? | **pi-family only (pi0 + pi0.5). SmolVLA in v0.3.1 behind a kill-gate. GR00T deferred to v0.5+.** | pi0/pi0.5: SnapFlow's exact tested configuration (evidence Section A row 1), 82% denoise fraction on A10G TRT (profile Section B), cleanest empirical + cleanest architecture fit. SmolVLA: 81% denoise but our own adaptive-denoise returned 0/25 triggers (profile Section E point 3) — velocity field doesn't compress cleanly, so distillation is the only path but also the riskiest. GR00T: 35% denoise on RTX 5090 (profile Section B) caps distillation at 1.35× E2E; the higher-leverage lever is Eagle VLM TRT-buildability on Jetson (profile Section D ranking), which closes the 216.5→~130 ms gap without touching distillation. |
| 3 | Which methods / trainers in v0.3? | **One trainer: reimplemented SnapFlow for flow-matching. No DDPM path in v0.3. `pi_flow.py` and `dmpo.py` scaffolds are deprecated on the v0.3 cut.** | methods_survey Section B picks SnapFlow as primary: paper-validated on pi0.5 and SmolVLA, self-distillation (no teacher copy), ~300 LOC reimplementation, ~200-400 A10G-hours. Consistency Policy (DDPM) is the natural pair — but it only lights up GR00T (see Q2: deferred) so it has no v0.3 customer. pi-Flow is superseded by SnapFlow on 1-NFE quality (methods_survey 40 vs 60). DMPO is from-scratch not distillation (evidence Table row on DMPO) — different product. One paradigm, one trainer keeps the axiom from finetune_architecture (A7: "training backend identity is a single enum"). |
| 4 | Integration with existing reflex stack? | **Reuse 80% of `reflex finetune` Proposal 2 substrate. Distill is a second `phase` value alongside `imitation`, not a separate subsystem.** | finetune_architecture Section F7 explicitly anticipates `FinetuneConfig.phase = imitation \| rl \| distill`. Preflight (schema, memory, norm_stats, dataset_size), backend registry, hook layer, postprocess (export → parity-gate → calibration → VERIFICATION.md) all transfer. What's new: (a) `src/reflex/distill/snapflow.py` trainer (~300 LOC), (b) `action_heads/distilled_flow_matching.py` strategy (zero-init time embed + consistency loss wiring), (c) `preflight/teacher_export.py` (verify teacher VLA cos-parity before distilling — we already export it), (d) `hooks/libero_drop_gate.py` (kill-switch on task-success regression). Nothing else moves. |
| 5 | Monetization posture? | **Pro-tier gated at the trainer. Distilled artifact is free to deploy, serve, and benchmark. Calibration + parity-gate reports are still Pro as in finetune.** | finetune_SYNTHESIS section 8 sets the pattern: finetune free, validators Pro. Distillation asymmetry justifies gating the trainer itself: (a) it consumes real compute ($100-400 per distilled model), (b) it requires VLA-specific expertise to tune (SnapFlow's "2-3 weeks debugging" warning, methods_survey Section E point 1), (c) the output is a publishable artifact (cos=1.0 distilled ONNX) that creates an asymmetric moat vs DIY. Free-tier users can *consume* a community-published distilled checkpoint from HF Hub via `reflex export --from-distilled`; Pro users *produce* them. This also hedges the monetization-blocker goal (GOALS.yaml `stripe-license-gating` weight 6) — distillation is the first feature that deserves a hard paywall. |

---

## Section C — Skeptic Rebuttal

**Skeptic 1: "OpenVLA-OFT and VLM2VLA are moving past flow-matching. You're distilling a dying technique. Why?"**

The skeptic is half-right and we are doing it anyway. OFT's L1 regression + parallel decoding (26× speedup, arXiv 2502.19645) and VLM2VLA's action-as-language (arXiv 2509.22195) are structurally better solutions to the latency problem — no denoise loop to distill. In a 2028 world where 60%+ of deployed VLAs ship without a denoise loop, `reflex distill` has no flow-matching customer.

But the installed base in 2026 is pi0, pi0.5, SmolVLA, GR00T — all flow-matching (or flow-matching-adjacent for GR00T). Customers who have a pi0.5 checkpoint in production *today* can't retrain to an OFT-style head — that's a months-long data + compute effort to reproduce teacher quality, not an overnight refactor. `reflex distill` is a 12-18 month bridge capability for that installed base. Methods survey Section C concludes the same: not misprioritized *if* scoped to installed base. We scope it to installed base. The F1/F8 forward-compat hooks in finetune_architecture already anticipate the OFT transition (add a new `action_head` strategy, register, done). When OFT becomes dominant we retire the flow-matching-only distill trainer and route customers to `reflex finetune --head l1_regression` instead.

Kill-gate 2 (Section D) protects us from doubling down: if OFT-style models cross 40% of the VLA market by Q4 2026, we freeze `reflex distill` flow-matching work and pivot to `reflex finetune --head l1_regression` as the faster path to 100Hz.

---

**Skeptic 2: "If VLM is the new bottleneck post-distill (~72% of E2E), you're optimizing for a future where denoise dominates — but by then, customers will use a faster base VLA (SmolVLA2, tiny models). Why is this worth doing today?"**

The skeptic is partially right but the math still works. SnapFlow itself (evidence Section C.1) admits the post-distill VLM bottleneck — "with denoising compressed to one step, the VLM prefix (60 ms) becomes the new bottleneck (72% of E2E)." For pi0.5 on A800 PyTorch, 274 ms → 83 ms. The customer-facing win is a 3.3× E2E speedup, not 10×.

But 3.3× is already the single largest latency lever left in `reflex`'s shipped stack. Latency_profile Section D ranks levers for the pi-family on A10G/Thor:
- Lever 1 (TRT FP16): 2.6-3.3× — **already shipped**
- Lever 2 (distill 10→1): 3.3× E2E on pi0.5 — **this proposal**
- Lever 4 (FP8/NVFP4 on Thor): 1.7× — Thor-only, ~zero eng risk, but narrower hardware reach

Levers 1 and 2 stack. Stacked E2E on pi0.5: 274 ms (PyTorch) → 91 ms (TRT FP16) → 28 ms (+ distill). At 28 ms, pi0.5 hits 35 Hz on A10G — the threshold where "reactive control" becomes a realistic claim for a cloud-hosted VLA. Without distillation we cap at ~11 Hz on A10G. That's the difference between "deploy-to-cloud viable for real-time" and "deploy-to-cloud for batch inference only."

The "future where customers use SmolVLA2" argument defers compute budget to a hypothetical model that doesn't exist in Apr 2026. pi0 is 6 months old in production, pi0.5 is newer, SmolVLA is 10 months old. These models have 2-3 years of installed-base tail. We're not optimizing for a future bottleneck — we're cutting the current bottleneck in half for the installed base.

---

**Skeptic 3: "Physical Intelligence has the talent and compute to ship a distilled pi0. They haven't, which means it's either not worth it or not working. Why do you know better?"**

We don't know better. We have the same information. The skeptic's inference is one valid read. The alternative reads (evidence Section B) are: (a) PI has a distilled pi0 internally and keeps it as competitive moat, (b) PI's customers haven't asked for it because they pay for per-query inference and PI absorbs the compute, (c) PI's engineering bandwidth is on pi0-FAST (autoregressive, slower) which is their *different* speed-lever bet.

The parsimonious read that supports the skeptic is bet (c): if distillation were trivially winning, PI would have shipped it before pi0-FAST. pi0-FAST is them explicitly rejecting the "fast inference is the customer problem to solve today" framing.

But here's the honest asymmetry: **PI's job is to build VLAs. Reflex's job is to deploy them.** PI shipping a distilled pi0 would be them moving into Reflex's wedge. They haven't — so the deployment wedge is still open. If PI ships a distilled pi0 with open weights before we ship v0.3, we kill the flow-matching distill track and become the *orchestration layer* for PI's distilled artifacts (kill-gate 3, Section D). That's a worse-but-still-viable product — call it "reflex orchestrate" — which runs multiple distilled checkpoints behind one endpoint and handles the fallback logic.

Honest floor: the probability PI ships an openly-distilled pi0 by v0.3 ship date (~8 weeks) is non-zero but low — they've shipped pi0, pi0.5, pi0-FAST, and Knowledge Insulation in the past 18 months and none of those were distillation work. We bet "low enough to ship our own."

---

## Section D — Kill Criteria

Explicit thresholds at which reflex abandons or rescopes the distill project. Each is a boolean check that a product-call can make without re-litigating the scope decision.

**Gate 1 — Single-VLA quality floor.**
- **Trigger:** v0.3 SnapFlow reimpl on pi0 LIBERO-10 N=25 drops more than **8pp** vs PyTorch native teacher (teacher baseline: 40% → student must be ≥32%).
- **Action:** Kill flow-matching distill track. pi-family deferred to v0.5. Reflex's `distill` command publishes a honest "under investigation" status.
- **Why 8pp:** SnapFlow paper claims +1pp; our reimpl budget is 5pp ("median case" from evidence Section E); 8pp is a 3σ-ish miss that indicates reimplementation failure, not teacher variance.

**Gate 2 — OFT/L1-regression market share.**
- **Trigger:** OpenVLA-OFT-style regression-head VLAs (or any successor with `head_type=regression` and no denoise loop) cross **40% of new deployed VLAs** by Q4 2026, measured by HF Hub download stats + lerobot issue frequency proxies.
- **Action:** Rescope `reflex distill` from "accelerate flow-matching" to "OFT-native deployment support." Retire SnapFlow reimpl (move to `archive/`); prioritize `reflex finetune --head l1_regression` as the speed path. Existing distilled artifacts from v0.3 customers remain deployable (no breaking change).
- **Why 40%:** below this threshold, flow-matching is still the dominant installed base. Above it, the bridge has ended.

**Gate 3 — Physical Intelligence / NVIDIA ships a distilled open-weight VLA.**
- **Trigger:** PI releases a distilled pi0/pi0.5 with open weights **OR** NVIDIA releases a distilled GR00T N1.x with open weights, before reflex v0.3 ships.
- **Action:** Abandon our SnapFlow reimpl. Pivot to **"reflex orchestrate"**: a serve-layer wrapper that routes requests to the vendor's distilled checkpoint, handles fallback when confidence is low, and produces calibration reports. Monetization posture unchanged (orchestration is still Pro-tier complexity).
- **Why:** If PI ships a distilled pi0, they'll claim cos=1.0 parity and task-success delta from a position we can't contest. Our distilled reimpl becomes a strictly-worse second-source. The orchestration layer is defensible; the trainer is not.

**Gate 4 — SmolVLA distill structural failure.**
- **Trigger:** v0.3.1 SmolVLA LIBERO drop at N=25 exceeds **10pp** vs teacher OR training loss doesn't converge within 2× the SnapFlow paper's 12-hour budget (i.e., >24 GPU-hours without a loss plateau below teacher MSE).
- **Action:** Defer SmolVLA indefinitely. Document in CHANGELOG.md as "not supported." Escalate to an engineering spike only when a paying customer requests it with signed intent.
- **Why 10pp:** SmolVLA's adaptive-denoise returned 0/25 triggers, signaling velocity field is structurally harder to compress than pi0's (latency_profile Section E point 3). This is the empirically-expected failure mode; we need headroom to tell it apart from reimpl bugs.

**Gate 5 — GR00T DDPM deferral.**
- **Trigger:** Before any GR00T distill work starts, Eagle VLM must TRT-build on Jetson (closes the 216.5 → ~130 ms gap per latency_profile Section D). If this ships and GR00T denoise is still the measured bottleneck on measured customer hardware, revisit.
- **Action (v0.3):** GR00T distill is explicitly NOT on the roadmap. If asked, point at the VLM-TRT lever first.
- **Why:** GR00T denoise is 35% of E2E. 4→1 distill caps at 1.35× E2E win. VLM TRT is 58% of E2E (per Isaac-GR00T README RTX 5090 breakdown). Higher-leverage lever.

**Gate 6 — Customer signal floor.**
- **Trigger:** If between v0.3 ship and 60 days post-ship, fewer than **3 paying Pro-tier customers activate `reflex distill`**, downgrade from a v0.5 expansion item to a maintained-but-not-invested-in feature.
- **Action:** Freeze new features. Keep SnapFlow trainer alive at feature parity with whatever it is on the v0.3 cut date. Redirect roadmap compute to the next wedge (TBD).
- **Why:** customer-signal analysis (latency_profile Section C) already shows only 1/10 customer pains is denoise-specific. If the wedge is smaller than expected, distill deserves maintenance mode, not continued investment.

---

## Section E — First Customer Release Shape

**v0.3 customer-facing command:**

```bash
reflex distill \
  --base lerobot/pi05_libero \
  --method snapflow \
  --dataset lerobot/libero_10 \
  --teacher-export ./exports/pi05_libero/ \
  --output ./distilled/pi05_1nfe/ \
  --target-nfe 1 \
  --gpus 2 \
  --calibration-gate 0.2 \
  --libero-drop-gate 8pp
```

**What the customer gets in `./distilled/pi05_1nfe/`:**

1. **`model.onnx`** — the distilled 1-NFE pi0.5 as a monolithic ONNX (cos-parity verified vs distilled PyTorch at ≥0.999).
2. **`reflex_config.json`** — same format as today's export. `num_steps=1` is baked in, with a `distill_provenance: {method: snapflow, teacher_export: ..., teacher_nfe: 10, student_nfe: 1, training_run_id: ...}` block.
3. **`VERIFICATION.md`** — the existing verification report format, extended with new sections:
   - **Parity** — cos vs distilled PyTorch at shared noise (required ≥0.999).
   - **Calibration** — ECE, Brier, NLL on held-out libero_10 shard.
   - **Task success** — LIBERO-10 N=25 success rate for student vs teacher. **This is the load-bearing number**: if it shows >8pp drop vs teacher, the export fails Gate 1 and exits non-zero.
   - **Latency** — measured per-step and E2E ms for student on A10G TRT FP16 (reflex's reference hardware).
   - **Distillation provenance** — teacher ID, method version, training hours, LIBERO task-success delta per task (not just aggregate — tight-tolerance tasks like Tool Hang get flagged).
4. **`training_log.jsonl`** — per-step loss + LIBERO smoketest every 1000 steps. Same format as `reflex finetune` emits.
5. **`CALIBRATION_DRIFT.md`** — a new artifact specific to distillation: how calibration metrics evolved student-vs-teacher across training. First time we've ever measured this; critical for the "distilled VLA confidence is still trustworthy" pitch.

**What the customer does not get in v0.3:**

- GR00T distill. (Gate 5; explicit kill.)
- SmolVLA distill. (Gate 4; v0.3.1 or later.)
- DDPM support. (No customer in v0.3.)
- Training-time RL (DMPO). (Different product; scaffold stays in `dmpo.py` unused.)

**What v0.3 looks like from the `reflex serve` side (zero change):**

```bash
reflex serve ./distilled/pi05_1nfe/ --target a10g
```

The monolithic ONNX contract is unchanged — `num_steps=1` is baked in, the runtime doesn't know or care that the weights came from distillation. This is the same promise `reflex finetune → reflex export → reflex serve` already delivers. Distillation is a *training-time intervention*; the deploy path is untouched.

**Demo shape for v0.3 launch post (60-day post-ship target):**

> "`reflex distill` on pi0.5: 83 ms end-to-end on A10G TRT FP16 (vs 274 ms teacher), 98.0% LIBERO-10 task success (vs 97.75% teacher), 400 A10G-hours to produce. The calibration report in VERIFICATION.md shows ECE within 5% of teacher. Distilled model.onnx checksummed and signed — your Orin Nano deployment works the same as your A10G deployment."

That's the promise. The kill-gates above protect us from shipping that sentence with a "*" footnote.

---

## Compromises accepted

- **Single paradigm.** No DDPM trainer in v0.3. GR00T customers get a "not yet supported" message. This is the right move given GR00T's 35% denoise fraction, but it looks like scope timidity from the outside. Accept.
- **Single method.** No pi-Flow fallback, no Consistency Policy pair. If SnapFlow reimpl breaks in a fundamental way, there's no trainer B to fall back to — Gate 1 becomes the kill. Accept for velocity.
- **Pro gating may narrow adoption.** Free tier produces distilled outputs for existing `reflex finetune` users; Pro-gate on the trainer reduces top-of-funnel activation. Accept because the compute-cost asymmetry is real and the monetization story matters for v0.3 ship.
- **Zero real-robot validation.** LIBERO sim-only, like everyone else in the literature (evidence Section C.1 flags this as SnapFlow's own limitation). We ship with this caveat loud in VERIFICATION.md and in the launch post — if a real-robot deployment shows regression, it informs Gate 1's threshold.
- **SmolVLA deferred to a v0.3.1 point release.** Customers expecting all-4-VLAs coverage at v0.3 will be disappointed. The decision is defensible (Gate 4 reasoning) but it's a product pitch compromise.

## Needs product call

- **v0.3 ship date.** This doc assumes ~8 weeks from today (Apr 19 → Jun 14). If the calendar is tighter, cut SmolVLA entirely (no v0.3.1) and ship pi-family only. If the calendar is looser, fold SmolVLA v0.3.1 into the v0.3 ship.
- **Pro tier list price.** Distillation-per-model cost scaling ($100-400) should be reflected in Pro pricing. Current finetune_SYNTHESIS doesn't commit a number. Recommend: Pro tier $500/mo, includes 3 distillations/month included + $100 per additional. Escalate.
- **Stripe licensing code.** GOALS.yaml `stripe-license-gating` weight 6 is open. This goal becomes load-bearing if distill ships as the first paid feature. Elevate weight 6 → weight 9, or distill stays free at launch.

---

## File references (for the downstream architecture agent)

Absolute paths:
- `/Users/romirjain/Desktop/building projects/reflex-vla/reflex_context/01_architecture/distill_methods_survey.md` (primary: SnapFlow, pi-Flow, Consistency Policy picks)
- `/Users/romirjain/Desktop/building projects/reflex-vla/reflex_context/01_architecture/distill_vla_evidence.md` (primary: empirical baseline, industry-silence evidence)
- `/Users/romirjain/Desktop/building projects/reflex-vla/reflex_context/01_architecture/distill_latency_profile.md` (primary: per-VLA denoise fractions, kill-gate thresholds)
- `/Users/romirjain/Desktop/building projects/reflex-vla/reflex_context/01_architecture/finetune_architecture.md` (reuse substrate — Proposal 2)
- `/Users/romirjain/Desktop/building projects/reflex-vla/reflex_context/01_architecture/finetune_SYNTHESIS.md` (monetization posture precedent)
- `/Users/romirjain/Desktop/building projects/reflex-vla/GOALS.yaml` (current goal set — `distill-dmpo` weight 9 gets rewritten as `distill-snapflow` after this decision lands)
- `/Users/romirjain/Desktop/building projects/reflex-vla/src/reflex/distill/pi_flow.py` (deprecated scaffold; archive on v0.3 cut)
- `/Users/romirjain/Desktop/building projects/reflex-vla/src/reflex/distill/dmpo.py` (deprecated scaffold; keep in archive/ for future phase=rl product)
- `/Users/romirjain/Desktop/building projects/reflex-vla/src/reflex/finetune/` (reuse target: preflight, backends, hooks, postprocess)
