# `reflex distill` — Architecture

**Date:** 2026-04-19
**Author:** architecture agent
**Status:** Opinionated. One concrete recommended architecture. Senior engineer implements from this without follow-up questions.
**Primary input:** `reflex_context/01_architecture/distill_scope_decision.md` (scope is committed; do not relitigate).
**Scope recap:** v0.3 ships pi0 + pi0.5 distillation via a reimplemented SnapFlow trainer. SmolVLA is v0.3.1 behind a kill-gate. GR00T deferred to v0.5+. Pro-gated at the trainer; distilled ONNX free to deploy. Integration: `phase=distill` is a variant of the `reflex finetune` Proposal-2 in-process trainer registry, NOT a parallel subsystem.

---

## Section A — Design axioms

Seven axioms specific to distillation. Fine-tune axioms A1 (thin orchestrator), A2 (validation moat), A5 (LeRobotDataset v3 native), A6 (export-on-success) are inherited unchanged.

**D1. SnapFlow loss math is first-party; gradient math is not.**
Source: `distill_methods_survey.md` §E.1 — no public code as of 2026-04-19, ~300 LOC reimpl.
Delta vs finetune A1: distillation requires writing the SnapFlow *loss* (consistency loss on 2-step Euler shortcut velocity + zero-init target-time embedding) because it doesn't exist upstream. Optimizer step, FSDP, data loader still come from lerobot. We own the loss; we do not own the loop.
Forbids: reimplementing AdamW, custom state-dict, departing from lerobot's checkpoint layout.

**D2. Teacher identity is first-class.**
Source: scope §E — `reflex_config.json` must ship a `distill_provenance` block; evidence §E flags first-party benchmarking as mandatory.
Forbids: producing a distilled ONNX indistinguishable from a fine-tuned one in metadata, silent teacher substitution (provenance mismatch is an export-time error).

**D3. Task-success gate is load-bearing; cos=1.0 is necessary but not sufficient.**
Source: evidence §C.1 — naive 1-NFE on pi0.5 gets 96.75% vs 97.75% with high per-task variance; scope §D Gate 1 — 8pp drop kills the track.
Delta vs finetune A2: collapsing 10→1 steps re-parameterizes the velocity field; cos-parity vs *distilled* PyTorch at shared noise does NOT prove task success. We MUST run LIBERO against the *teacher* baseline before shipping.
Forbids: declaring SUCCESS on cos parity alone; skipping LIBERO sim eval; treating fine-tune's gate set as complete.

**D4. Distilled ONNX is indistinguishable from fine-tuned ONNX at export.**
Source: scope §E — "the monolithic ONNX contract is unchanged — `num_steps=1` is baked in, the runtime doesn't know or care that the weights came from distillation."
Forbids: a `reflex serve --distilled-mode` flag, a separate ONNX format, runtime decision logic. Export uses the unchanged `src/reflex/exporters/monolithic.py` chain with `num_steps=1`.

**D5. No unified flow/DDPM abstraction. Paradigm leak into the trainer is acceptable.**
Source: methods_survey §D — "Do NOT attempt a unified trainer — the paradigm abstraction would leak." Scope §B Q3 commits to one paradigm in v0.3.
Forbids: a `BaseDistillTrainer` superclass spanning velocity- and noise-prediction; premature abstraction for GR00T (deferred to v0.5+).

**D6. Pro-gating lives at trainer entry, not at artifact consumption.**
Source: scope §B Q5.
Forbids: gating the export chain, `reflex serve`, `reflex validate`, or runtime license checks. One guard at `run_finetune()` entry when `phase=distill`.

**D7. Training stability is a product risk, not a bug. Intermediate observability is load-bearing.**
Source: methods_survey §E.1 — "2-3 weeks of debugging before first clean result." Evidence §E — floor 30-35% if teacher is 40%.
Delta vs finetune: lerobot ships tested profiles; distillation is research-grade. The architecture must surface loss curves, LIBERO smoketests every N checkpoints, intermediate parity — failing runs are caught at hour 3, not hour 24.
Forbids: shipping with a "train and pray" endpoint that lacks mid-training observability.

---

## Section B — The `phase=distill` integration

### B.1 Plug-in point

Today `src/reflex/finetune/` has `config.py`, `run.py` (subprocess-lerobot-train shape), `cli.py`, and `preflight/` (schema + dataset_size). It does NOT yet have `backends/`, `hooks/`, `postprocess.py` — those are Proposal-2 target shape.

**Distill cannot ship without three Proposal-2 prerequisites landing first** (~200 LOC, shared with finetune v0.5):
1. `src/reflex/finetune/backends/base.py` — `Backend` Protocol with `.fit(ctx) -> CheckpointResult`.
2. `src/reflex/finetune/hooks/__init__.py` — `HookRegistry` + `TrainerContext.hooks`.
3. `src/reflex/finetune/postprocess.py` — `finalize(ckpt_dir, cfg) -> FinetuneResult`.

### B.2 `FinetuneConfig` additions

Edit `src/reflex/finetune/config.py` — all new fields default-valued, backwards compatible:

```
phase: Literal["imitation", "distill"] = "imitation"
distillation_method: Literal["snapflow"] | None = None
teacher_export: Path | None = None
teacher_nfe: int = 10
student_nfe: int = 1
libero_drop_gate_pp: float = 8.0        # scope §D Gate 1
libero_drop_gate_tasks: list[int] = [0..9]
libero_drop_gate_n_episodes: int = 25
distill_provenance_pointer: str | None = None   # populated by trainer
```

### B.3 CLI: `reflex distill` and `reflex finetune --phase distill`

Both work. `reflex distill` is the customer-facing command (scope §E); `--phase distill` is the underlying dispatcher. The existing `reflex distill` at `src/reflex/cli.py:1121` is REWRITTEN (not new): drop DMPO/pi-Flow imports; build `FinetuneConfig(phase="distill", distillation_method="snapflow", ...)`; call `run_finetune(cfg)`. Delete the hidden DMPO helper commands at lines 1192/1207/1222/1237.

### B.4 Preflight additions

New files under `src/reflex/finetune/preflight/`:

- `teacher_export.py::check_teacher_export` — validates the directory has `model.onnx` + `reflex_config.json`; resolves teacher's `model_type`.
- `teacher_paradigm.py::check_teacher_paradigm` — asserts `model_type in {"pi0", "pi05"}` for v0.3 (rejects SmolVLA/GR00T per scope §D Gates 4-5); teacher `num_denoising_steps` matches `cfg.teacher_nfe`.
- `pro_license.py::check_pro_license` — Pro subscription check when `phase=distill` (Section I Gate A).

Edit `preflight/runner.py`:

```
check_fns = (check_schema, check_dataset_size)
if cfg.phase == "distill":
    check_fns += (check_pro_license, check_teacher_export, check_teacher_paradigm)
```

### B.5 Backend dispatch

`src/reflex/finetune/backends/__init__.py::resolve_backend(cfg)`:

```
if cfg.phase == "distill":
    if cfg.distillation_method != "snapflow":
        raise ValueError(...)
    return SnapFlowBackend()
return LerobotBackend()
```

The existing subprocess code in `run.py` moves into `LerobotBackend.fit()` — migration, not new feature. `Backend.fit()` is paradigm-agnostic; both backends return the same `CheckpointResult`.

---

## Section C — SnapFlow trainer module

### C.1 Layout

- `src/reflex/distill/snapflow.py` — SnapFlow math: flow-matching loss, consistency loss, zero-init target-time embedding. ~300 LOC. Unit-testable without GPU.
- `src/reflex/distill/teacher_loader.py` — load teacher PyTorch policy from reflex-exported dir; freeze + eval.
- `src/reflex/finetune/backends/snapflow_backend.py` — thin glue implementing `Backend` protocol. Loads teacher + student, runs the SnapFlow loop from `snapflow.py`, wires hooks, saves checkpoints.

The split is intentional: research-y math isolated (distill/), protocol-conforming glue in the finetune package. When Consistency Policy lands in v0.5+, `src/reflex/distill/consistency.py` joins alongside; backend layer stays clean.

### C.2 `TrainerContext` + `CheckpointResult`

`TrainerContext` carries: `config: FinetuneConfig`, `hooks: HookRegistry`, `teacher_path: Path`, `training_log_path: Path`.
`CheckpointResult` carries: `final_checkpoint_path`, `training_steps_completed`, `intermediate_metrics: dict`.

### C.3 Training-step pseudocode

```
def fit(self, ctx):
    teacher = load_teacher_pytorch(ctx.config.teacher_export)
    teacher.eval(); freeze(teacher)

    student = copy.deepcopy(teacher)
    _install_target_time_embedding(student, init_scale=0.0)  # SnapFlow trick
    student.train()

    opt = AdamW(student.parameters(), lr=ctx.config.learning_rate)
    loader = build_dataloader(ctx.config.dataset, ctx.config.batch_size)

    for step, batch in enumerate(loader):
        # (a) Flow-matching term
        t = torch.rand(batch_size)
        x_t, v_target = fm_interp(batch.action, t, noise=randn())
        v_student = student.velocity(batch.obs, batch.lang, x_t, t)
        fm_loss = mse(v_student, v_target)

        # (b) SnapFlow consistency term: 2-step Euler shortcut from teacher
        with torch.no_grad():
            v_t = teacher.velocity(batch.obs, batch.lang, x_t, t)
            x_mid = x_t + 0.5 * v_t
            v_shortcut = teacher.velocity(batch.obs, batch.lang, x_mid, t + 0.5)
        v_one = student.velocity(batch.obs, batch.lang, x_t, t, target_time=1.0)
        consistency_loss = mse(v_one, v_shortcut)

        loss = fm_loss + alpha * consistency_loss
        opt.zero_grad(); loss.backward(); opt.step()

        ctx.hooks.run("on_step", step=step, loss=loss.item())
        if step % ctx.config.checkpoint_every == 0:
            ckpt = save_checkpoint(student, step)
            ctx.hooks.run("on_checkpoint", step=step, ckpt=ckpt)

    return CheckpointResult(
        final_checkpoint_path=save_checkpoint(student, ctx.config.num_steps),
        training_steps_completed=ctx.config.num_steps,
        intermediate_metrics=...,
    )
```

Real `(obs, lang)` is mandatory (methods_survey §B.1 — "noise-only does NOT work").

### C.4 Checkpoint format

**Full weights, NOT a LoRA adapter.** Two reasons: (i) scope §E requires a monolithic `model.onnx` — `export_monolithic` wants a self-contained checkpoint, (ii) Hub publishing wants self-contained weights, not adapters. Student starts from a full-weight copy of teacher and saves full weights directly; no merge step needed.

Format matches lerobot's `pretrained_model/` layout (`model.safetensors` + `config.json` + `policy_preprocessor.json` + `policy_postprocessor.json`). The existing `_auto_export` path in `run.py` works unchanged.

Student's `config.json` carries `distill_provenance`:
```
{"method": "snapflow", "teacher_export": "...", "teacher_nfe": 10,
 "student_nfe": 1, "training_run_id": "<uuid>", "trained_at": "<iso>"}
```

---

## Section D — Module / file plan

All paths relative to repo root. Senior engineer implements in 3-4 weeks.

### D.1 Deprecate existing scaffolds (scope §B Q3)

- `src/reflex/distill/pi_flow.py` → `archive/v0.2/distill/pi_flow.py`.
- `src/reflex/distill/dmpo.py` → `archive/v0.2/distill/dmpo.py` (stays archived; future `phase=rl` hook per finetune F7).
- `src/reflex/distill/__init__.py::get_recipe` rewritten to raise `ValueError("Deprecated in v0.3; use snapflow. See distill_scope_decision.md")` for any non-snapflow name.
- `GOALS.yaml::distill-dmpo` renamed to `distill-snapflow`; check expression updated to `test -f src/reflex/distill/snapflow.py`.
- Tests importing `DMPOTrainer`/`PiFlowTrainer` updated to assert deprecation error or removed.

### D.2 New Python files

| File | Purpose |
|---|---|
| `src/reflex/distill/__init__.py` | Rewrite. `get_method("snapflow")` registry; all other names error. |
| `src/reflex/distill/snapflow.py` | SnapFlow math (fm loss + consistency loss + zero-init embed installer). |
| `src/reflex/distill/teacher_loader.py` | Load teacher PyTorch from reflex-export dir; freeze, eval, resolve `model_type`. |
| `src/reflex/finetune/backends/base.py` | Shared infra. `Backend` Protocol + `TrainerContext`. |
| `src/reflex/finetune/backends/__init__.py` | Shared infra. `resolve_backend(cfg)`. |
| `src/reflex/finetune/backends/lerobot_backend.py` | Migration: wraps existing `run.py` subprocess code behind `Backend`. |
| `src/reflex/finetune/backends/snapflow_backend.py` | Glue: loads teacher + student, runs SnapFlow loop, wires hooks. |
| `src/reflex/finetune/hooks/__init__.py` | Shared infra. `HookRegistry`. |
| `src/reflex/finetune/hooks/libero_drop_gate.py` | Load-bearing (Section E). End-of-training LIBERO teacher-vs-student eval. |
| `src/reflex/finetune/hooks/libero_smoketest_hook.py` | Periodic LIBERO eval during training. NOT a gate. |
| `src/reflex/finetune/hooks/parity_intermediate_hook.py` | Cos-parity check every N checkpoints; detects divergence early. |
| `src/reflex/finetune/postprocess.py` | Shared infra. `finalize()` runs export → validate_roundtrip → libero_drop_gate → write_verification_report. |
| `src/reflex/finetune/preflight/teacher_export.py` | Validate teacher export dir. |
| `src/reflex/finetune/preflight/teacher_paradigm.py` | Assert teacher `model_type` in v0.3 allowlist. |
| `src/reflex/finetune/preflight/pro_license.py` | Pro subscription check when `phase=distill`. |
| `src/reflex/finetune/profiles/distill_pi0_snapflow.py` | Reproducibility profile: pi0 + SnapFlow on LIBERO-10. |
| `src/reflex/finetune/profiles/distill_pi05_snapflow.py` | Same for pi0.5. |
| `src/reflex/finetune/templates/modal_distill.py` | Modal scaffold. |
| `src/reflex/finetune/cli_distill.py` | Thin Typer wrapper building `FinetuneConfig(phase="distill", ...)`. |

### D.3 Edits to existing files

| File | Edit |
|---|---|
| `src/reflex/finetune/config.py` | Add 9 fields from B.2. All defaulted. |
| `src/reflex/finetune/run.py` | Refactor: replace direct `_run_lerobot_training()` with `resolve_backend(cfg).fit(ctx)`. Existing subprocess code moves to `lerobot_backend.py`. |
| `src/reflex/finetune/preflight/runner.py` | Conditional checks for `phase=distill` per B.4. |
| `src/reflex/finetune/cli.py` | Add `--phase`, `--teacher`, `--teacher-nfe`, `--libero-drop-gate` flags. Non-distill defaults. |
| `src/reflex/cli.py` | Rewrite `distill` command body at line 1121 per B.3. Delete hidden DMPO commands 1192/1207/1222/1237. |
| `src/reflex/verification_report.py` | Add optional `distill_report=...` kwarg → renders "Distillation" section with teacher/student NFE + per-task task-success delta + provenance. |
| `src/reflex/exporters/pi0_exporter.py` + `pi05_exporter.py` | Propagate `distill_provenance` from source config into exported `reflex_config.json`. Small additive edit. |
| `src/reflex/eval/calibration.py` | Add `run_on_distilled(teacher_dir, student_dir, dataset, n_samples) -> CalibrationDriftReport`. Existing metric functions untouched. |
| `GOALS.yaml` | Rename `distill-dmpo` → `distill-snapflow`; update check. |

### D.4 Tests

| File | Purpose |
|---|---|
| `tests/test_distill_snapflow_loss.py` | Unit: loss math on synthetic tensors; zero-init embed behavior; no GPU. |
| `tests/test_distill_teacher_loader.py` | Unit: paradigm detection, mock export dir, error cases. |
| `tests/test_distill_preflight.py` | Unit: teacher_export + teacher_paradigm + pro_license on fixtures. |
| `tests/test_distill_config.py` | Unit: `FinetuneConfig(phase="distill")` validation; CLI parsing. |
| `tests/test_distill_libero_drop_gate.py` | Unit: hook with mock rollout runner; pass/fail at threshold. |
| `tests/test_distill_cli_smoke.py` | Integration: `reflex distill --help`; `reflex distill ... --dry-run`. |
| `tests/test_distill_end_to_end_pi0.py` | Integration (Modal, daily CI, GPU-gated): 200-step pi0 distill on LIBERO-10 subset; asserts libero_gate runs cleanly + ONNX exports. |

### D.5 Modal scripts

| File | Purpose |
|---|---|
| `scripts/modal_reflex_distill.py` | Customer-facing cloud scaffold. Mirrors finetune's Modal scaffold. |
| `scripts/modal_distill_libero_gate.py` | Wraps existing `scripts/modal_libero_monolithic_onnx.py` for student-vs-teacher eval. |

### D.6 Docs

| File | Purpose |
|---|---|
| `docs/distill.md` | How-to: `reflex distill`, `--teacher-export`, kill-gate semantics. |
| `docs/distill_recipes.md` | Profiles: pi0 LIBERO, pi0.5 LIBERO; A10G-hours + Modal budget. |
| `docs/distill_troubleshooting.md` | SnapFlow failure modes → fixes. Explicit contact-rich task caveat (evidence §C.1). |
| `docs/architecture/distill.md` | Contributor pointer at this doc. |
| `CHANGELOG.md` + `README.md` | v0.3 entries. |

---

## Section E — The `libero_drop_gate` hook

Per scope §B Q4, new hook separate from the parity gate.

### E.1 Registry integration

`src/reflex/finetune/hooks/libero_drop_gate.py` registers under a new hook slot `on_postprocess` (introduced by distillation; useful for any future end-of-training validator). Same layer as the parity-gate conceptually, different signal.

### E.2 When it runs

**v0.3 default:** end of training only, inside `postprocess.finalize()`. Not every checkpoint — LIBERO-10 × N=25 costs ~$3-5 A10G per eval. The lighter `libero_smoketest_hook` (5 tasks × 5 episodes, ~$0.30) runs every N checkpoints for observability per D7, but is NOT the gate.

`--skip-libero-gate` is allowed but loud: VERIFICATION.md reports `libero_drop_gate: SKIPPED` and the deploy claim loses the task-success receipt.

### E.3 What it does

Invokes `scripts/modal_libero_monolithic_onnx.py` twice: once against teacher export, once against student export. Computes per-task success + delta. Writes `<output>/libero_drop_gate.json`.

Teacher rollout is **memoized** per `(teacher_export_sha256, libero_config)` tuple at `~/.cache/reflex/libero_eval/`. Five distillations from the same teacher → one teacher eval. `--no-cache-teacher-eval` escape available.

### E.4 Reporting back

Three surfaces:
1. `FinetuneResult.libero_drop_gate: dict` — new field.
2. `VERIFICATION.md` — new section via `distill_report=` kwarg on `write_verification_report`.
3. Standalone `<output>/libero_drop_gate.json` (machine-readable).

### E.5 v0.3 blocking

**BLOCKING.** Per scope §D Gate 1, `gate_passed=False` returns `FinetuneResult(status="failed_libero_gate", error="...per-task breakdown at libero_drop_gate.json")` with non-zero exit. Override: `--force-ship-distilled` (emits loud VERIFICATION.md banner, exit 0). v0.3.1 SmolVLA: same gate at 10pp threshold (scope §D Gate 4).

---

## Section F — Recommended architecture + rationale

**Pick: extend-the-finetune-registry. Distill lives inside `src/reflex/finetune/` as a second phase, with only SnapFlow math + teacher loader outside it.**

Call path: `reflex distill` → `cli_distill.py` → `FinetuneConfig(phase="distill")` → `run_finetune()` → `preflight.run_preflight()` → `resolve_backend()` → `SnapFlowBackend.fit()` → `postprocess.finalize()`. One linear path, one entry function, no parallel `run_distill()`.

SnapFlow *math* lives in `src/reflex/distill/snapflow.py` (heavy, research-grade, isolated unit tests). The math is invoked by `SnapFlowBackend` inside the finetune package. Every existing fine-tune hook (calibration, parity, wandb) composes for free; every new distill preflight check composes into the same `PreflightReport`.

### F.1 Rejected alternatives

- **Parallel subsystem at `src/reflex/distill/runner.py`.** Rejects scope §Q4's "80% reuse" mandate; duplicates preflight/postprocess/hook/VERIFICATION code; two CLI dispatch paths drift.
- **Unified `BaseTrainer` covering fine-tune + distill.** D5 axiom ("paradigm leak acceptable") makes abstraction premature; `Backend` protocol is the right seam.
- **Subprocess distill runner (mirror today's subprocess-lerobot-train).** SnapFlow isn't upstream — there's no binary to invoke. Loss math is ours to write. Intermediate parity / LIBERO smoketests (D7) require in-process state.

### F.2 Where code lands

- **New files, additive only:** SnapFlow math, teacher loader, SnapFlowBackend, three preflight checks, three hooks, two profiles, cli_distill.
- **Edits, backward-compatible:** `FinetuneConfig` (new fields), `run.py` (delegate to `resolve_backend`), `preflight/runner.py` (conditional checks), `src/reflex/cli.py::distill` (rewrite body), `verification_report.py` (optional kwarg), pi0/pi05 exporters (provenance propagation).
- **New hook slots:** `on_postprocess` is new (distill introduces it). `on_checkpoint` + `on_step` are reused.
- **Tests extend, not replace:** `test_finetune_config.py` gains `phase="distill"` cases; `test_finetune_cli_smoke.py` gains `reflex distill --dry-run`. Any test importing `DMPOTrainer`/`PiFlowTrainer` asserts deprecation or is removed.

---

## Section G — Forward-compat for deferred paradigms

**Consistency Policy (DDPM) — v0.5+ for GR00T.** When scope §D Gate 5 clears, add `src/reflex/distill/consistency.py` (DDPM math) + `src/reflex/finetune/backends/consistency_backend.py`. Widen `distillation_method` Literal to include `"consistency"`. `resolve_backend` gains one `elif`. `teacher_paradigm.py` allowlists `gr00t`. Nothing else moves — Backend protocol, hook registry, export chain are paradigm-agnostic.

**SmolVLA — v0.3.1.** Assuming SnapFlow works on SmolVLA: `teacher_paradigm.py` allowlists `smolvla`; add `profiles/distill_smolvla_snapflow.py`; default `libero_drop_gate_pp=10.0` for SmolVLA (scope §D Gate 4). If SnapFlow fails on SmolVLA's harder velocity field, ship a different method in `src/reflex/distill/pi_flow_2nfe.py` (distinct from the deprecated scaffold) and widen the Literal. Do NOT pre-build this branch.

**`distillation_method` extensibility.** Literal-typed, not enum. Widening to `"consistency"` or `"pi_flow_2nfe"` is a one-line type change + backend registration. Typer rejects unknown values at parse time. When the Literal has 4+ values, migrate to runtime-registered string (like `ActionHeadStrategy` in finetune's target architecture).

**No premature paradigm abstraction (D5).** `BaseDistillMethod` / `FlowMatchingMixin` are NOT shipped in v0.3. Extract a shared interface when the second concrete implementation lands (v0.5+) — rule of three minus one.

---

## Section H — Open product calls (defaults picked)

**Q1. Student shares weights with teacher (LoRA/DoRA) or full-weight copy?**
**A: Full-weight copy.** SnapFlow's self-distillation setup (methods_survey §B.1) uses full weights; `monolithic.py` wants self-contained weights; LoRA adds merge+rank complexity for no paper-backed win.

**Q2. `reflex distill --teacher-export` as exported ONNX dir or HF model id?**
**A: Exported ONNX dir.** Forces customer through the teacher-export gate where parity receipts exist; v0.3 timeline doesn't budget auto-export-teacher tooling. Revisit with `--teacher-from-hf` if early friction.

**Q3. Distillation preserves teacher's policy_preprocessor / policy_postprocessor, or student re-learns?**
**A: Preserve teacher's exactly.** Behavioral parity is the whole point; norm-stats drift is a documented silent killer (evidence §C.1 Tool Hang −9pp). If a customer distills on a different dataset than teacher trained on, revisit — v0.5.

**Q4. Default `student_nfe` for v0.3 pi-family?**
**A: 1.** Scope §E commits to it; SnapFlow targets 1. If Gate 1 fires at 1, rescope v0.3 to 2 before killing the track.

**Q5. `snapflow_alpha` (consistency loss weight) at CLI or fixed in profile?**
**A: Fixed in profile.** Customer-level tuning invites bad outcomes; paper ablations don't span customer space.

**Q6. `libero_drop_gate` against fixed LIBERO-10 or user-supplied task set?**
**A: LIBERO-10 fixed in v0.3.** Matches SnapFlow paper + our shipped harness. `--gate-tasks` expands in v0.5 alongside multi-benchmark support.

**Q7. Pro-license mode — hard-block / warn-bypass / free-trial?**
**A: Hard-block in v0.3.** First feature deserving a hard paywall (scope §B Q5). Requires `stripe-license-gating` (GOALS weight 6) elevated to 9; else fall back to warn+bypass.

**Q8. Auto-publish distilled checkpoints to HF Hub?**
**A: Opt-in flag.** Match finetune. Hub publishing unlocks free-tier consumption ("export community distilled checkpoints"); surface later via `reflex distill publish`, not a training-time flag.

**Q9. Does `reflex export --from-distilled <hub_id>` work in v0.3?**
**A: Yes.** Scope §B Q5 commits to it. `monolithic.py` already accepts an HF model_id; just detect `distill_provenance` and propagate. Trivial edit.

**Q10. Memoize teacher LIBERO-eval on disk?**
**A: Yes,** keyed on SHA256(teacher_ckpt). Cache at `~/.cache/reflex/libero_eval/`. `--no-cache-teacher-eval` escape.

---

## Section I — Kill-gate enforcement in code

| Gate | Source | File | Severity | Override |
|---|---|---|---|---|
| A — Pro-license | scope §B Q5 | `preflight/pro_license.py` | fail-preflight | none in v0.3 |
| B — LIBERO drop (pi-family) | scope §D Gate 1 | `hooks/libero_drop_gate.py` | fail-postprocess | `--force-ship-distilled` (loud VERIFICATION.md banner) |
| C — Cos parity | finetune A2 | `postprocess.finalize` → `validate_roundtrip` | fail-postprocess | `--force-export-anyway` |
| D — SmolVLA drop | scope §D Gate 4 | `hooks/libero_drop_gate.py` (profile overrides threshold) | fail-postprocess | `--force-ship-distilled` + SmolVLA caveat |
| E — Teacher paradigm | scope §D Gate 5 | `preflight/teacher_paradigm.py` | fail-preflight | none (bypass would ship untested paradigm) |
| F — Customer signal (60d post-ship) | scope §D Gate 6 | N/A (product gate) | manual review | N/A — tracked in `reflex_context/02_metrics/distill_adoption.md` |

Each code-enforced gate surfaces as structured `FinetuneResult.error` plus non-zero exit. No gate lives only in a docstring. The `--force-*` overrides propagate to VERIFICATION.md as prominent banners — customers who bypass a gate accept a documented receipt reflecting the risk.

---

## Constraints honored

- **Scope committed.** No DDPM in v0.3. No pi-Flow fallback (deprecated). No DMPO (different product). No SmolVLA in v0.3.
- **80% finetune-substrate reuse.** All distill files except SnapFlow math + teacher loader live inside `src/reflex/finetune/`. Backend protocol, hooks, preflight orchestrator, postprocess, VERIFICATION.md layer are shared.
- **Pro-gating clean.** One preflight check, one override flag, no runtime/export contamination.
- **Deprecations explicit.** `pi_flow.py` + `dmpo.py` archived with redirect errors. `GOALS.yaml::distill-dmpo` renamed. Existing CLI commands rewritten, not patched over.
- **No code written** — only the architecture markdown, per constraints.
