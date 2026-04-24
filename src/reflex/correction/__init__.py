"""A2C2 — Asynchronous Action Chunk Correction.

Plug-in residual correction head atop any chunk-predicting VLA. Frozen base policy;
small MLP (~100 KB) emits per-chunk-position residuals conditioned on observation +
estimated inference delay. Composes with RTC (boundary smoothing) — A2C2 corrects
*within* a chunk; RTC smooths *between* chunks.

Paper: arxiv 2509.23224 — Sendai, Alvarez, Matsushima, Matsuo, Iwasawa.

Phase B.4 (this module): transfer-validation gate harness. Train on offline LeRobot
LIBERO traces; eval on synthetic-latency-injected serve traces; emit gate report
(MSE ratio + task-success delta vs A2C2-off). Acceptance: MSE ratio <= 1.2x AND
task-success delta >= +5 pp. Hard-abort if ratio > 2.0x OR delta < 0 pp.

Phase B.5 (deferred to next ship): wire trained head into /act handler.
"""

from reflex.correction.a2c2_head import (
    A2C2Config,
    A2C2Head,
    build_a2c2_input,
    chunk_pos_encoding,
    latency_log_encoding,
)
from reflex.correction.transfer_gate import (
    GateDecision,
    GateReport,
    GateThresholds,
    compute_gate_report,
)

__all__ = [
    "A2C2Config",
    "A2C2Head",
    "build_a2c2_input",
    "chunk_pos_encoding",
    "latency_log_encoding",
    "GateDecision",
    "GateReport",
    "GateThresholds",
    "compute_gate_report",
]
