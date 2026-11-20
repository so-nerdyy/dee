"""Phase-7 component A: the discrete-event digital twin of the dee hierarchy.

Entry point (called by ``theory.run_all`` as ``theory.sim.run(anchor_result,
fast=fast)``) runs three modes and writes only ``theory/data/sim_*``,
``theory/figs/sim_*`` and ``theory/sim/SIM_VS_THEORY.md``:

  REPLAY MODE  — the anchor reproduction gate (``theory/sim/replay.py``).  The
                 sealed routed_experts.jsonl stream through the twin, scored
                 against the PRE-REGISTERED ``theory/anchor_gate.py
                 GATE_TOLERANCE``.  A gate, not a guideline: on any registered
                 failure the stage reports the magnitude and STOPS — no
                 predictions are produced from an unanchored sim.

  CROSS-CHECK  — mutual falsification against the closed forms
                 (``theory/sim/crosscheck.py``): the same synthetic streams
                 through (a) the event-driven twin, (b) ``theory.cache`` Che
                 steady + finite-window two-level, (c) ``theory.roofline``
                 T_pred.  Tolerances are declared in ``sim/constants.py``
                 BEFORE the cross-check runs; every disagreement is quantified
                 in SIM_VS_THEORY.md.

  SYNTH MODE   — long synthetic streams (>= 100k requests) from the fitted
                 popularity/temporal models (``theory/sim/synth.py``), testing
                 the steady-state claims a 5k trace cannot reach: the Che
                 finite-window error shrinkage, the LRU knee's steady-state
                 value, and the LRU-vs-MIN gap.
"""
from __future__ import annotations

from typing import Any, Dict

from ..util import dlog


def run(anchor_result, fast: bool = False) -> Dict[str, Any]:
    """Run all three sim modes.  Deterministic (fixed seeds throughout)."""
    from . import crosscheck, replay, synth

    out: Dict[str, Any] = {}

    # ---------------- REPLAY (the gate) ---------------------------------
    out["replay"] = replay.run(anchor_result, fast=fast)
    out["gate_status"] = out["replay"]["gate_status"]
    if out["gate_status"] != "PASS":
        # HARD RULE (AGENT_BRIEF_PHASE7.md L72-75): the anchor gate is a gate,
        # not a guideline.  Do NOT generate predictions from an unanchored sim.
        dlog("SIM: ANCHOR GATE FAILED — reporting and stopping. "
             "No cross-check / synth output is produced from an unanchored sim.")
        _write_report(out, gated_out=True)
        return out

    # ---------------- CROSS-CHECK (mutual falsification) -----------------
    out["crosscheck"] = crosscheck.run(fast=fast)

    # ---------------- SYNTH (long stationary streams) --------------------
    out["synth"] = synth.run(fast=fast)

    _write_report(out, gated_out=False)
    dlog("SIM: gate PASS; cross-check pass rate "
         "%.1f%%; synth streams written" %
         (100.0 * out["crosscheck"]["pass_rate"]))
    return out


def _write_report(out: Dict[str, Any], gated_out: bool) -> None:
    from .report import write_sim_vs_theory

    write_sim_vs_theory(out, gated_out=gated_out)
