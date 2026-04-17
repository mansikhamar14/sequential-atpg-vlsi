"""
9-Valued Logic ATPG Engine (Muth's Algorithm) — Mode B
========================================================
Implements sequential ATPG using 9-valued logic to overcome the
initialization limitations of the Extended D-Algorithm.

9-valued logic represents each wire as a (good/faulty) pair:
  0/0, 0/1(D̄), 1/0(D), 1/1, 0/X, X/0, 1/X, X/1, X/X

Key advantage over 5-valued logic:
  Partial knowledge states (e.g., 0/X, X/1) allow the algorithm to
  make progress even when flip-flop initial states are unknown.
  The 5-valued system collapses these into X, losing information.

Algorithm:
  1. Initialization — all wires X/X
  2. Fault Activation — inject D (1/0) or D̄ (0/1) on fault wire
  3. Fault Propagation — 9-valued forward implication + D-frontier
  4. Backward Justification — PODEM-style backtrace
  5. Backtracking with configurable limit
"""

import copy
import time
import logging
from typing import Optional, List, Dict, Tuple, Set

from logic9val import (
    evaluate_gate_9v, fault_value_9v, is_discrepant,
    is_unknown, good_val, faulty_val, ctrl_val_9v, inv_flag_9v,
    decode, encode, merge_9val,
)
from parser import Circuit, Gate, topological_order
from timeframe import UnrolledCircuit, unroll

logger = logging.getLogger("9Val")


# ── Result (shared format with ExtD) ─────────────────────────────────────────

class ATPGResult:
    DETECTED     = "DETECTED"
    UNDETECTABLE = "UNDETECTABLE"
    ABORTED      = "ABORTED"

    def __init__(self, status, fault_wire, fault_type,
                 test_vector=None, backtracks=0, time_s=0.0, detail=""):
        self.status      = status
        self.fault_wire  = fault_wire
        self.fault_type  = fault_type
        self.test_vector = test_vector
        self.backtracks  = backtracks
        self.time_s      = time_s
        self.detail      = detail

    def __repr__(self):
        tv = "" if self.test_vector is None else f", tv_len={len(self.test_vector)}"
        d = f", {self.detail}" if self.detail else ""
        return (f"ATPGResult({self.status}, {self.fault_wire}/{self.fault_type}"
                f", bt={self.backtracks}{tv}, t={self.time_s:.4f}s{d})")


# ── 9-Valued Logic ATPG Engine ───────────────────────────────────────────────

class NineValueAlgorithm:
    """
    ATPG engine using Muth's 9-valued logic.
    Operates on a time-frame expanded (unrolled) circuit.
    """

    def __init__(self, circuit: Circuit, num_frames: int = 3,
                 backtrack_limit: int = 50):
        self.circuit = circuit
        self.num_frames = num_frames
        self.backtrack_limit = backtrack_limit
        self._topo = topological_order(circuit)
        self._frames = list(range(-(num_frames - 1), 1))
        self._trace_recorder = None  # opt-in hook, set externally

    def _trace(self, label, state, frontier_keys=None, objective_wire=None):
        if self._trace_recorder is not None:
            self._trace_recorder.record(label, state, frontier_keys, objective_wire)

    def generate_test(self, fault_wire: str, fault_type: str) -> ATPGResult:
        """Attempt to generate a test vector for the given stuck-at fault."""
        t0 = time.time()

        uc = unroll(self.circuit, self.num_frames)

        # Permanent fault model: fault exists in ALL time frames
        fval = fault_value_9v(fault_type)
        fault_wire_keys = []
        for t in self._frames:
            fwk = uc.wire_key(fault_wire, t)
            if fwk in uc.wires:
                fault_wire_keys.append(fwk)

        if not fault_wire_keys:
            return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                              time_s=time.time() - t0,
                              detail="fault wire not in unrolled circuit")

        state = {wk: "X/X" for wk in uc.wires}
        backtracks = [0]

        # ── Step 1: Install permanent stuck-at on faulty side only ────
        # The good-circuit value must be justified through the driver —
        # we do NOT pre-set it to the activation polarity. The faulty
        # side is fixed at the stuck value in every frame (permanent
        # fault model); the good side starts as X and is resolved by
        # forward implication + search.
        f_stuck = faulty_val(fval)
        for fwk in fault_wire_keys:
            state[fwk] = encode("X", f_stuck)
        self._trace("Fault activation in all frames", state)

        # ── Step 2: Forward imply ────────────────────────────────────────
        self._imply(state, uc, fault_wire_keys, fval)
        self._trace("After initial forward implication", state,
                    frontier_keys=self._d_frontier(state, uc))

        if state.get("__conflict__"):
            return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                              time_s=time.time() - t0, detail="initial conflict")

        if self._fault_at_po(state, uc):
            tv = self._extract_tv(state, uc)
            if self._verify_tv(tv, fault_wire, fault_type):
                return ATPGResult(ATPGResult.DETECTED, fault_wire, fault_type,
                                  test_vector=tv, time_s=time.time() - t0)

        # ── Step 3: Search (PODEM-style) ─────────────────────────────────
        success = self._search(state, uc, fault_wire_keys, fval, backtracks)
        elapsed = time.time() - t0

        if backtracks[0] > self.backtrack_limit:
            return ATPGResult(ATPGResult.ABORTED, fault_wire, fault_type,
                              backtracks=backtracks[0], time_s=elapsed)

        if success:
            tv = self._extract_tv(state, uc)
            if self._verify_tv(tv, fault_wire, fault_type):
                return ATPGResult(ATPGResult.DETECTED, fault_wire, fault_type,
                                  test_vector=tv, backtracks=backtracks[0],
                                  time_s=elapsed)
            # Internal D-path claimed detection but fault simulation
            # disagrees — unjustified PI/PPI values on the support of the
            # D-path likely broke under default concretization.
            return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                              backtracks=backtracks[0], time_s=elapsed,
                              detail="verification failed")

        return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                          backtracks=backtracks[0], time_s=elapsed)

    def _verify_tv(self, tv, fault_wire, fault_type):
        """Fault-simulate the extracted TV to confirm the fault is really
        detected. Prevents false positives caused by unjustified PI/PPI
        values getting concretized to arbitrary defaults."""
        from fault_sim import FaultSimulator
        sim = FaultSimulator(self.circuit)
        return (fault_wire, fault_type) in sim.simulate([tv], [(fault_wire, fault_type)])

    # ── Recursive Search ─────────────────────────────────────────────────

    def _search(self, state, uc, fault_wks, fval, backtracks):
        if backtracks[0] > self.backtrack_limit:
            return False

        if self._fault_at_po(state, uc):
            return True

        objective = self._get_objective(state, uc, fault_wks, fval)
        if objective is None:
            return False

        obj_wire, obj_val = objective

        pi_result = self._backtrace(state, uc, obj_wire, obj_val)
        if pi_result is None:
            return False

        pi_wire, pi_val = pi_result

        # Try the suggested value and its complement. When the primary
        # value has an unknown good side, try both concrete polarities.
        g = good_val(pi_val)
        if g == "X":
            pi_val = encode(0, 0)
            alt_val = encode(1, 1)
        else:
            alt_val = encode(1 - g, 1 - g)

        fault_wk_set = set(fault_wks)
        f_stuck = faulty_val(fval)

        for try_val in [pi_val, alt_val]:
            saved = state.copy()

            # When the PI being assigned is itself a fault wire, the faulty
            # side must stay pinned to the stuck value — preserve partial
            # activation rather than forcing full equality.
            if pi_wire in fault_wk_set:
                g = good_val(try_val)
                state[pi_wire] = encode(g, f_stuck)
            else:
                state[pi_wire] = try_val
            self._trace(f"Assign {pi_wire}={try_val}", state,
                        objective_wire=obj_wire)
            state.pop("__conflict__", None)
            self._imply(state, uc, fault_wks, fval)
            self._trace(f"After imply (assigned {pi_wire}={try_val})", state,
                        frontier_keys=self._d_frontier(state, uc))

            if not state.get("__conflict__"):
                if self._fault_at_po(state, uc):
                    return True
                if self._d_frontier(state, uc) or \
                   self._needs_activation(state, fault_wks):
                    if self._search(state, uc, fault_wks, fval, backtracks):
                        return True

            state.clear()
            state.update(saved)
            backtracks[0] += 1

        return False

    def _needs_activation(self, state, fault_wks):
        """True if no fault wire has become discrepant yet."""
        return not any(is_discrepant(state.get(fwk, "X/X")) for fwk in fault_wks)

    # ── Objective ────────────────────────────────────────────────────────

    def _get_objective(self, state, uc, fault_wks, fval):
        good_needed = good_val(fval)
        # Only generate an activation objective if NO frame is yet activated.
        if self._needs_activation(state, fault_wks):
            best = None
            for fwk in fault_wks:
                g = good_val(state.get(fwk, "X/X"))
                if g == "X":
                    best = fwk
                    break
            if best is None:
                # Every frame's good side is already pinned to the stuck
                # value — no frame can activate with current commitments.
                return None
            driver = uc.fanin.get(best)
            if driver is None:
                return (best, good_needed)
            if driver[0] == "DFF_CONNECT":
                return (driver[1], good_needed)
            return (best, good_needed)

        # All fault wires activated — propagate via D-frontier
        frontier = self._d_frontier(state, uc)
        if not frontier:
            return None

        gate_key = frontier[0]

        # Skip DFF frontier entries — implication handles DFF propagation
        if isinstance(gate_key, tuple) and gate_key[0] == "__DFF__":
            # Find next combinational gate in frontier
            for gk in frontier[1:]:
                if not (isinstance(gk, tuple) and gk[0] == "__DFF__"):
                    gate_key = gk
                    break
            else:
                return None  # Only DFF frontier entries remain

        gate = uc.gates[gate_key]
        gtype = gate.gate_type
        cv = ctrl_val_9v(gtype)

        if cv is None:
            return None

        nc = 1 - cv
        # Only pick an input whose known good side doesn't already clash with
        # the non-controlling value we need (e.g. "0/X" on an AND input when
        # nc=1 would be a dead-end — good=0 is already controlling).
        for iw in gate.inputs:
            iv = state.get(iw, "X/X")
            if is_discrepant(iv):
                continue
            g = good_val(iv)
            if g == "X":
                return (iw, nc)
            if g == nc:
                # already non-controlling, no objective needed for this input
                continue
            # g is the controlling value — this input can't be justified
            # to nc without conflict; skip it.
        return None

    # ── Backtrace ────────────────────────────────────────────────────────

    def _backtrace(self, state, uc, wire_key, value_needed):
        current_wire = wire_key
        current_val = value_needed if isinstance(value_needed, int) else value_needed
        visited = set()

        while len(visited) < 100:
            if current_wire in visited:
                break
            visited.add(current_wire)

            cv = state.get(current_wire, "X/X")

            if current_wire in uc.primary_inputs or current_wire in uc.pseudo_primary_inputs:
                # A PI is assignable if its good side is still unresolved,
                # including the fault-wire case where state is "X/<stuck>".
                if good_val(cv) == "X":
                    if isinstance(current_val, int):
                        return (current_wire, encode(current_val, current_val))
                    return (current_wire, encode(0, 0))
                break

            if cv != "X/X" and not is_unknown(cv):
                break

            driver = uc.fanin.get(current_wire)
            if driver is None:
                if cv == "X/X":
                    if isinstance(current_val, int):
                        return (current_wire, encode(current_val, current_val))
                    return (current_wire, encode(0, 0))
                break

            if driver[0] == "DFF_CONNECT":
                current_wire = driver[1]
                continue

            frame, gname = driver
            gk = (frame, gname)
            if gk not in uc.gates:
                break
            gate = uc.gates[gk]
            gtype = gate.gate_type

            if gtype in ('not', 'buf'):
                if gtype == 'not' and isinstance(current_val, int):
                    current_val = 1 - current_val
                current_wire = gate.inputs[0]
                continue

            cv_gate = ctrl_val_9v(gtype)
            is_inv = inv_flag_9v(gtype)

            if isinstance(current_val, int):
                pre_inv = (1 - current_val) if is_inv else current_val
            else:
                pre_inv = current_val

            found = False
            for iw in gate.inputs:
                iv = state.get(iw, "X/X")
                if iv == "X/X" or is_unknown(iv):
                    if isinstance(pre_inv, int) and cv_gate is not None:
                        current_val = cv_gate if pre_inv == cv_gate else (1 - cv_gate)
                    current_wire = iw
                    found = True
                    break

            if not found:
                break

        # Backtrace fell off the path without reaching an assignable PI.
        # Rather than picking an arbitrary PI with a meaningless polarity
        # (which produces random assignments and inflates backtracks),
        # signal failure so _search can try a different objective.
        return None

    # ── Forward Implication ──────────────────────────────────────────────

    def _imply(self, state, uc, fault_wks, fval):
        fault_wk_set = set(fault_wks)
        changed = True
        max_iters = 30
        iters = 0

        while changed and iters < max_iters:
            changed = False
            state.pop("__conflict__", None)

            # Resolve DFF connections
            f_stuck = faulty_val(fval)
            for ff_gname in self.circuit.flip_flops:
                ff_gate = self.circuit.gates[ff_gname]
                d_wire = ff_gate.inputs[0]
                q_wire = ff_gate.output
                for idx, t in enumerate(self._frames[1:], start=1):
                    prev_t = self._frames[idx - 1]
                    d_wk = uc.wire_key(d_wire, prev_t)
                    q_wk = uc.wire_key(q_wire, t)
                    d_val = state.get(d_wk, "X/X")
                    q_val = state.get(q_wk, "X/X")
                    if d_val == "X/X" or d_val == q_val:
                        continue
                    if q_wk in fault_wk_set:
                        # Fault wire Q: inherit good side from D, keep
                        # faulty side pinned to the stuck value.
                        g_new = good_val(d_val)
                        candidate = encode(g_new, f_stuck)
                    else:
                        candidate = d_val
                    merged = merge_9val(q_val, candidate)
                    if merged is None:
                        state["__conflict__"] = True
                        return
                    if merged != q_val:
                        state[q_wk] = merged
                        changed = True

            # Evaluate combinational gates
            for t in self._frames:
                for gname in self._topo:
                    gate_key = (t, gname)
                    if gate_key not in uc.gates:
                        continue
                    gate = uc.gates[gate_key]
                    in_vals = [state.get(iw, "X/X") for iw in gate.inputs]

                    if all(v == "X/X" for v in in_vals):
                        continue

                    new_val = evaluate_gate_9v(gate.gate_type, in_vals)
                    if new_val == "X/X":
                        continue

                    out_wire = gate.output

                    if out_wire in fault_wk_set:
                        # Update the good-circuit value while preserving the
                        # faulty value (permanent stuck-at). Use merge so that
                        # any previously-implied good value conflicts are
                        # caught rather than silently overwritten.
                        old = state.get(out_wire, "X/X")
                        g_new = good_val(new_val)
                        f_old = faulty_val(fval)
                        if g_new != "X":
                            candidate = encode(g_new, f_old)
                            merged = merge_9val(old, candidate)
                            if merged is None:
                                state["__conflict__"] = True
                                return
                            if merged != old:
                                state[out_wire] = merged
                                changed = True
                        continue

                    old_val = state.get(out_wire, "X/X")
                    if new_val != old_val:
                        merged = merge_9val(old_val, new_val)
                        if merged is None:
                            state["__conflict__"] = True
                            return
                        if merged != old_val:
                            state[out_wire] = merged
                            changed = True

            # Re-enforce the stuck value on the faulty side in ALL frames.
            # DO NOT force the good side — it must be derived from the driver.
            for fwk in fault_wks:
                cur = state.get(fwk, "X/X")
                g_cur = good_val(cur)
                f_fault = faulty_val(fval)
                state[fwk] = encode(g_cur, f_fault)
            iters += 1

    # ── D-Frontier ───────────────────────────────────────────────────────

    def _d_frontier(self, state, uc):
        frontier = []
        for gate_key, gate in uc.gates.items():
            if gate.gate_type == "dff":
                continue
            out_val = state.get(gate.output, "X/X")
            if is_discrepant(out_val):
                continue
            if out_val != "X/X" and not is_unknown(out_val):
                continue
            in_vals = [state.get(iw, "X/X") for iw in gate.inputs]
            if any(is_discrepant(v) for v in in_vals):
                frontier.append(gate_key)

        # Also check DFF boundaries for fault propagation across frames
        for ff_gname in self.circuit.flip_flops:
            ff_gate = self.circuit.gates[ff_gname]
            d_wire = ff_gate.inputs[0]
            q_wire = ff_gate.output
            for idx, t in enumerate(self._frames[1:], start=1):
                prev_t = self._frames[idx - 1]
                d_wk = uc.wire_key(d_wire, prev_t)
                q_wk = uc.wire_key(q_wire, t)
                d_val = state.get(d_wk, "X/X")
                q_val = state.get(q_wk, "X/X")
                if is_discrepant(d_val) and not is_discrepant(q_val):
                    # DFF boundary needs propagation — handled by implication
                    # but signals that frontier still exists
                    frontier.append(("__DFF__", ff_gname))
                    break

        # Prioritize gates closer to POs (later in topo order = closer to output)
        topo_index = {g: i for i, g in enumerate(self._topo)}
        frontier.sort(key=lambda gk: topo_index.get(gk[1], 0) if isinstance(gk[0], int) else -1, reverse=True)
        return frontier

    # ── Fault at PO ──────────────────────────────────────────────────────

    def _fault_at_po(self, state, uc):
        for po_wk in uc.primary_outputs:
            if is_discrepant(state.get(po_wk, "X/X")):
                return True
        return False

    # ── Test Vector Extraction ───────────────────────────────────────────

    def _extract_tv(self, state, uc):
        frames = list(range(-(uc.num_frames - 1), 1))
        earliest = frames[0]
        tv = []
        for t in frames:
            frame_vals = {}
            for pi in self.circuit.primary_inputs:
                wk = uc.wire_key(pi, t)
                val_9 = state.get(wk, "X/X")
                g = good_val(val_9)
                frame_vals[pi] = str(g) if g != "X" else "0"
            # Include FF initial states (PPIs) in the earliest frame
            if t == earliest:
                for ff_gname in self.circuit.flip_flops:
                    q_wire = self.circuit.gates[ff_gname].output
                    wk = uc.wire_key(q_wire, t)
                    val_9 = state.get(wk, "X/X")
                    g = good_val(val_9)
                    frame_vals[q_wire] = str(g) if g != "X" else "0"
            tv.append(frame_vals)
        return tv


# ── Fault List Generator ─────────────────────────────────────────────────────

def generate_fault_list(circuit: Circuit) -> List[Tuple[str, str]]:
    faults = []
    for wire in circuit.wires:
        faults.append((wire, "SA0"))
        faults.append((wire, "SA1"))
    return faults


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from parser import parse_verilog

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"
    nf   = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    circ = parse_verilog(path)
    faults = generate_fault_list(circ)
    engine = NineValueAlgorithm(circ, num_frames=nf)

    det = undet = abort = 0
    for fw, ft in faults:
        r = engine.generate_test(fw, ft)
        print(f"  {r}")
        if r.status == ATPGResult.DETECTED: det += 1
        elif r.status == ATPGResult.UNDETECTABLE: undet += 1
        else: abort += 1

    total = len(faults)
    print(f"\nDetected={det}  Undetectable={undet}  Aborted={abort}")
    print(f"Coverage={100.0*det/total:.1f}%")
