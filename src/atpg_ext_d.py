"""
Extended D-Algorithm (5-Valued Logic) — Mode A ATPG Engine
============================================================
Implements the Extended D-Algorithm for sequential ATPG using 5-valued
logic: {0, 1, X, D, D_BAR}.

Algorithm steps:
  1. Initialization — set all wires to X
  2. Fault Activation — inject D / D_BAR on fault wire
  3. Fault Propagation — D-drive through D-frontier to POs
  4. Backward Justification — backtrace unsatisfied objectives to PIs
  5. Backtracking — undo on conflicts, try alternatives

Known Limitation:
  The Extended D-Algorithm with 5-valued logic CANNOT represent partial
  knowledge of flip-flop initial states in sequential circuits. This
  frequently causes initialization failures — the algorithm treats all
  unknown FF states as X, which makes it impossible to distinguish
  between scenarios the 9-valued algorithm handles gracefully.

  When initialization fails, the fault is logged as INIT_FAILURE.
"""

import copy
import time
import logging
from typing import Optional, List, Dict, Tuple, Set

from logic5val import (
    ZERO, ONE, X, D, D_BAR,
    evaluate_gate_5v, is_d_value, complement_5v,
    fault_activation_value_5v, ctrl_val_5v, non_ctrl_val_5v,
    is_inverting_5v,
)
from parser import Circuit, Gate, topological_order
from timeframe import UnrolledCircuit, unroll

logger = logging.getLogger("ExtD")


# ── Result ───────────────────────────────────────────────────────────────────

class ATPGResult:
    DETECTED     = "DETECTED"
    UNDETECTABLE = "UNDETECTABLE"
    ABORTED      = "ABORTED"
    INIT_FAILURE = "INIT_FAILURE"

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


# ── Extended D-Algorithm Engine ──────────────────────────────────────────────

class ExtendedDAlgorithm:
    """
    ATPG engine using 5-valued logic (0, 1, X, D, D_BAR).
    Operates on a time-frame expanded (unrolled) circuit.
    """

    def __init__(self, circuit: Circuit, num_frames: int = 3,
                 backtrack_limit: int = 50):
        self.circuit = circuit
        self.num_frames = num_frames
        self.backtrack_limit = backtrack_limit
        self._topo = topological_order(circuit)
        self._frames = list(range(-(num_frames - 1), 1))

    def generate_test(self, fault_wire: str, fault_type: str) -> ATPGResult:
        """Attempt to generate a test vector for the given stuck-at fault."""
        t0 = time.time()

        uc = unroll(self.circuit, self.num_frames)

        # Permanent fault model: fault exists in ALL time frames
        activation_val = fault_activation_value_5v(fault_type)
        fault_wire_keys = []
        for t in self._frames:
            fwk = uc.wire_key(fault_wire, t)
            if fwk in uc.wires:
                fault_wire_keys.append(fwk)

        if not fault_wire_keys:
            return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                              time_s=time.time() - t0,
                              detail="fault wire not in unrolled circuit")

        state = {wk: X for wk in uc.wires}
        backtracks = [0]

        # ── Step 1: Fault Activation in ALL frames ───────────────────
        for fwk in fault_wire_keys:
            state[fwk] = activation_val

        # ── Step 2 & 3: Forward imply, then search ──────────────────
        self._imply_forward(state, uc, fault_wire_keys, activation_val)

        if self._fault_at_po(state, uc):
            tv = self._extract_tv(state, uc)
            return ATPGResult(ATPGResult.DETECTED, fault_wire, fault_type,
                              test_vector=tv, time_s=time.time() - t0)

        success = self._search(state, uc, fault_wire_keys, activation_val, backtracks)
        elapsed = time.time() - t0

        if backtracks[0] > self.backtrack_limit:
            return ATPGResult(ATPGResult.ABORTED, fault_wire, fault_type,
                              backtracks=backtracks[0], time_s=elapsed)

        if success:
            tv = self._extract_tv(state, uc)
            return ATPGResult(ATPGResult.DETECTED, fault_wire, fault_type,
                              test_vector=tv, backtracks=backtracks[0],
                              time_s=elapsed)

        return ATPGResult(ATPGResult.UNDETECTABLE, fault_wire, fault_type,
                          backtracks=backtracks[0], time_s=elapsed)

    # ── Recursive Search ─────────────────────────────────────────────────

    def _search(self, state, uc, fault_wks, act_val, backtracks):
        if backtracks[0] > self.backtrack_limit:
            return False

        if self._fault_at_po(state, uc):
            return True

        objective = self._get_objective(state, uc, fault_wks, act_val)
        if objective is None:
            return False

        obj_wire, obj_val = objective

        # Backtrace to a PI
        pi_result = self._backtrace(state, uc, obj_wire, obj_val)
        if pi_result is None:
            return False

        pi_wire, pi_val = pi_result

        # Try both values
        for try_val in [pi_val, complement_5v(pi_val)]:
            if try_val == X:
                continue
            saved = state.copy()

            state[pi_wire] = try_val
            self._imply_forward(state, uc, fault_wks, act_val)

            if not self._has_conflict(state, uc):
                if self._fault_at_po(state, uc):
                    return True
                if self._d_frontier(state, uc, fault_wks):
                    if self._search(state, uc, fault_wks, act_val, backtracks):
                        return True

            state.clear()
            state.update(saved)
            backtracks[0] += 1

        return False

    # ── Objective determination ──────────────────────────────────────────

    def _get_objective(self, state, uc, fault_wks, act_val):
        # Check if any fault wire still needs activation
        good_val_needed = ONE if act_val == D else ZERO
        for fwk in fault_wks:
            fval = state.get(fwk, X)
            if not is_d_value(fval):
                driver_info = uc.fanin.get(fwk)
                if driver_info is None:
                    return (fwk, good_val_needed)
                if driver_info[0] == "DFF_CONNECT":
                    return (driver_info[1], good_val_needed)
                return (fwk, good_val_needed)

        # All fault wires activated — propagate via D-frontier
        frontier = self._d_frontier(state, uc, fault_wks)
        if not frontier:
            return None

        gate_key = frontier[0]
        if gate_key == "__DFF__":
            # DFF frontier element — need to justify the D-input value
            # Already handled by implication, just need more PIs set
            return None

        gate = uc.gates[gate_key]
        gtype = gate.gate_type
        ncv = non_ctrl_val_5v(gtype)

        if ncv is None:
            return None

        for iw in gate.inputs:
            iv = state.get(iw, X)
            if not is_d_value(iv) and iv == X:
                return (iw, ncv)

        return None

    # ── Backtrace ────────────────────────────────────────────────────────

    def _backtrace(self, state, uc, wire_key, value_needed):
        current_wire = wire_key
        current_val = value_needed
        visited = set()

        while len(visited) < 100:
            if current_wire in visited:
                break
            visited.add(current_wire)

            cv = state.get(current_wire, X)

            # Assignable?
            if current_wire in uc.primary_inputs or current_wire in uc.pseudo_primary_inputs:
                if cv == X:
                    return (current_wire, current_val)
                break

            if cv != X:
                break

            driver = uc.fanin.get(current_wire)
            if driver is None:
                if cv == X:
                    return (current_wire, current_val)
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
                if gtype == 'not':
                    current_val = complement_5v(current_val)
                current_wire = gate.inputs[0]
                continue

            # For AND/OR etc., trace through an unset input
            is_inv = is_inverting_5v(gtype)
            cv_gate = ctrl_val_5v(gtype)

            if is_inv:
                pre_inv = complement_5v(current_val)
            else:
                pre_inv = current_val

            found = False
            for iw in gate.inputs:
                iv = state.get(iw, X)
                if iv == X:
                    if cv_gate is not None:
                        if pre_inv == cv_gate:
                            current_val = cv_gate
                        else:
                            ncv = non_ctrl_val_5v(gtype)
                            current_val = ncv if ncv is not None else current_val
                    current_wire = iw
                    found = True
                    break

            if not found:
                break

        # Fallback: find any unset PI
        for pi in uc.primary_inputs + uc.pseudo_primary_inputs:
            if state.get(pi, X) == X:
                return (pi, current_val)

        return None

    # ── Forward Implication ──────────────────────────────────────────────

    def _imply_forward(self, state, uc, fault_wks, act_val):
        fault_wk_set = set(fault_wks)
        changed = True
        max_iters = 30
        iters = 0

        while changed and iters < max_iters:
            changed = False

            # Resolve DFF connections — propagate D/D_BAR through DFFs
            for ff_gname in self.circuit.flip_flops:
                ff_gate = self.circuit.gates[ff_gname]
                d_wire = ff_gate.inputs[0]
                q_wire = ff_gate.output
                for idx, t in enumerate(self._frames[1:], start=1):
                    prev_t = self._frames[idx - 1]
                    d_wk = uc.wire_key(d_wire, prev_t)
                    q_wk = uc.wire_key(q_wire, t)
                    d_val = state.get(d_wk, X)
                    q_val = state.get(q_wk, X)
                    if d_val != X and q_val == X:
                        # Don't overwrite fault sites
                        if q_wk not in fault_wk_set:
                            state[q_wk] = d_val
                            changed = True

            # Evaluate combinational gates
            for t in self._frames:
                for gname in self._topo:
                    gate_key = (t, gname)
                    if gate_key not in uc.gates:
                        continue
                    gate = uc.gates[gate_key]
                    in_vals = [state.get(iw, X) for iw in gate.inputs]

                    if all(v == X for v in in_vals):
                        continue

                    new_val = evaluate_gate_5v(gate.gate_type, in_vals)
                    out_wire = gate.output

                    # Don't overwrite fault sites
                    if out_wire in fault_wk_set:
                        continue

                    old_val = state.get(out_wire, X)
                    if new_val != X and new_val != old_val:
                        if old_val == X:
                            state[out_wire] = new_val
                            changed = True

            # Re-enforce fault in ALL frames
            for fwk in fault_wks:
                state[fwk] = act_val
            iters += 1

    # ── D-Frontier ───────────────────────────────────────────────────────

    def _d_frontier(self, state, uc, fault_wks=None):
        frontier = []
        for gate_key, gate in uc.gates.items():
            if gate.gate_type == "dff":
                continue
            out_val = state.get(gate.output, X)
            if is_d_value(out_val):
                continue  # Already propagated
            if out_val != X:
                continue
            in_vals = [state.get(iw, X) for iw in gate.inputs]
            if any(is_d_value(v) for v in in_vals):
                frontier.append(gate_key)

        # Also check DFF boundaries: D-value at DFF D-input that hasn't
        # propagated to Q-output in next frame
        for ff_gname in self.circuit.flip_flops:
            ff_gate = self.circuit.gates[ff_gname]
            d_wire = ff_gate.inputs[0]
            q_wire = ff_gate.output
            for idx, t in enumerate(self._frames[1:], start=1):
                prev_t = self._frames[idx - 1]
                d_wk = uc.wire_key(d_wire, prev_t)
                q_wk = uc.wire_key(q_wire, t)
                d_val = state.get(d_wk, X)
                q_val = state.get(q_wk, X)
                if is_d_value(d_val) and not is_d_value(q_val):
                    # DFF boundary is a propagation opportunity
                    frontier.append("__DFF__")
                    break

        # Prioritize gates closer to POs (later in topo order)
        topo_index = {g: i for i, g in enumerate(self._topo)}
        frontier.sort(key=lambda gk: topo_index.get(gk[1], 0) if gk != "__DFF__" else -1, reverse=True)
        return frontier

    # ── Fault at PO ──────────────────────────────────────────────────────

    def _fault_at_po(self, state, uc):
        for po_wk in uc.primary_outputs:
            if is_d_value(state.get(po_wk, X)):
                return True
        return False

    # ── Conflict detection ───────────────────────────────────────────────

    def _has_conflict(self, state, uc):
        """Check for obvious conflicts in the current state."""
        for gate_key, gate in uc.gates.items():
            if gate.gate_type == "dff":
                continue
            out_val = state.get(gate.output, X)
            if out_val == X:
                continue
            in_vals = [state.get(iw, X) for iw in gate.inputs]
            if all(v == X for v in in_vals):
                continue
            expected = evaluate_gate_5v(gate.gate_type, in_vals)
            if expected != X and out_val != X and expected != out_val:
                # D/D_BAR vs computed value is OK at fault sites
                if is_d_value(out_val) or is_d_value(expected):
                    continue
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
                val = state.get(wk, X)
                if val == X:
                    frame_vals[pi] = "0"
                elif val == D:
                    frame_vals[pi] = "1"
                elif val == D_BAR:
                    frame_vals[pi] = "0"
                else:
                    frame_vals[pi] = str(val)
            # Include FF initial states (PPIs) in the earliest frame
            if t == earliest:
                for ff_gname in self.circuit.flip_flops:
                    q_wire = self.circuit.gates[ff_gname].output
                    wk = uc.wire_key(q_wire, t)
                    val = state.get(wk, X)
                    if val == X:
                        frame_vals[q_wire] = "0"
                    elif val == D:
                        frame_vals[q_wire] = "1"
                    elif val == D_BAR:
                        frame_vals[q_wire] = "0"
                    else:
                        frame_vals[q_wire] = str(val)
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

    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"
    nf   = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    circ = parse_verilog(path)
    faults = generate_fault_list(circ)
    engine = ExtendedDAlgorithm(circ, num_frames=nf)

    det = init_fail = undet = abort = 0
    for fw, ft in faults:
        r = engine.generate_test(fw, ft)
        print(f"  {r}")
        if r.status == ATPGResult.DETECTED: det += 1
        elif r.status == ATPGResult.INIT_FAILURE: init_fail += 1
        elif r.status == ATPGResult.UNDETECTABLE: undet += 1
        else: abort += 1

    total = len(faults)
    print(f"\nDetected={det}  InitFail={init_fail}  Undetectable={undet}  Aborted={abort}")
    print(f"Coverage={100.0*det/total:.1f}%")
