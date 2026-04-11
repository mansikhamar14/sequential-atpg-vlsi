"""
Fault Simulation — Verify ATPG Test Vectors
============================================
Given test vectors and a fault list, simulates the circuit under each
fault to confirm detection.

Supports:
  - Good-circuit simulation (reference)
  - Single stuck-at fault simulation
  - Multi-frame sequential simulation
  - Detection reporting
"""

from typing import List, Dict, Set, Tuple
from parser import Circuit, topological_order


# ── 3-valued simulation helpers ──────────────────────────────────────────────

def _and3(a, b):
    if a == 0 or b == 0: return 0
    if a == 1 and b == 1: return 1
    return "X"

def _or3(a, b):
    if a == 1 or b == 1: return 1
    if a == 0 and b == 0: return 0
    return "X"

def _not3(a):
    return {0: 1, 1: 0, "X": "X"}[a]

def _xor3(a, b):
    if a == "X" or b == "X": return "X"
    return a ^ b

def _reduce(fn, lst, default):
    if not lst: return default
    acc = lst[0]
    for x in lst[1:]:
        acc = fn(acc, x)
    return acc

GATE_OP = {
    "and":  lambda ins: _reduce(_and3, ins, 1),
    "or":   lambda ins: _reduce(_or3, ins, 0),
    "nand": lambda ins: _not3(_reduce(_and3, ins, 1)),
    "nor":  lambda ins: _not3(_reduce(_or3, ins, 0)),
    "not":  lambda ins: _not3(ins[0]),
    "buf":  lambda ins: ins[0],
    "xor":  lambda ins: _reduce(_xor3, ins, 0),
    "xnor": lambda ins: _not3(_reduce(_xor3, ins, 0)),
}

def _eval_gate(gate_type, in_vals):
    gt = gate_type.lower()
    if gt == "dff":
        return in_vals[0] if in_vals else "X"
    fn = GATE_OP.get(gt)
    if fn is None:
        raise ValueError(f"Unknown gate type: {gate_type}")
    return fn(in_vals)


# ── Good-circuit simulation ──────────────────────────────────────────────────

def simulate_good(circuit, input_vector, initial_state=None):
    state = {}
    for pi in circuit.primary_inputs:
        raw = input_vector.get(pi, "X")
        if raw == "X" or raw == "x":
            state[pi] = "X"
        elif isinstance(raw, str) and raw.isdigit():
            state[pi] = int(raw)
        else:
            state[pi] = raw

    if initial_state:
        for ff_gname in circuit.flip_flops:
            q_wire = circuit.gates[ff_gname].output
            state[q_wire] = initial_state.get(q_wire, "X")
    else:
        for ff_gname in circuit.flip_flops:
            q_wire = circuit.gates[ff_gname].output
            state[q_wire] = "X"

    topo = topological_order(circuit)
    for gname in topo:
        gate = circuit.gates[gname]
        in_vals = [state.get(iw, "X") for iw in gate.inputs]
        state[gate.output] = _eval_gate(gate.gate_type, in_vals)

    return state


# ── Faulty-circuit simulation ────────────────────────────────────────────────

def simulate_faulty(circuit, input_vector, fault_wire, fault_type,
                    initial_state=None):
    state = simulate_good(circuit, input_vector, initial_state)
    stuck_val = 0 if fault_type.upper() == "SA0" else 1
    state[fault_wire] = stuck_val

    # Re-evaluate downstream gates
    topo = topological_order(circuit)
    pending = list(circuit.fanout.get(fault_wire, []))
    visited = set()
    while pending:
        gname = pending.pop()
        if gname in visited:
            continue
        visited.add(gname)
        gate = circuit.gates[gname]
        if gate.gate_type == "dff":
            continue  # DFFs handled by sequential frame logic
        in_vals = [state.get(iw, "X") for iw in gate.inputs]
        state[gate.output] = _eval_gate(gate.gate_type, in_vals)
        for succ in circuit.fanout.get(gate.output, []):
            pending.append(succ)

    return state


# ── Fault Simulator ──────────────────────────────────────────────────────────

class FaultSimulator:
    def __init__(self, circuit):
        self.circuit = circuit

    def simulate(self, test_vectors, fault_list):
        detected = set()
        remaining = set(map(tuple, fault_list))

        for tv_seq in test_vectors:
            if not remaining:
                break

            good_states = self._sim_seq_good(tv_seq)
            newly = set()
            for fault in remaining:
                fw, ft = fault
                faulty_states = self._sim_seq_faulty(tv_seq, fw, ft)
                if self._outputs_differ(good_states, faulty_states):
                    newly.add(fault)

            detected |= newly
            remaining -= newly

        return detected

    def _sim_seq_good(self, tv_seq):
        states = []
        ff_state = None
        for i, frame_inputs in enumerate(tv_seq):
            # On the first frame, extract FF initial states from test vector
            if i == 0 and ff_state is None:
                init_ff = {}
                for ff in self.circuit.flip_flops:
                    q_wire = self.circuit.gates[ff].output
                    if q_wire in frame_inputs:
                        raw = frame_inputs[q_wire]
                        init_ff[q_wire] = int(raw) if isinstance(raw, str) and raw.isdigit() else raw
                    else:
                        init_ff[q_wire] = "X"
                if any(v != "X" for v in init_ff.values()):
                    ff_state = init_ff
            state = simulate_good(self.circuit, frame_inputs, ff_state)
            # Next-state: use the D-input values as the new FF state
            next_ff = {}
            for ff in self.circuit.flip_flops:
                d_wire = self.circuit.gates[ff].inputs[0]
                next_ff[self.circuit.gates[ff].output] = state.get(d_wire, "X")
            states.append(state)
            ff_state = next_ff
        return states

    def _sim_seq_faulty(self, tv_seq, fault_wire, fault_type):
        """Permanent fault: inject stuck-at in EVERY frame of the sequence."""
        states = []
        ff_state = None
        for i, frame_inputs in enumerate(tv_seq):
            # On the first frame, extract FF initial states from test vector
            if i == 0 and ff_state is None:
                init_ff = {}
                for ff in self.circuit.flip_flops:
                    q_wire = self.circuit.gates[ff].output
                    if q_wire in frame_inputs:
                        raw = frame_inputs[q_wire]
                        init_ff[q_wire] = int(raw) if isinstance(raw, str) and raw.isdigit() else raw
                    else:
                        init_ff[q_wire] = "X"
                if any(v != "X" for v in init_ff.values()):
                    ff_state = init_ff
            # Simulate good first, then inject fault and re-propagate
            state = simulate_faulty(self.circuit, frame_inputs,
                                    fault_wire, fault_type, ff_state)
            # Next-state: capture D-inputs (with fault affecting them)
            next_ff = {}
            for ff in self.circuit.flip_flops:
                d_wire = self.circuit.gates[ff].inputs[0]
                d_val = state.get(d_wire, "X")
                # If the D-input wire IS the fault wire, it's stuck
                if d_wire == fault_wire:
                    d_val = 0 if fault_type.upper() == "SA0" else 1
                next_ff[self.circuit.gates[ff].output] = d_val
            states.append(state)
            ff_state = next_ff
        return states

    def _outputs_differ(self, good_states, faulty_states):
        for gs, fs in zip(good_states, faulty_states):
            for po in self.circuit.primary_outputs:
                gv = gs.get(po, "X")
                fv = fs.get(po, "X")
                if gv != "X" and fv != "X" and gv != fv:
                    return True
        return False

    def report(self, detected, fault_list):
        total = len(fault_list)
        n_det = len(detected)
        cov = 100.0 * n_det / total if total > 0 else 0.0

        print("=" * 50)
        print("  Fault Simulation Verification Report")
        print("=" * 50)
        print(f"  Total faults    : {total}")
        print(f"  Detected        : {n_det}")
        print(f"  Undetected      : {total - n_det}")
        print(f"  Fault Coverage  : {cov:.2f}%")
        print("=" * 50)
