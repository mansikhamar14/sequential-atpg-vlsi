"""
Time Frame Expansion — Sequential Circuit Unroller
===================================================
Unrolls a sequential circuit with DFFs into k combinational copies.

Time frame labeling:
  Frame 0  : current cycle (fault activation + propagation target)
  Frame -1 : one cycle before (initialization)
  Frame -2 : two cycles before
  ...

Each DFF output Q in frame t is connected to DFF input D in frame t-1.
DFF outputs in the earliest frame are "pseudo-primary inputs" (PPI).

The target fault is present in EVERY replicated copy of the circuit
to match the permanent fault model.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from parser import Circuit, Gate


@dataclass
class UnrolledCircuit:
    """
    A flat circuit obtained by unrolling a sequential circuit over
    `num_frames` time steps.
    """
    num_frames: int
    original_name: str

    # All gates across all frames: (frame_index, original_gate_name) → Gate
    gates: Dict[Tuple[int, str], Gate] = field(default_factory=dict)

    # All wires: wire_key → value (initial "X")
    wires: Dict[str, str] = field(default_factory=dict)

    # PI/PO in the unrolled circuit (wire keys)
    primary_inputs: List[str] = field(default_factory=list)
    primary_outputs: List[str] = field(default_factory=list)

    # Pseudo-primary inputs (DFF Q outputs in earliest frame)
    pseudo_primary_inputs: List[str] = field(default_factory=list)

    # Wire-to-driver map
    fanin: Dict[str, object] = field(default_factory=dict)     # wire_key → (frame, gname) or ("DFF_CONNECT", wire_key)
    fanout: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)

    def wire_key(self, wire_name, frame):
        return f"{wire_name}@{frame}"

    def get_wire(self, wire_name, frame):
        return self.wires.get(self.wire_key(wire_name, frame), "X")

    def set_wire(self, wire_name, frame, value):
        self.wires[self.wire_key(wire_name, frame)] = value

    def summary(self):
        print(f"\n  Unrolled Circuit: {self.original_name}")
        print(f"    Frames      : {self.num_frames}")
        print(f"    Total gates : {len(self.gates)}")
        print(f"    Total wires : {len(self.wires)}")
        print(f"    PIs (real)  : {len(self.primary_inputs)}")
        print(f"    POs (real)  : {len(self.primary_outputs)}")
        print(f"    PPIs        : {len(self.pseudo_primary_inputs)}")


# ── Unroller ─────────────────────────────────────────────────────────────────

def unroll(circuit: Circuit, num_frames: int) -> UnrolledCircuit:
    """
    Unroll `circuit` into `num_frames` time-frame copies.
    Frame indices: -(num_frames-1) to 0.
    """
    assert num_frames >= 1

    uc = UnrolledCircuit(
        num_frames=num_frames,
        original_name=circuit.module_name,
    )

    frames = list(range(-(num_frames - 1), 1))
    earliest = frames[0]

    # ── Build combinational gates for each frame ─────────────────────────
    for t in frames:
        for gname, gate in circuit.gates.items():
            if gate.gate_type == "dff":
                continue

            new_gate = Gate(
                name=gname,
                gate_type=gate.gate_type,
                inputs=[uc.wire_key(iw, t) for iw in gate.inputs],
                output=uc.wire_key(gate.output, t),
                time_frame=t,
            )
            uc.gates[(t, gname)] = new_gate

            uc.wires.setdefault(new_gate.output, "X")
            uc.fanin[new_gate.output] = (t, gname)

            for wk in new_gate.inputs:
                uc.wires.setdefault(wk, "X")
                uc.fanout.setdefault(wk, []).append((t, gname))

    # ── Primary inputs — appear in every frame ───────────────────────────
    for t in frames:
        for pi in circuit.primary_inputs:
            wk = uc.wire_key(pi, t)
            uc.wires.setdefault(wk, "X")
            uc.primary_inputs.append(wk)

    # ── Primary outputs — only from frame 0 ──────────────────────────────
    for po in circuit.primary_outputs:
        wk = uc.wire_key(po, 0)
        uc.wires.setdefault(wk, "X")
        uc.primary_outputs.append(wk)

    # ── DFF inter-frame connections ───────────────────────────────────────
    for ff_gname in circuit.flip_flops:
        ff_gate = circuit.gates[ff_gname]
        d_wire = ff_gate.inputs[0]
        q_wire = ff_gate.output

        for idx, t in enumerate(frames):
            q_wk = uc.wire_key(q_wire, t)
            uc.wires.setdefault(q_wk, "X")

            if t == earliest:
                uc.pseudo_primary_inputs.append(q_wk)
            else:
                prev_t = frames[idx - 1]
                d_wk = uc.wire_key(d_wire, prev_t)
                uc.wires.setdefault(d_wk, "X")
                uc.fanin[q_wk] = ("DFF_CONNECT", d_wk)

            d_wk = uc.wire_key(d_wire, t)
            uc.wires.setdefault(d_wk, "X")

    return uc


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from parser import parse_verilog

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"
    nf   = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    circ = parse_verilog(path)
    circ.summary()
    uc = unroll(circ, nf)
    uc.summary()
    print(f"\n  PPIs: {uc.pseudo_primary_inputs}")
    print(f"  POs:  {uc.primary_outputs}")
