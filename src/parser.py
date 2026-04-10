"""
Netlist Parser — Gate-level Verilog → Internal Circuit Graph
=============================================================
Supports: and, or, nand, nor, not, buf, xor, xnor, dff
Handles ISCAS-89 style netlists with embedded dff module definitions.

DFF positional port convention (ISCAS-89): dff(CK, Q, D)
  ports[0] = CK  (clock — skipped for ATPG)
  ports[1] = Q   (output)
  ports[2] = D   (input)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Gate:
    name: str
    gate_type: str          # "and", "or", "nand", "nor", "not", "buf", "xor", "xnor", "dff"
    inputs: List[str]
    output: str
    time_frame: int = 0

    def __repr__(self):
        return f"Gate({self.name}, {self.gate_type}, in={self.inputs}, out={self.output})"


@dataclass
class Circuit:
    module_name: str = ""
    gates: Dict[str, Gate] = field(default_factory=dict)
    wires: Dict[str, str] = field(default_factory=dict)     # wire_name → value
    primary_inputs: List[str] = field(default_factory=list)
    primary_outputs: List[str] = field(default_factory=list)
    flip_flops: List[str] = field(default_factory=list)      # gate names of DFFs
    fanout: Dict[str, List[str]] = field(default_factory=dict)  # wire → [gate_names driven]
    fanin: Dict[str, str] = field(default_factory=dict)         # wire → gate_name driving it

    def summary(self):
        n_comb = sum(1 for g in self.gates.values() if g.gate_type != "dff")
        print(f"  Module        : {self.module_name}")
        print(f"  Comb. gates   : {n_comb}")
        print(f"  Flip-flops    : {len(self.flip_flops)}")
        print(f"  Wires         : {len(self.wires)}")
        print(f"  Primary In    : {self.primary_inputs}")
        print(f"  Primary Out   : {self.primary_outputs}")
        for ff in self.flip_flops:
            g = self.gates[ff]
            print(f"    DFF {g.name}: D={g.inputs[0]}, Q={g.output}")


# ── Helpers ──────────────────────────────────────────────────────────────────

COMB_GATES = {"and", "or", "nand", "nor", "not", "buf", "xor", "xnor"}


def _strip_comments(text):
    text = re.sub(r"//.*?\n", "\n", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return text


def _extract_top_module(text):
    """Extract only the top-level (last) module body from a multi-module Verilog file."""
    modules = list(re.finditer(
        r'\bmodule\s+(\w+)\s*\([^)]*\)\s*;(.*?)endmodule',
        text, re.DOTALL
    ))
    if not modules:
        raise ValueError("No module found in Verilog file.")
    m = modules[-1]
    return m.group(1), m.group(2)


def _parse_port_list(port_str):
    inner = port_str.strip().lstrip("(").rstrip(")")
    return [p.strip() for p in inner.split(",") if p.strip()]


# ── Main Parser ──────────────────────────────────────────────────────────────

def parse_verilog(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    text = _strip_comments(text)
    circuit = Circuit()

    module_name, body = _extract_top_module(text)
    circuit.module_name = module_name

    # ── Primary inputs / outputs ──────────────────────────────────────────
    for pi in re.findall(r'\binput\b\s+([\w\s,]+?);', body):
        for w in re.split(r'[\s,]+', pi):
            w = w.strip()
            if w and w not in ("GND", "VDD", "CK"):
                circuit.primary_inputs.append(w)
                circuit.wires[w] = "X"

    for po in re.findall(r'\boutput\b\s+([\w\s,]+?);', body):
        for w in re.split(r'[\s,]+', po):
            w = w.strip()
            if w:
                circuit.primary_outputs.append(w)
                circuit.wires.setdefault(w, "X")

    # ── Internal wires ────────────────────────────────────────────────────
    for wdecl in re.findall(r'\bwire\b\s+([\w\s,]+?);', body):
        for w in re.split(r'[\s,]+', wdecl):
            w = w.strip()
            if w:
                circuit.wires.setdefault(w, "X")

    # ── Gate instantiations ───────────────────────────────────────────────
    gate_pat = re.compile(r'\b(\w+)\s+(\w+)\s*\(([^)]+)\)\s*;', re.DOTALL)

    for m in gate_pat.finditer(body):
        gtype_raw = m.group(1).lower()
        gname     = m.group(2)
        ports_raw = m.group(3)
        ports = _parse_port_list(ports_raw)

        output_wire = None
        input_wires = []

        if gtype_raw == "dff":
            if len(ports) >= 3:
                output_wire = ports[1]
                input_wires = [ports[2]]
            elif len(ports) == 2:
                output_wire = ports[0]
                input_wires = [ports[1]]
            else:
                continue
        elif gtype_raw in COMB_GATES:
            if not ports:
                continue
            output_wire = ports[0]
            input_wires = ports[1:]
        else:
            continue

        circuit.wires.setdefault(output_wire, "X")
        for iw in input_wires:
            circuit.wires.setdefault(iw, "X")

        gate = Gate(
            name=gname,
            gate_type=gtype_raw if gtype_raw != "dff" else "dff",
            inputs=input_wires,
            output=output_wire,
            time_frame=0,
        )
        circuit.gates[gname] = gate

        if gtype_raw == "dff":
            circuit.flip_flops.append(gname)

    # ── Build fanout / fanin ──────────────────────────────────────────────
    for gname, gate in circuit.gates.items():
        circuit.fanin[gate.output] = gname
        for iw in gate.inputs:
            circuit.fanout.setdefault(iw, []).append(gname)

    return circuit


# ── Topological sort ─────────────────────────────────────────────────────────

def topological_order(circuit):
    """
    Return combinational gates in topological order.
    DFF outputs are treated as pseudo-primary inputs (sources).
    """
    from collections import deque

    dff_outputs = {circuit.gates[ff].output for ff in circuit.flip_flops}
    pi_wires    = set(circuit.primary_inputs) | dff_outputs

    comb_gates = {g for g, gate in circuit.gates.items() if gate.gate_type != "dff"}

    in_degree = {g: 0 for g in comb_gates}
    for g in comb_gates:
        gate = circuit.gates[g]
        for iw in gate.inputs:
            driver = circuit.fanin.get(iw)
            if driver and driver in comb_gates:
                in_degree[g] += 1

    queue = deque([g for g in comb_gates if in_degree[g] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        out_wire = circuit.gates[node].output
        for succ in circuit.fanout.get(out_wire, []):
            if succ in comb_gates:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

    return order


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"
    c = parse_verilog(path)
    c.summary()
    topo = topological_order(c)
    print(f"\nTopological order ({len(topo)} comb. gates):")
    for g in topo:
        gate = c.gates[g]
        print(f"  {g} ({gate.gate_type}): {gate.inputs} → {gate.output}")
