"""
S-Graph Construction & Sequential Depth Calculation
=====================================================
Builds a dependency DAG between flip-flops (the S-Graph) and computes
the sequential depth d_seq of the circuit.

S-Graph definition:
  - Nodes: One per D-type flip-flop
  - Edges: FF_i → FF_j if there is a combinational path from FF_i.Q to FF_j.D

Sequential depth:
  - Level 1: FFs with in-degree 0 (driven only by Primary Inputs)
  - Level k: FFs whose predecessors all have level < k
  - d_seq = max level across all FFs
  - Time frame limit for ATPG = d_seq + 1
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from parser import Circuit


# ── S-Graph Builder ──────────────────────────────────────────────────────────

def build_sgraph(circuit: Circuit) -> Dict[str, Set[str]]:
    """
    Build the S-Graph: a dict mapping each FF gate name to the set of
    FF gate names it depends on (i.e., there is a combinational path
    from predecessor FF's Q to this FF's D).

    Returns:
        adjacency: {ff_gname: set of ff_gnames that feed into it}
        Also returns predecessors (reverse) for convenience.
    """
    # Map: FF output wire (Q) → FF gate name
    ff_q_to_name = {}
    # Map: FF gate name → D input wire
    ff_name_to_d = {}
    for ff_gname in circuit.flip_flops:
        gate = circuit.gates[ff_gname]
        ff_q_to_name[gate.output] = ff_gname
        ff_name_to_d[ff_gname] = gate.inputs[0]

    ff_q_wires = set(ff_q_to_name.keys())

    # For each FF, trace backward from its D-input through combinational
    # logic to find which FF Q-outputs (or PIs) can reach it.
    predecessors = {ff: set() for ff in circuit.flip_flops}

    for ff_gname in circuit.flip_flops:
        d_wire = ff_name_to_d[ff_gname]
        # BFS backward from d_wire through combinational logic
        visited = set()
        queue = deque([d_wire])

        while queue:
            wire = queue.popleft()
            if wire in visited:
                continue
            visited.add(wire)

            # If this wire is an FF output, record the dependency
            if wire in ff_q_to_name:
                src_ff = ff_q_to_name[wire]
                if src_ff != ff_gname:  # skip self-loops for level calc
                    predecessors[ff_gname].add(src_ff)
                continue  # Don't trace through FFs

            # If this wire is a PI, it has no further predecessors
            if wire in circuit.primary_inputs:
                continue

            # Find the gate that drives this wire
            driver_gname = circuit.fanin.get(wire)
            if driver_gname is None:
                continue
            driver_gate = circuit.gates.get(driver_gname)
            if driver_gate is None or driver_gate.gate_type == "dff":
                continue

            # Add all inputs of the driving gate to the queue
            for iw in driver_gate.inputs:
                if iw not in visited:
                    queue.append(iw)

    # Build forward adjacency: FF_i → FF_j means FF_i.Q feeds FF_j.D
    forward = {ff: set() for ff in circuit.flip_flops}
    for ff_j, preds in predecessors.items():
        for ff_i in preds:
            forward[ff_i].add(ff_j)

    return forward, predecessors


# ── Sequential Depth ─────────────────────────────────────────────────────────

def compute_sequential_depth(circuit: Circuit) -> Tuple[int, Dict[str, int]]:
    """
    Compute the sequential depth d_seq of the circuit.

    Algorithm:
      1. Build S-Graph (predecessor map)
      2. FFs with in-degree 0 → Level 1
      3. Propagate levels (BFS): level(FF) = max(level(pred)) + 1
      4. d_seq = max level

    Returns:
        (d_seq, {ff_gname: level})
    """
    forward, predecessors = build_sgraph(circuit)

    if not circuit.flip_flops:
        return 0, {}

    # Compute in-degree from the predecessor map
    in_degree = {ff: len(preds) for ff, preds in predecessors.items()}

    # Initialize: FFs with in-degree 0 get level 1
    levels = {}
    queue = deque()
    for ff in circuit.flip_flops:
        if in_degree[ff] == 0:
            levels[ff] = 1
            queue.append(ff)

    # BFS level propagation
    while queue:
        ff_i = queue.popleft()
        for ff_j in forward[ff_i]:
            # Update ff_j level if all predecessors are resolved
            pred_levels = []
            all_resolved = True
            for pred in predecessors[ff_j]:
                if pred in levels:
                    pred_levels.append(levels[pred])
                else:
                    all_resolved = False

            if all_resolved and ff_j not in levels:
                levels[ff_j] = max(pred_levels) + 1 if pred_levels else 1
                queue.append(ff_j)

    # Handle any unresolved FFs (cycles in S-Graph — shouldn't happen in DAG)
    for ff in circuit.flip_flops:
        if ff not in levels:
            # Assign based on what predecessors are available
            pred_levels = [levels.get(p, 0) for p in predecessors[ff]]
            levels[ff] = max(pred_levels) + 1 if pred_levels else 1

    d_seq = max(levels.values()) if levels else 0
    return d_seq, levels


# ── S-Graph Report ───────────────────────────────────────────────────────────

def print_sgraph_report(circuit: Circuit):
    """Pretty-print S-Graph structure, degrees, levels, and d_seq."""
    forward, predecessors = build_sgraph(circuit)
    d_seq, levels = compute_sequential_depth(circuit)

    print("\n" + "=" * 60)
    print("  S-Graph (Flip-Flop Dependency Graph)")
    print("=" * 60)

    for ff in circuit.flip_flops:
        gate = circuit.gates[ff]
        in_deg = len(predecessors[ff])
        out_deg = len(forward[ff])
        level = levels.get(ff, "?")
        preds_str = ", ".join(predecessors[ff]) if predecessors[ff] else "(PIs only)"
        succs_str = ", ".join(forward[ff]) if forward[ff] else "(none)"

        print(f"  {ff} (Q={gate.output}, D={gate.inputs[0]}):")
        print(f"    IN-degree  = {in_deg}  ← {preds_str}")
        print(f"    OUT-degree = {out_deg}  → {succs_str}")
        print(f"    Level      = {level}")

    print(f"\n  Sequential Depth (d_seq) = {d_seq}")
    print(f"  Time Frame Limit         = {d_seq + 1}")
    print("=" * 60)

    return d_seq


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from parser import parse_verilog

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"
    circ = parse_verilog(path)
    circ.summary()
    print_sgraph_report(circ)
