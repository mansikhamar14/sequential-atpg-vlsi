"""
Interactive Circuit & ATPG Visualization
=========================================
Renders circuits, S-Graphs, and ATPG algorithm traces as interactive HTML
using pyvis (vis.js). Supports drag-to-move, zoom, pan, and node tooltips.

Entry points:
  render_circuit_html(circuit, out_path)           → gate-level netlist
  render_sgraph_html(circuit, out_path)            → flip-flop dependency DAG
  render_atpg_trace_html(circuit, ..., out_path)   → step-by-step ATPG trace

CLI:
  python src/visualize.py <verilog> [--out <dir>] [--trace <wire> <SA0|SA1>]
"""

import json
import os
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyvis.network import Network

from parser import Circuit, parse_verilog, topological_order
from sgraph import build_sgraph, compute_sequential_depth


# ── Styling ───────────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "pi":   {"background": "#a7e9b3", "border": "#2d8a3e"},
    "po":   {"background": "#ffd39a", "border": "#d97706"},
    "ff":   {"background": "#ffb3d1", "border": "#be185d"},
    "and":  {"background": "#b3d9ff", "border": "#1971c2"},
    "nand": {"background": "#b3d9ff", "border": "#1971c2"},
    "or":   {"background": "#c2c6ff", "border": "#4c4fbd"},
    "nor":  {"background": "#c2c6ff", "border": "#4c4fbd"},
    "not":  {"background": "#e5d4ff", "border": "#7c4dff"},
    "buf":  {"background": "#e5d4ff", "border": "#7c4dff"},
    "xor":  {"background": "#ffe0a3", "border": "#c98500"},
    "xnor": {"background": "#ffe0a3", "border": "#c98500"},
}

GATE_SHAPE = {
    "and":  "circle",   "nand": "circle",
    "or":   "diamond",  "nor":  "diamond",
    "not":  "triangle", "buf":  "triangle",
    "xor":  "star",     "xnor": "star",
}


def _node_style(role):
    """Returns (color_dict, shape) for a vis.js node."""
    color = ROLE_COLORS.get(role, {"background": "#eceff4", "border": "#4c566a"}).copy()
    if role in ("pi", "po"):
        shape = "ellipse"
    elif role == "ff":
        shape = "database"
    elif role in ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf"):
        shape = "box"  # keep box for readability; gate-type shown in label
    else:
        shape = "box"
    return color, shape


def _base_network(hierarchical=True, direction="LR"):
    """Create a pyvis Network with sensible defaults."""
    net = Network(
        height="800px", width="100%",
        bgcolor="#f9fbfe", font_color="#10243f",
        directed=True,
        cdn_resources="remote",
    )
    opts = {
        "nodes": {
            "font": {"size": 14, "face": "Segoe UI, Arial, sans-serif", "color": "#10243f"},
            "borderWidth": 2,
            "shadow": {"enabled": True, "size": 4, "x": 1, "y": 2},
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4},
            "color": {"color": "#8fa2ba", "highlight": "#1971c2"},
            "width": 1.5,
        },
        "interaction": {
            "hover": True, "navigationButtons": True, "keyboard": True,
            "tooltipDelay": 120,
        },
        "physics": {"enabled": False},
    }
    if hierarchical:
        opts["layout"] = {
            "hierarchical": {
                "enabled": True,
                "direction": direction,
                "sortMethod": "directed",
                "levelSeparation": 220,
                "nodeSpacing": 140,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
            }
        }
    else:
        opts["physics"] = {
            "enabled": True,
            "barnesHut": {"gravitationalConstant": -8000, "springLength": 150},
            "stabilization": {"iterations": 200},
        }
    net.set_options(json.dumps(opts))
    return net


# ── 1. Circuit Netlist ────────────────────────────────────────────────────────

def build_circuit_network(circuit: Circuit,
                          fault_wire: Optional[str] = None,
                          wire_values: Optional[Dict[str, str]] = None) -> Network:
    """
    Build a pyvis Network representing the gate-level circuit.
    Nodes are PIs, POs, gates, and DFFs.
    Edges are wires (labeled with the wire name).
    """
    net = _base_network(hierarchical=True, direction="LR")

    # Add PIs
    pi_color, pi_shape = _node_style("pi")
    for pi in circuit.primary_inputs:
        net.add_node(
            f"PI:{pi}",
            label=pi,
            title=f"Primary Input: {pi}",
            color=pi_color, shape=pi_shape,
            level=0,
        )

    # Figure out gate levels via topo sort
    topo = topological_order(circuit)
    gate_level: Dict[str, int] = {}
    for gname in topo:
        gate = circuit.gates[gname]
        max_in = 1
        for iw in gate.inputs:
            driver = circuit.fanin.get(iw)
            if driver is None:
                max_in = max(max_in, 1)  # PI input
            elif driver in gate_level:
                max_in = max(max_in, gate_level[driver] + 1)
        gate_level[gname] = max_in

    max_gate_level = max(gate_level.values(), default=1)

    # Combinational gates
    for gname, gate in circuit.gates.items():
        if gate.gate_type == "dff":
            continue
        lvl = gate_level.get(gname, 1)
        fault_marker = " 🔥" if fault_wire and gate.output == fault_wire else ""
        label = f"{gate.gate_type.upper()}\n{gname}{fault_marker}"
        title = (
            f"<b>{gname}</b> ({gate.gate_type.upper()})<br>"
            f"inputs: {', '.join(gate.inputs)}<br>"
            f"output: {gate.output}"
        )
        g_color, g_shape = _node_style(gate.gate_type)
        border_color = "#e03131" if fault_wire and gate.output == fault_wire else g_color["border"]
        net.add_node(
            f"G:{gname}",
            label=label, title=title, level=lvl,
            color={"background": g_color["background"], "border": border_color},
            shape=g_shape,
            borderWidth=3 if fault_wire and gate.output == fault_wire else 2,
        )

    # DFFs — on the right
    ff_color, ff_shape = _node_style("ff")
    for ff_gname in circuit.flip_flops:
        gate = circuit.gates[ff_gname]
        label = f"DFF\n{ff_gname}"
        title = (
            f"<b>{ff_gname}</b> (DFF)<br>"
            f"D = {gate.inputs[0]}<br>"
            f"Q = {gate.output}"
        )
        net.add_node(
            f"FF:{ff_gname}",
            label=label, title=title, level=max_gate_level + 1,
            color=ff_color, shape=ff_shape,
        )

    # POs — far right
    po_color, po_shape = _node_style("po")
    for po in circuit.primary_outputs:
        net.add_node(
            f"PO:{po}",
            label=po, title=f"Primary Output: {po}",
            level=max_gate_level + 2,
            color=po_color, shape=po_shape,
        )

    # Edges — one per wire
    def wire_source_node(wire):
        driver = circuit.fanin.get(wire)
        if driver is None:
            if wire in circuit.primary_inputs:
                return f"PI:{wire}"
            return None
        gate = circuit.gates.get(driver)
        if gate is None:
            return None
        if gate.gate_type == "dff":
            return f"FF:{driver}"
        return f"G:{driver}"

    def edge_style(wire):
        val = (wire_values or {}).get(wire)
        if val in ("D", "1/0"):
            return {"color": "#e03131", "width": 3}
        if val in ("D_BAR", "0/1"):
            return {"color": "#b02a37", "width": 3, "dashes": True}
        if val in ("1", "1/1"):
            return {"color": "#1971c2"}
        if val in ("0", "0/0"):
            return {"color": "#495057"}
        return {}

    seen_edges = set()

    for gname, gate in circuit.gates.items():
        if gate.gate_type == "dff":
            # D input edge
            src = wire_source_node(gate.inputs[0])
            if src is None:
                continue
            key = (src, f"FF:{gname}", gate.inputs[0])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            net.add_edge(src, f"FF:{gname}",
                         title=f"wire: {gate.inputs[0]}",
                         label=gate.inputs[0],
                         font={"size": 9, "color": "#5b6f86"},
                         **edge_style(gate.inputs[0]))
            continue
        for iw in gate.inputs:
            src = wire_source_node(iw)
            if src is None:
                continue
            key = (src, f"G:{gname}", iw)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            net.add_edge(src, f"G:{gname}",
                         title=f"wire: {iw}",
                         label=iw,
                         font={"size": 9, "color": "#5b6f86"},
                         **edge_style(iw))

    # PO edges
    for po in circuit.primary_outputs:
        src = wire_source_node(po)
        if src is None:
            continue
        key = (src, f"PO:{po}", po)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        net.add_edge(src, f"PO:{po}", title=f"wire: {po}", label=po,
                     font={"size": 9, "color": "#5b6f86"},
                     **edge_style(po))

    # DFF feedback (Q→next-cycle input) — dashed backwards edge
    for ff_gname in circuit.flip_flops:
        gate = circuit.gates[ff_gname]
        q_wire = gate.output
        for dst_gate in circuit.fanout.get(q_wire, []):
            dst_gate_obj = circuit.gates[dst_gate]
            dst_id = f"FF:{dst_gate}" if dst_gate_obj.gate_type == "dff" else f"G:{dst_gate}"
            key = (f"FF:{ff_gname}", dst_id, q_wire)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            net.add_edge(f"FF:{ff_gname}", dst_id,
                         title=f"wire: {q_wire} (feedback)",
                         label=q_wire,
                         font={"size": 9, "color": "#be185d"},
                         color={"color": "#be185d"},
                         dashes=True)

    return net


def render_circuit_html(circuit: Circuit, out_path,
                        fault_wire: Optional[str] = None) -> Path:
    net = build_circuit_network(circuit, fault_wire=fault_wire)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_with_header(net, out_path,
                      title=f"Netlist: {circuit.module_name}",
                      subtitle=_circuit_subtitle(circuit))
    return out_path


def _circuit_subtitle(circuit):
    n_comb = sum(1 for g in circuit.gates.values() if g.gate_type != "dff")
    return (
        f"{n_comb} gates · {len(circuit.flip_flops)} flip-flops · "
        f"{len(circuit.primary_inputs)} PIs · {len(circuit.primary_outputs)} POs"
    )


# ── 2. S-Graph ─────────────────────────────────────────────────────────────────

def render_sgraph_html(circuit: Circuit, out_path) -> Path:
    net = _base_network(hierarchical=True, direction="LR")

    if not circuit.flip_flops:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "<html><body><h2>No flip-flops in circuit — S-Graph is empty.</h2></body></html>",
            encoding="utf-8",
        )
        return out_path

    forward, predecessors = build_sgraph(circuit)
    d_seq, levels = compute_sequential_depth(circuit)

    ff_color, ff_shape = _node_style("ff")
    for ff in circuit.flip_flops:
        gate = circuit.gates[ff]
        lvl = levels.get(ff, 1)
        label = f"{ff}\nQ={gate.output} L{lvl}"
        title = (
            f"<b>{ff}</b><br>D = {gate.inputs[0]}<br>Q = {gate.output}<br>"
            f"Level = {lvl}<br>in-degree = {len(predecessors[ff])}<br>"
            f"out-degree = {len(forward[ff])}"
        )
        net.add_node(f"FF:{ff}", label=label, title=title, level=lvl,
                     color=ff_color, shape=ff_shape)

    for src, succs in forward.items():
        for dst in succs:
            net.add_edge(f"FF:{src}", f"FF:{dst}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_with_header(
        net, out_path,
        title=f"S-Graph: {circuit.module_name}",
        subtitle=f"d_seq = {d_seq} · time-frame limit = {d_seq + 1} · {len(circuit.flip_flops)} FFs",
    )
    return out_path


# ── 3. ATPG Trace ─────────────────────────────────────────────────────────────

class TraceRecorder:
    """Records ATPG state snapshots for later rendering."""

    def __init__(self, circuit: Circuit, num_frames: int,
                 fault_wire: str, fault_type: str, max_steps: int = 40):
        self.circuit = circuit
        self.num_frames = num_frames
        self.fault_wire = fault_wire
        self.fault_type = fault_type
        self.max_steps = max_steps
        self.snapshots: List[dict] = []

    def record(self, label: str, state: dict,
               frontier_keys=None, objective_wire=None):
        if len(self.snapshots) >= self.max_steps:
            return
        snap = {k: v for k, v in state.items() if not k.startswith("__")}
        self.snapshots.append({
            "label": label,
            "state": snap,
            "frontier_keys": list(frontier_keys) if frontier_keys else [],
            "objective_wire": objective_wire,
        })


def build_unrolled_network(circuit: Circuit, num_frames: int,
                            state: Dict, frontier_keys, objective_wire,
                            fault_wire, fault_type) -> Network:
    """Build a pyvis Network for one trace snapshot with time frames as groups."""
    from timeframe import unroll
    uc = unroll(circuit, num_frames)
    frames = list(range(-(num_frames - 1), 1))
    frontier_set = set(tuple(k) if isinstance(k, list) else k for k in frontier_keys)

    net = _base_network(hierarchical=True, direction="LR")

    topo = topological_order(circuit)
    gate_level: Dict[str, int] = {}
    for gname in topo:
        gate = circuit.gates[gname]
        max_in = 1
        for iw in gate.inputs:
            driver = circuit.fanin.get(iw)
            if driver is None:
                max_in = max(max_in, 1)
            elif driver in gate_level:
                max_in = max(max_in, gate_level[driver] + 1)
        gate_level[gname] = max_in

    max_gate_level = max(gate_level.values(), default=1)
    level_width = max_gate_level + 3  # PI col + comb + PO col + PPI col

    def get_val(wire_name, t):
        return state.get(uc.wire_key(wire_name, t), "X/X" if any("/" in str(v) for v in state.values()) else "X")

    def val_display(v):
        s = str(v)
        if s == "D_BAR":
            return "D̄"
        if s in ("X", "X/X"):
            return ""
        return s

    def val_border(v):
        s = str(v)
        if s in ("D", "1/0", "D_BAR", "0/1") or (isinstance(s, str) and "/" in s and "X" not in s and s.split("/")[0] != s.split("/")[1]):
            return "#e03131"
        if s in ("1", "1/1"):
            return "#1971c2"
        if s in ("0", "0/0"):
            return "#495057"
        return None

    def edge_color(v):
        s = str(v)
        if s in ("D", "1/0"):
            return {"color": "#e03131", "width": 3.5}
        if s in ("D_BAR", "0/1"):
            return {"color": "#b02a37", "width": 3.5, "dashes": True}
        if s in ("1", "1/1"):
            return {"color": "#1971c2", "width": 2}
        if s in ("0", "0/0"):
            return {"color": "#495057", "width": 2}
        if "/" in s and s != "X/X":
            return {"color": "#e03131", "width": 2.5}
        return {}

    # Build per-frame
    for fi, t in enumerate(frames):
        # Labels/groups
        frame_prefix = f"t{t}"

        for pi_idx, pi in enumerate(circuit.primary_inputs):
            val = val_display(get_val(pi, t))
            node_id = f"{frame_prefix}:PI:{pi}"
            wk = uc.wire_key(pi, t)
            lvl = fi * level_width + 0
            base_color, _ = _node_style("pi")
            border = val_border(get_val(pi, t)) or base_color["border"]
            if wk == objective_wire:
                border = "#9c36b5"
            label = f"{pi}[t={t}]\n{val}" if val else f"{pi}[t={t}]"
            net.add_node(node_id, label=label,
                         title=f"PI {pi} @ frame {t}<br>value = {get_val(pi, t)}",
                         level=lvl, shape="ellipse",
                         color={"background": base_color["background"], "border": border},
                         borderWidth=3 if wk == objective_wire else 2)

        for gname, gate in circuit.gates.items():
            if gate.gate_type == "dff":
                continue
            node_id = f"{frame_prefix}:G:{gname}"
            wk = uc.wire_key(gate.output, t)
            val = val_display(get_val(gate.output, t))
            lvl = fi * level_width + gate_level.get(gname, 1)
            g_color, g_shape = _node_style(gate.gate_type)
            border = val_border(get_val(gate.output, t)) or g_color["border"]
            extras = []
            if (t, gname) in frontier_set:
                border = "#f59f00"
                extras.append("D-frontier")
            if gate.output == fault_wire:
                border = "#e03131"
                extras.append("fault site")
            if wk == objective_wire:
                border = "#9c36b5"
                extras.append("objective")
            label_v = f"\n{val}" if val else ""
            label = f"{gate.gate_type.upper()}\n{gname}{label_v}"
            title = (
                f"<b>{gname}</b> ({gate.gate_type.upper()}) @ t={t}<br>"
                f"out = {gate.output} = {get_val(gate.output, t)}<br>"
                f"inputs: {', '.join(gate.inputs)}"
            )
            if extras:
                title += f"<br><i>{', '.join(extras)}</i>"
            net.add_node(node_id, label=label, title=title, level=lvl,
                         shape=g_shape,
                         color={"background": g_color["background"], "border": border},
                         borderWidth=3 if extras else 2)

        for po in circuit.primary_outputs:
            node_id = f"{frame_prefix}:PO:{po}"
            wk = uc.wire_key(po, t)
            val = val_display(get_val(po, t))
            lvl = fi * level_width + max_gate_level + 1
            po_color, _ = _node_style("po")
            border = val_border(get_val(po, t)) or po_color["border"]
            label = f"{po}[t={t}]\n{val}" if val else f"{po}[t={t}]"
            net.add_node(node_id, label=label,
                         title=f"PO {po} @ t={t}<br>value = {get_val(po, t)}",
                         level=lvl, shape="ellipse",
                         color={"background": po_color["background"], "border": border},
                         borderWidth=3 if border == "#e03131" else 2)

        # PPIs shown only in earliest frame
        if fi == 0:
            for ff in circuit.flip_flops:
                gate = circuit.gates[ff]
                node_id = f"{frame_prefix}:PPI:{ff}"
                q_wire = gate.output
                val = val_display(get_val(q_wire, t))
                lvl = fi * level_width + max_gate_level + 2
                ff_color_loc, _ = _node_style("ff")
                border = val_border(get_val(q_wire, t)) or ff_color_loc["border"]
                label = f"PPI {ff}\nQ={q_wire} {val}" if val else f"PPI {ff}\nQ={q_wire}"
                net.add_node(node_id, label=label,
                             title=f"Pseudo-primary input: {ff}.Q = {q_wire}<br>value = {get_val(q_wire, t)}",
                             level=lvl, shape="database",
                             color={"background": ff_color_loc["background"], "border": border})

    # Edges within each frame
    for fi, t in enumerate(frames):
        frame_prefix = f"t{t}"

        def src_id(wire):
            driver = circuit.fanin.get(wire)
            if driver is None:
                if wire in circuit.primary_inputs:
                    return f"{frame_prefix}:PI:{wire}"
                return None
            gate = circuit.gates.get(driver)
            if gate is None:
                return None
            if gate.gate_type == "dff":
                # DFF Q-output: from PPI in earliest frame, or from prev frame's D-input
                if fi == 0:
                    return f"{frame_prefix}:PPI:{driver}"
                # Prev frame's D driver
                prev_t = frames[fi - 1]
                return src_id_at_frame(gate.inputs[0], prev_t)
            return f"{frame_prefix}:G:{driver}"

        def src_id_at_frame(wire, tt):
            pref = f"t{tt}"
            driver = circuit.fanin.get(wire)
            if driver is None:
                if wire in circuit.primary_inputs:
                    return f"{pref}:PI:{wire}"
                return None
            gate = circuit.gates.get(driver)
            if gate is None or gate.gate_type == "dff":
                return None
            return f"{pref}:G:{driver}"

        # Gate input edges
        for gname, gate in circuit.gates.items():
            if gate.gate_type == "dff":
                continue
            dst = f"{frame_prefix}:G:{gname}"
            for iw in gate.inputs:
                src = src_id(iw)
                if src is None:
                    continue
                style = edge_color(get_val(iw, t))
                net.add_edge(src, dst, title=f"{iw} @ t={t} = {get_val(iw, t)}", **style)

        # PO edges
        for po in circuit.primary_outputs:
            dst = f"{frame_prefix}:PO:{po}"
            src = src_id(po)
            if src is None:
                continue
            style = edge_color(get_val(po, t))
            net.add_edge(src, dst, title=f"{po} @ t={t} = {get_val(po, t)}", **style)

    return net


def render_atpg_trace_html(recorder: TraceRecorder, out_path) -> Path:
    """Render a multi-step trace as a single HTML with step navigation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build each step's network HTML, extract body, combine into tabs
    step_htmls = []
    for snap in recorder.snapshots:
        net = build_unrolled_network(
            recorder.circuit, recorder.num_frames,
            snap["state"], snap["frontier_keys"], snap["objective_wire"],
            recorder.fault_wire, recorder.fault_type,
        )
        tmp = out_path.parent / f"_tmp_step.html"
        net.save_graph(str(tmp))
        html = tmp.read_text(encoding="utf-8")
        tmp.unlink()
        # Extract just the body content (everything between <body> and </body>)
        body_start = html.find("<body>") + len("<body>")
        body_end = html.find("</body>")
        body = html[body_start:body_end]
        # Extract head scripts/styles so vis.js loads
        head_start = html.find("<head>") + len("<head>")
        head_end = html.find("</head>")
        head = html[head_start:head_end]
        step_htmls.append({"head": head, "body": body, "label": snap["label"]})

    # Use only the first step's head (they're identical pyvis scaffolding)
    common_head = step_htmls[0]["head"] if step_htmls else ""

    # Build step tabs — each "step" gets a unique div id so we can show/hide
    tab_buttons = "".join(
        f'<button class="step-btn" data-step="{i}">{i+1}</button>'
        for i in range(len(step_htmls))
    )
    tab_labels = "".join(
        f'<div class="step-label" data-step="{i}" style="display:{"block" if i==0 else "none"}">'
        f'<b>Step {i+1}/{len(step_htmls)}:</b> {escape(s["label"])}</div>'
        for i, s in enumerate(step_htmls)
    )

    # Each step's body has a <div id="mynetwork"> and an inline <script> that
    # creates a vis Network. We need to rename the div + make the script fire
    # only when that step is active.
    panels = []
    for i, s in enumerate(step_htmls):
        body = s["body"]
        body = body.replace('id="mynetwork"', f'id="mynetwork{i}"')
        body = body.replace('"mynetwork"', f'"mynetwork{i}"')
        panels.append(
            f'<div class="panel" data-step="{i}" '
            f'style="display:{"block" if i==0 else "none"}; height: 85vh;">{body}</div>'
        )

    html_out = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ATPG Trace: {escape(recorder.fault_wire)}/{escape(recorder.fault_type)}</title>
{common_head}
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #eef3f8; }}
  header {{ background: #10243f; color: white; padding: 14px 22px; }}
  header h1 {{ margin: 0; font-size: 17px; }}
  header .sub {{ font-size: 12px; opacity: 0.8; margin-top: 2px; }}
  .controls {{ padding: 10px 22px; background: white; border-bottom: 1px solid #d0dae5;
               position: sticky; top: 0; z-index: 100;
               display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .step-btn {{ padding: 4px 10px; border: 1px solid #8fa2ba; background: white;
               cursor: pointer; border-radius: 4px; font-size: 12px; }}
  .step-btn:hover {{ background: #e4f1ff; }}
  .step-btn.active {{ background: #1971c2; color: white; border-color: #1971c2; }}
  .step-label {{ padding: 8px 22px; background: #f1f5fa; font-size: 13px;
                  border-bottom: 1px solid #d0dae5; }}
  .panel {{ background: #f9fbfe; }}
  .legend {{ display: flex; gap: 10px; font-size: 11px; margin-left: auto; flex-wrap: wrap; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .legend .swatch {{ width: 16px; height: 10px; display: inline-block; border-radius: 2px; }}
</style>
</head>
<body>
  <header>
    <h1>ATPG Trace: fault {escape(recorder.fault_wire)} / {escape(recorder.fault_type)}</h1>
    <div class="sub">{escape(recorder.circuit.module_name)} — {recorder.num_frames} time frames — {len(step_htmls)} steps</div>
  </header>
  <div class="controls">
    <strong>Steps:</strong> {tab_buttons}
    <span class="legend">
      <span><span class="swatch" style="background:#e03131"></span>D / D̄ (fault effect)</span>
      <span><span class="swatch" style="background:#1971c2"></span>1</span>
      <span><span class="swatch" style="background:#495057"></span>0</span>
      <span><span class="swatch" style="background:#f59f00"></span>D-frontier</span>
      <span><span class="swatch" style="background:#9c36b5"></span>Objective</span>
    </span>
  </div>
  {tab_labels}
  {"".join(panels)}
  <script>
    function showStep(i) {{
      document.querySelectorAll('.panel').forEach(p => {{
        p.style.display = (parseInt(p.dataset.step) === i) ? 'block' : 'none';
      }});
      document.querySelectorAll('.step-label').forEach(l => {{
        l.style.display = (parseInt(l.dataset.step) === i) ? 'block' : 'none';
      }});
      document.querySelectorAll('.step-btn').forEach(b => {{
        b.classList.toggle('active', parseInt(b.dataset.step) === i);
      }});
      // Trigger network fit for newly visible panel
      if (typeof window['network' + i] !== 'undefined') {{
        window['network' + i].fit();
      }}
    }}
    document.querySelectorAll('.step-btn').forEach(b => {{
      b.addEventListener('click', () => showStep(parseInt(b.dataset.step)));
    }});
    document.addEventListener('keydown', (e) => {{
      const active = document.querySelector('.step-btn.active');
      const idx = active ? parseInt(active.dataset.step) : 0;
      if (e.key === 'ArrowRight') showStep(Math.min(idx + 1, {len(step_htmls) - 1}));
      if (e.key === 'ArrowLeft') showStep(Math.max(idx - 1, 0));
    }});
    showStep(0);
  </script>
</body>
</html>"""

    out_path.write_text(html_out, encoding="utf-8")
    return out_path


# ── HTML wrapper helper ────────────────────────────────────────────────────────

def _save_with_header(net: Network, out_path: Path, title: str, subtitle: str = ""):
    """Save a pyvis network with a custom header banner."""
    tmp = out_path.with_suffix(".tmp.html")
    net.save_graph(str(tmp))
    html = tmp.read_text(encoding="utf-8")
    tmp.unlink()

    header = f"""
<div style="background:#10243f; color:white; padding:14px 22px; font-family:'Segoe UI',Arial,sans-serif;">
  <div style="font-size:17px; font-weight:700;">{escape(title)}</div>
  <div style="font-size:12px; opacity:0.8; margin-top:2px;">{escape(subtitle)}</div>
</div>
<div style="padding:8px 22px; background:#f1f5fa; border-bottom:1px solid #d0dae5; font-size:11px; font-family:'Segoe UI',Arial,sans-serif;">
  <b>Tip:</b> scroll to zoom, drag to pan, drag nodes to rearrange, hover for details.
</div>
"""
    html = html.replace("<body>", "<body>" + header, 1)
    out_path.write_text(html, encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Interactive circuit & ATPG visualization")
    ap.add_argument("verilog", help="Path to Verilog benchmark")
    ap.add_argument("--out", default="results/vis", help="Output directory")
    ap.add_argument("--trace", nargs=2, metavar=("WIRE", "TYPE"),
                    help="Render ATPG trace for this fault (WIRE SA0|SA1)")
    ap.add_argument("--algorithm", choices=["ext_d", "9val"], default="9val")
    ap.add_argument("--frames", type=int, default=0,
                    help="Time frames for trace (0 = auto from d_seq+1)")
    args = ap.parse_args()

    circuit = parse_verilog(args.verilog)
    bench_name = Path(args.verilog).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Static netlist
    p1 = render_circuit_html(circuit, out_dir / f"{bench_name}_netlist.html",
                              fault_wire=args.trace[0] if args.trace else None)
    print(f"  Wrote {p1}")

    # S-Graph
    p2 = render_sgraph_html(circuit, out_dir / f"{bench_name}_sgraph.html")
    print(f"  Wrote {p2}")

    if args.trace:
        fault_wire, fault_type = args.trace
        frames = args.frames
        if frames == 0:
            d_seq, _ = compute_sequential_depth(circuit)
            frames = max(2, d_seq + 1)

        if args.algorithm == "9val":
            from atpg_9val import NineValueAlgorithm
            engine = NineValueAlgorithm(circuit, num_frames=frames, backtrack_limit=50)
        else:
            from atpg_ext_d import ExtendedDAlgorithm
            engine = ExtendedDAlgorithm(circuit, num_frames=frames, backtrack_limit=50)

        recorder = TraceRecorder(circuit, frames, fault_wire, fault_type)
        engine._trace_recorder = recorder
        result = engine.generate_test(fault_wire, fault_type)

        p3 = render_atpg_trace_html(recorder,
                                    out_dir / f"{bench_name}_trace_{fault_wire}_{fault_type}.html")
        print(f"  Wrote {p3}  ({len(recorder.snapshots)} steps)")
        print(f"  ATPG result: {result}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    _cli()
