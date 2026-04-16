"""
Time Frame Expansion — Sequential Circuit Unroller
===================================================
Unrolls a sequential circuit with DFFs into k combinational copies.
"""

import copy                                      # Imports Python's object duplication library (good practice for deep copying)
from dataclasses import dataclass, field         # Allows us to create clean, structured data containers easily
from typing import List, Dict, Tuple, Set        # Type hints to make the code easier to read and debug
from parser import Circuit, Gate                 # Imports the Circuit and Gate structures from your custom parser.py


@dataclass                                       # This decorator automatically generates __init__ and other class methods behind the scenes
class UnrolledCircuit:                           # A class acting as a container for the newly flattened, purely combinational circuit
    
    num_frames: int                              # Stores the total number of time frames (clock cycles) we are unrolling into
    original_name: str                           # Stores the name of the original Verilog module (e.g., "s27")

    # All gates across all frames: (frame_index, original_gate_name) → Gate
    gates: Dict[Tuple[int, str], Gate] = field(default_factory=dict)      # A dictionary to hold all duplicated gates. Keyed by (time_frame, gate_name)

    # All wires: wire_key → value (initial "X")
    wires: Dict[str, str] = field(default_factory=dict)                   # A dictionary holding the logic state (0, 1, or X) of every wire in every frame

    # PI/PO in the unrolled circuit (wire keys)
    primary_inputs: List[str] = field(default_factory=list)               # A list tracking all primary inputs (PIs) duplicated across all time frames
    primary_outputs: List[str] = field(default_factory=list)              # A list tracking all primary outputs (POs) duplicated across all time frames

    # Pseudo-primary inputs (DFF Q outputs in earliest frame)
    pseudo_primary_inputs: List[str] = field(default_factory=list)        # A list to track the unknown initial states of flip-flops in the very first frame

    # Wire-to-driver map
    fanin: Dict[str, object] = field(default_factory=dict)                # Maps a specific wire to the gate that drives it (helps ATPG trace backward)
    fanout: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)# Maps a specific wire to all the gates it feeds into (helps ATPG trace forward)

    def wire_key(self, wire_name, frame):                                 # Helper method to create a unique string ID for a wire in a specific frame
        return f"{wire_name}@{frame}"                                     # E.g., The "reset" wire in frame -1 becomes the string "reset@-1"

    def get_wire(self, wire_name, frame):                                 # Helper method to safely look up a wire's logic value
        return self.wires.get(self.wire_key(wire_name, frame), "X")       # If the wire hasn't been set yet, it safely returns "X" (unknown)

    def set_wire(self, wire_name, frame, value):                          # Helper method to update a wire's logic value
        self.wires[self.wire_key(wire_name, frame)] = value               # Assigns the given value (0, 1, D, etc.) to that specific wire key

    def summary(self):                                                    # A debugging method to print out statistics about the unrolled circuit
        print(f"\n  Unrolled Circuit: {self.original_name}")              # Prints the module's name
        print(f"    Frames      : {self.num_frames}")                     # Prints the number of frames it was expanded into
        print(f"    Total gates : {len(self.gates)}")                     # Prints the massive total of all duplicated gates
        print(f"    Total wires : {len(self.wires)}")                     # Prints the massive total of all duplicated wires
        print(f"    PIs (real)  : {len(self.primary_inputs)}")            # Prints the count of normal inputs across all frames
        print(f"    POs (real)  : {len(self.primary_outputs)}")           # Prints the count of normal outputs across all frames
        print(f"    PPIs        : {len(self.pseudo_primary_inputs)}")     # Prints the count of Pseudo-Primary Inputs (FF initial states)


# ── Unroller ─────────────────────────────────────────────────────────────────

def unroll(circuit: Circuit, num_frames: int) -> UnrolledCircuit:         # Main Logic: Takes a parsed circuit and frame count, returns the UnrolledCircuit
    """Unroll `circuit` into `num_frames` time-frame copies."""
    assert num_frames >= 1                                                # Safety check: You can't unroll into 0 or negative frames

    uc = UnrolledCircuit(                                                 # Initialize our new, empty UnrolledCircuit container
        num_frames=num_frames,                                            # Pass in the requested number of frames
        original_name=circuit.module_name,                                # Pass in the original circuit's name
    )

    frames = list(range(-(num_frames - 1), 1))                            # Generate the frame IDs. E.g., if num_frames is 3, this makes [-2, -1, 0]
    earliest = frames[0]                                                  # Identify the very first frame in the list (e.g., -2)

    # ── Build combinational gates for each frame ─────────────────────────
    for t in frames:                                                      # Loop through every single time frame (e.g., first -2, then -1, then 0)
        for gname, gate in circuit.gates.items():                         # Within that frame, loop through every single gate in the original circuit
            if gate.gate_type == "dff":                                   # If the gate is a flip-flop (memory)...
                continue                                                  # ...SKIP IT! We only duplicate combinational (stateless) logic right now

            new_gate = Gate(                                              # Create a brand new Gate object for this specific time frame
                name=gname,                                               # Keep the original gate's name
                gate_type=gate.gate_type,                                 # Keep the original gate's type (AND, OR, NAND, etc.)
                inputs=[uc.wire_key(iw, t) for iw in gate.inputs],        # Append the frame ID to all of its input wires (e.g., 'inA' -> 'inA@-1')
                output=uc.wire_key(gate.output, t),                       # Append the frame ID to its output wire (e.g., 'outZ' -> 'outZ@-1')
                time_frame=t,                                             # Save the current time frame integer inside the gate object
            )
            uc.gates[(t, gname)] = new_gate                               # Save this new gate into the UnrolledCircuit's dictionary using (frame, name) as the key

            uc.wires.setdefault(new_gate.output, "X")                     # Create the output wire in the wires dictionary, defaulting its logic to "X"
            uc.fanin[new_gate.output] = (t, gname)                        # Update fanin: Record that THIS specific gate drives THIS specific output wire

            for wk in new_gate.inputs:                                    # Loop through all the renamed input wires for this gate
                uc.wires.setdefault(wk, "X")                              # Create them in the dictionary if they don't exist, defaulting to "X"
                uc.fanout.setdefault(wk, []).append((t, gname))           # Update fanout: Record that this wire feeds INTO this specific gate

    # ── Primary inputs — appear in every frame ───────────────────────────
    for t in frames:                                                      # Loop through all time frames again
        for pi in circuit.primary_inputs:                                 # Loop through the original Primary Inputs (the physical pins on the chip)
            wk = uc.wire_key(pi, t)                                       # Create a frame-specific name for the PI
            uc.wires.setdefault(wk, "X")                                  # Add it to the wires dictionary, defaulting to "X"
            uc.primary_inputs.append(wk)                                  # Add it to the UnrolledCircuit's master list of PIs

    # ── Primary outputs — from ALL frames ────────────────────────────────
    for t in frames:                                                      # Loop through all time frames again
        for po in circuit.primary_outputs:                                # Loop through the original Primary Outputs
            wk = uc.wire_key(po, t)                                       # Create a frame-specific name for the PO
            uc.wires.setdefault(wk, "X")                                  # Add it to the wires dictionary, defaulting to "X"
            uc.primary_outputs.append(wk)                                 # Add it to the UnrolledCircuit's master list of POs

    # ── DFF inter-frame connections ───────────────────────────────────────
    for ff_gname in circuit.flip_flops:                                   # Now we handle the skipped flip-flops. Loop through all original DFFs
        ff_gate = circuit.gates[ff_gname]                                 # Get the original DFF gate object
        d_wire = ff_gate.inputs[0]                                        # Identify the wire feeding INTO the flip-flop (D)
        q_wire = ff_gate.output                                           # Identify the wire coming OUT of the flip-flop (Q)

        for idx, t in enumerate(frames):                                  # Loop through the frames, keeping track of the loop index (`idx`)
            q_wk = uc.wire_key(q_wire, t)                                 # Get the frame-specific name for the Q output
            uc.wires.setdefault(q_wk, "X")                                # Ensure the Q wire exists in the dictionary, defaulting to "X"

            if t == earliest:                                             # IF this is the very first (earliest) time frame...
                uc.pseudo_primary_inputs.append(q_wk)                     # ...there is no previous cycle to drive it! Treat it as a Pseudo-Primary Input
            else:                                                         # IF this is any frame AFTER the first frame...
                prev_t = frames[idx - 1]                                  # Calculate the frame number of the PREVIOUS clock cycle
                d_wk = uc.wire_key(d_wire, prev_t)                        # Get the name of the D input wire from that PREVIOUS cycle
                uc.wires.setdefault(d_wk, "X")                            # Ensure the previous D wire exists
                uc.fanin[q_wk] = ("DFF_CONNECT", d_wk)                    # THE CRITICAL STITCH: The current Q output is mathematically driven by the previous D input!

            d_wk = uc.wire_key(d_wire, t)                                 # Finally, get the name of the D input for the CURRENT cycle
            uc.wires.setdefault(d_wk, "X")                                # Ensure it exists in the dictionary

    return uc                                                             # The sequential circuit is now purely combinational. Return it!


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":                                                # A Python safeguard: only run the code below if executing this file directly
    import sys                                                            # Import sys to read terminal arguments
    from parser import parse_verilog                                      # Import our Verilog parser

    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/s27.v"       # If the user typed a file path in the terminal, use it. Otherwise default to s27.v
    nf   = int(sys.argv[2]) if len(sys.argv) > 2 else 3                   # If the user typed a frame count, use it. Otherwise default to 3 frames

    circ = parse_verilog(path)                                            # Read and parse the target Verilog file into a Circuit object
    circ.summary()                                                        # Print the standard circuit statistics
    
    uc = unroll(circ, nf)                                                 # Call our unroll logic to flatten the circuit
    uc.summary()                                                          # Print the new, much larger unrolled circuit statistics
    
    print(f"\n  PPIs: {uc.pseudo_primary_inputs}")                        # Print out the exact wire names identified as unknown initial states
    print(f"  POs:  {uc.primary_outputs}")                                # Print out the exact wire names identified as primary outputs