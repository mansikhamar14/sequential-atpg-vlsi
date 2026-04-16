# Sequential ATPG Project: Team Work Distribution (10 Members)

This document formally divides the Sequential ATPG implementation, experimental methodology, and reporting workload across 10 team members. The squads are divided exactly into groups of 3, 3, and 4.

This distribution groups members by structural objective: **Circuit Foundation + Validation**, **Mode A (5-Val) + Time Scaling**, and **Mode B (9-Val) + Integration**.

---

## Group 1 (3 members) • Circuit foundation + Validation
**Focus:** Establishing the internal representations, parsing Verilog, assessing memory geometries, and maintaining the standalone validation testing layer.

*   **Member 1 (M1): `parser.py`**
    *   *Role:* Verilog $\rightarrow$ Circuit graph
    *   *Implementation Details:* Constructing the `Circuit` and `Gate` data structures, performing regex string evaluations on netlists, and executing Kahn's BFS algorithm for topological node combinations.
*   **Member 2 (M2): `sgraph.py`**
    *   *Role:* FF dependency DAG, $d_{seq}$
    *   *Implementation Details:* Developing the loop extraction sequence measuring flip-flop mappings to correctly extract the geometric `Sequential Depth` mapping required for algorithms.
*   **Member 3 (M3): `fault_sim.py`**
    *   *Role:* Fault simulation & verification
    *   *Implementation Details:* Writing the true-boolean validation structure mapping real 0/1 logic sequences. Providing the independent benchmark required to prove whether downstream ATPG engines suffer from false-positive `X` state dependencies.

---

## Group 2 (3 members) • Mode A — 5-val ATPG + Time Scaling
**Focus:** Writing the industry-standard 5-valued DFS backtracing matrix and managing the geometric unrolling logic that explicitly fuels its search trees.

*   **Member 4 (M4): `logic5val.py`**
    *   *Role:* 5-valued truth tables
    *   *Implementation Details:* Writing rigid dictionary structures evaluating math vectors covering `0`, `1`, `X`, `D`, and `D_BAR`.
*   **Member 5 (M5): `atpg_ext_d.py`**
    *   *Role:* Extended D-Algorithm engine
    *   *Implementation Details:* Building the DFS core looping through `_imply`, tracking `_d_frontier` constraints, and resolving PI conflicts via `_backtrace` bounds.
*   **Member 6 (M6): `timeframe.py`**
    *   *Role:* Unrolled circuit across frames
    *   *Implementation Details:* Cloning combinatorial boundaries directly and resolving $D \rightarrow Q$ hooks recursively across bounds effectively producing Pseudo-Primary Input (PPI) sets perfectly tuned for the ATPG DFS engine.

---

## Group 3 (4 members) • Mode B — 9-val ATPG + integration
**Focus:** Developing Muth's decoupled state models mapping 9-values, and bridging the entire project via visual UI integration and Design Space sweeps.

*   **Member 7 (M7): `logic9val.py`**
    *   *Role:* 9-valued logic engine
    *   *Implementation Details:* Splitting representations into decoupled `(good/faulty)` arrays cleanly navigating `merge_9val` topologies without overlapping false positives.
*   **Member 8 (M8): `atpg_9val.py`**
    *   *Role:* Muth's ATPG algorithm
    *   *Implementation Details:* Cloning the Mode A framework natively but adapting bounding strategies and implication mappings explicitly targeting the more complex 9-valued array systems.
*   **Member 9 (M9): `main.py`**
    *   *Role:* CLI, DSE sweep, results analysis
    *   *Implementation Details:* Architecting OS environment loops triggering exhaustive permutations storing metrics cleanly across CSV layers identifying parameter relationships (Backtrack-Limits vs Time-Frames).
*   **Member 10 (M10): `visualize.py`**
    *   *Role:* Interactive HTML circuit viewer
    *   *Implementation Details:* Anchoring PyVis topologies hooking `TraceRecorder` nodes across the DFS solver dynamically producing web-hosted debug geometries plotting objectives.
