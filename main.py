"""
main.py — Sequential ATPG Design Space Exploration Tool
=========================================================
Entry point for running both ATPG algorithms and producing
comparison metrics.

Modes:
  Mode A: Extended D-Algorithm (5-valued logic)
  Mode B: 9-Valued Logic Algorithm (Muth's)

Usage:
    python main.py                                    # s27, both modes
    python main.py --bench benchmarks/s298.v          # s298, both modes
    python main.py --mode ext_d                       # Mode A only
    python main.py --mode 9val                        # Mode B only
    python main.py --explore                          # Full DSE sweep
    python main.py --bench benchmarks/s27.v --verbose # Per-fault output
"""

import argparse
import csv
import os
import sys
import time
import tracemalloc
import itertools
from collections import defaultdict

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parser import parse_verilog
from sgraph import compute_sequential_depth, print_sgraph_report, build_sgraph
from atpg_ext_d import ExtendedDAlgorithm
from atpg_ext_d import ATPGResult as ExtDResult
from atpg_ext_d import generate_fault_list
from atpg_9val import NineValueAlgorithm
from atpg_9val import ATPGResult as NineValResult
from fault_sim import FaultSimulator


# ── Single Algorithm Run ─────────────────────────────────────────────────────

def run_algorithm(engine, engine_name, faults, verbose=False):
    """Run a single ATPG engine on all faults, return metrics dict."""
    stats = {
        "detected": 0, "undetectable": 0, "aborted": 0,
        "init_failure": 0, "total_bt": 0, "total_time": 0.0,
    }
    test_vectors = []

    for i, (fw, ft) in enumerate(faults):
        result = engine.generate_test(fw, ft)
        stats["total_bt"]   += result.backtracks
        stats["total_time"] += result.time_s

        if result.status == "DETECTED":
            stats["detected"] += 1
            if result.test_vector:
                test_vectors.append(result.test_vector)
            if verbose:
                print(f"    ✓ {fw}/{ft}  bt={result.backtracks}")
        elif result.status == "INIT_FAILURE":
            stats["init_failure"] += 1
            if verbose:
                print(f"    ⚠ {fw}/{ft}  INIT_FAILURE (5-val limitation)")
        elif result.status == "UNDETECTABLE":
            stats["undetectable"] += 1
            if verbose:
                print(f"    ✗ {fw}/{ft}  UNDETECTABLE")
        else:
            stats["aborted"] += 1
            if verbose:
                print(f"    ? {fw}/{ft}  ABORTED")

        if (i + 1) % max(1, len(faults) // 5) == 0:
            pct = 100 * (i + 1) / len(faults)
            print(f"    [{engine_name}] {pct:5.1f}% — detected so far: {stats['detected']}")

    total = len(faults)
    stats["coverage"] = 100.0 * stats["detected"] / total if total else 0.0
    stats["test_vectors"] = test_vectors
    stats["total_faults"] = total

    return stats


# ── Comparison Run ───────────────────────────────────────────────────────────

def run_comparison(bench_path, num_frames, bt_limit, verbose=False):
    """Run both Mode A and Mode B on the same benchmark and compare."""
    print("\n" + "=" * 70)
    print("  Sequential ATPG — Design Space Exploration")
    print("=" * 70)
    print(f"  Benchmark     : {bench_path}")
    print(f"  Time frames   : {num_frames}")
    print(f"  BT limit      : {bt_limit}")
    print("=" * 70)

    # Parse circuit
    circuit = parse_verilog(bench_path)
    circuit.summary()

    # S-Graph and sequential depth
    d_seq = print_sgraph_report(circuit)

    # Fault list
    faults = generate_fault_list(circuit)
    print(f"\n  Fault list size: {len(faults)} (SA0 + SA1 on every wire)")

    # Complexity estimate
    n_pi = len(circuit.primary_inputs)
    n_ff = len(circuit.flip_flops)
    print(f"\n  Complexity estimate (State & Search Space bounds):")
    print(f"    n (PIs) = {n_pi}, f (FFs) = {n_ff}, k (frames) = {num_frames}")
    print(f"    Physical Boolean Space         : O(2^(n×k) × 2^f) = O({(2**(n_pi * num_frames)) * (2**n_ff)})")
    print(f"    ExtD 5-Valued Space (k frames) : O(2^(n×k) × 5^f) = O({(2**(n_pi * num_frames)) * (5**n_ff)})")
    print(f"    9Val 9-Valued Space (k frames) : O(2^(n×k) × 9^f) = O({(2**(n_pi * num_frames)) * (9**n_ff)})")

    results = {}

    # ── Mode A: Extended D-Algorithm (5-valued) ──────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Mode A: Extended D-Algorithm (5-Valued Logic)")
    print(f"{'─' * 70}")

    tracemalloc.start()
    t0 = time.time()
    engine_a = ExtendedDAlgorithm(circuit, num_frames=num_frames,
                                   backtrack_limit=bt_limit)
    stats_a = run_algorithm(engine_a, "ExtD", faults, verbose)
    mem_a = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_a = time.time() - t0

    stats_a["wall_time"] = wall_a
    stats_a["peak_mem_mb"] = mem_a[1] / (1024 * 1024)
    results["ext_d"] = stats_a

    print(f"\n  Mode A Results:")
    print(f"    Detected       : {stats_a['detected']}")
    print(f"    Init Failures  : {stats_a['init_failure']}  ← 5-val limitation")
    print(f"    Undetectable   : {stats_a['undetectable']}")
    print(f"    Aborted        : {stats_a['aborted']}")
    print(f"    Coverage       : {stats_a['coverage']:.2f}%")
    print(f"    Wall time      : {wall_a:.3f}s")
    print(f"    Peak memory    : {stats_a['peak_mem_mb']:.2f} MB")
    print(f"    Total BTs      : {stats_a['total_bt']}")

    # ── Mode B: 9-Valued Logic (Muth's) ──────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Mode B: 9-Valued Logic Algorithm (Muth's)")
    print(f"{'─' * 70}")

    tracemalloc.start()
    t0 = time.time()
    engine_b = NineValueAlgorithm(circuit, num_frames=num_frames,
                                    backtrack_limit=bt_limit)
    stats_b = run_algorithm(engine_b, "9Val", faults, verbose)
    mem_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_b = time.time() - t0

    stats_b["wall_time"] = wall_b
    stats_b["peak_mem_mb"] = mem_b[1] / (1024 * 1024)
    results["9val"] = stats_b

    print(f"\n  Mode B Results:")
    print(f"    Detected       : {stats_b['detected']}")
    print(f"    Undetectable   : {stats_b['undetectable']}")
    print(f"    Aborted        : {stats_b['aborted']}")
    print(f"    Coverage       : {stats_b['coverage']:.2f}%")
    print(f"    Wall time      : {wall_b:.3f}s")
    print(f"    Peak memory    : {stats_b['peak_mem_mb']:.2f} MB")
    print(f"    Total BTs      : {stats_b['total_bt']}")

    # ── Fault Simulation Verification ────────────────────────────────────
    sim = FaultSimulator(circuit)

    print(f"\n{'─' * 70}")
    print(f"  Fault Simulation Verification")
    print(f"{'─' * 70}")

    if stats_a["test_vectors"]:
        print("\n  Verifying Mode A vectors:")
        verified_a = sim.simulate(stats_a["test_vectors"], faults)
        sim.report(verified_a, faults)
        results["ext_d"]["verified_coverage"] = 100.0 * len(verified_a) / len(faults) if faults else 0
    else:
        print("\n  Mode A: No test vectors to verify.")
        results["ext_d"]["verified_coverage"] = 0.0

    if stats_b["test_vectors"]:
        print("\n  Verifying Mode B vectors:")
        verified_b = sim.simulate(stats_b["test_vectors"], faults)
        sim.report(verified_b, faults)
        results["9val"]["verified_coverage"] = 100.0 * len(verified_b) / len(faults) if faults else 0
    else:
        print("\n  Mode B: No test vectors to verify.")
        results["9val"]["verified_coverage"] = 0.0

    # ── Side-by-side comparison ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'Mode A (ExtD)':<20} {'Mode B (9Val)':<20}")
    print(f"  {'─' * 65}")
    print(f"  {'Logic Values':<25} {'5 (0,1,X,D,D̄)':<20} {'9 (Muth pairs)':<20}")
    print(f"  {'Detected':<25} {stats_a['detected']:<20} {stats_b['detected']:<20}")
    print(f"  {'Init Failures':<25} {stats_a['init_failure']:<20} {'N/A':<20}")
    print(f"  {'Undetectable':<25} {stats_a['undetectable']:<20} {stats_b['undetectable']:<20}")
    print(f"  {'Aborted':<25} {stats_a['aborted']:<20} {stats_b['aborted']:<20}")
    print(f"  {'Coverage %':<25} {stats_a['coverage']:<20.2f} {stats_b['coverage']:<20.2f}")
    print(f"  {'Verified Coverage %':<25} {results['ext_d']['verified_coverage']:<20.2f} {results['9val']['verified_coverage']:<20.2f}")
    print(f"  {'Wall Time (s)':<25} {wall_a:<20.3f} {wall_b:<20.3f}")
    print(f"  {'Peak Memory (MB)':<25} {stats_a['peak_mem_mb']:<20.2f} {stats_b['peak_mem_mb']:<20.2f}")
    print(f"  {'Total Backtracks':<25} {stats_a['total_bt']:<20} {stats_b['total_bt']:<20}")
    print(f"  {'Test Vectors':<25} {len(stats_a['test_vectors']):<20} {len(stats_b['test_vectors']):<20}")
    print(f"{'=' * 70}")

    return results


# ── Design Space Exploration Sweep ───────────────────────────────────────────

def design_space_sweep(bench_paths, frames_list, bt_limits,
                        out_csv="results/design_space.csv"):
    """Sweep over benchmarks × frames × bt_limits for both algorithms."""
    all_results = []
    configs = list(itertools.product(bench_paths, frames_list, bt_limits))
    total = len(configs)

    print(f"\n{'=' * 70}")
    print(f"  DESIGN SPACE EXPLORATION")
    print(f"{'=' * 70}")
    print(f"  Benchmarks  : {[os.path.basename(b) for b in bench_paths]}")
    print(f"  Frames      : {frames_list}")
    print(f"  BT limits   : {bt_limits}")
    print(f"  Total configs: {total} × 2 algorithms = {total * 2} runs")
    print(f"{'=' * 70}\n")

    for i, (bench, nf, btl) in enumerate(configs, 1):
        bench_name = os.path.basename(bench).replace(".v", "")
        print(f"\n[{i}/{total}] {bench_name}  frames={nf}  bt={btl}")

        circuit = parse_verilog(bench)
        faults = generate_fault_list(circuit)
        d_seq, _ = compute_sequential_depth(circuit)

        for mode_name, EngineClass, ResultClass in [
            ("ExtD_5val", ExtendedDAlgorithm, ExtDResult),
            ("9Val_Muth", NineValueAlgorithm, NineValResult),
        ]:
            print(f"  Running {mode_name}...", end=" ", flush=True)

            tracemalloc.start()
            t0 = time.time()
            engine = EngineClass(circuit, num_frames=nf, backtrack_limit=btl)
            stats = run_algorithm(engine, mode_name, faults, verbose=False)
            wall = time.time() - t0
            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Fault simulation verification
            sim = FaultSimulator(circuit)
            if stats["test_vectors"]:
                verified = sim.simulate(stats["test_vectors"], faults)
                v_cov = 100.0 * len(verified) / len(faults) if faults else 0
            else:
                v_cov = 0.0

            row = {
                "benchmark":       bench_name,
                "algorithm":       mode_name,
                "n_PIs":           len(circuit.primary_inputs),
                "n_FFs":           len(circuit.flip_flops),
                "d_seq":           d_seq,
                "num_frames":      nf,
                "bt_limit":        btl,
                "total_faults":    len(faults),
                "detected":        stats["detected"],
                "init_failures":   stats.get("init_failure", 0),
                "undetectable":    stats["undetectable"],
                "aborted":         stats["aborted"],
                "coverage_%":      round(stats["coverage"], 2),
                "verified_cov_%":  round(v_cov, 2),
                "test_vectors":    len(stats["test_vectors"]),
                "total_bt":        stats["total_bt"],
                "wall_time_s":     round(wall, 4),
                "peak_mem_MB":     round(mem[1] / (1024 * 1024), 3),
            }
            all_results.append(row)
            print(f"coverage={row['coverage_%']}%  time={row['wall_time_s']}s")

    # Print results table
    print_results_table(all_results)

    # Analysis
    analyze_results(all_results)

    # Write CSV
    write_csv(all_results, out_csv)

    return all_results


def print_results_table(results):
    if not results:
        print("No results.")
        return

    cols = [
        ("benchmark",    10), ("algorithm",   12), ("num_frames",  6),
        ("bt_limit",      6), ("coverage_%", 10), ("verified_cov_%", 12),
        ("init_failures", 10), ("test_vectors", 8),
        ("total_bt",     10), ("wall_time_s", 10), ("peak_mem_MB", 10),
    ]

    print(f"\n{'=' * 110}")
    header = "  ".join(k.ljust(w) for k, w in cols)
    print(header)
    print("-" * 110)
    for r in results:
        row = "  ".join(str(r.get(k, "")).ljust(w) for k, w in cols)
        print(row)
    print("=" * 110)


def analyze_results(results):
    if not results:
        return

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS")
    print(f"{'=' * 70}")

    # Compare algorithms
    by_algo = defaultdict(list)
    for r in results:
        by_algo[r["algorithm"]].append(r)

    print("\n  === Algorithm Comparison ===")
    for algo, runs in sorted(by_algo.items()):
        coverages = [r["coverage_%"] for r in runs]
        times     = [r["wall_time_s"] for r in runs]
        init_f    = sum(r.get("init_failures", 0) for r in runs)
        avg_cov   = sum(coverages) / len(coverages)
        avg_time  = sum(times) / len(times)
        print(f"  {algo}:")
        print(f"    Avg coverage   = {avg_cov:.2f}%")
        print(f"    Avg wall time  = {avg_time:.3f}s")
        print(f"    Total init failures = {init_f}")

    # Scalability comparison
    by_bench = defaultdict(list)
    for r in results:
        by_bench[r["benchmark"]].append(r)

    print("\n  === Scalability (s27 vs s298) ===")
    for bench, runs in sorted(by_bench.items()):
        best = max(runs, key=lambda x: (x["coverage_%"], -x["wall_time_s"]))
        print(f"  {bench}: best coverage={best['coverage_%']}%  "
              f"(algo={best['algorithm']}, frames={best['num_frames']}, "
              f"bt={best['bt_limit']}, time={best['wall_time_s']}s)")

    # Effect of time frames
    print("\n  === Effect of num_frames on Coverage ===")
    by_frames = defaultdict(list)
    for r in results:
        by_frames[(r["algorithm"], r["num_frames"])].append(r["coverage_%"])
    for (algo, nf), covs in sorted(by_frames.items()):
        avg = sum(covs) / len(covs)
        print(f"    {algo} frames={nf}: avg_coverage={avg:.2f}%")

    print("=" * 70)


def write_csv(results, out_path):
    if not results:
        return
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequential ATPG — Design Space Exploration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # s27, both modes
  python main.py --bench benchmarks/s298.v          # s298
  python main.py --mode ext_d                       # Mode A only
  python main.py --mode 9val                        # Mode B only
  python main.py --explore                          # Full design space sweep
  python main.py --verbose                          # Per-fault output
        """
    )
    parser.add_argument("--bench", default="benchmarks/s27.v",
                        help="Path to benchmark Verilog file")
    parser.add_argument("--mode", choices=["ext_d", "9val", "both"],
                        default="both",
                        help="Algorithm mode: ext_d, 9val, or both (default)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Time frames (0 = auto from d_seq+1)")
    parser.add_argument("--bt", type=int, default=50,
                        help="Backtrack limit per fault (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-fault results")
    parser.add_argument("--explore", action="store_true",
                        help="Run full design space exploration sweep")
    args = parser.parse_args()

    if args.explore:
        benches = ["benchmarks/s27.v"]
        if os.path.isfile("benchmarks/s298.v"):
            benches.append("benchmarks/s298.v")

        # Auto-compute frame lists based on sequential depth of each benchmark
        max_d_seq = 1
        for b in benches:
            c = parse_verilog(b)
            d, _ = compute_sequential_depth(c)
            if d > max_d_seq:
                max_d_seq = d
        auto_frames = sorted(set([2, max_d_seq + 1, max_d_seq + 2]))

        design_space_sweep(
            bench_paths=benches,
            frames_list=auto_frames,
            bt_limits=[10, 50],
            out_csv="results/design_space.csv"
        )
        return

    if not os.path.isfile(args.bench):
        print(f"Error: benchmark not found — {args.bench}")
        sys.exit(1)

    # Determine time frames
    if args.frames == 0:
        circuit = parse_verilog(args.bench)
        d_seq, _ = compute_sequential_depth(circuit)
        num_frames = d_seq + 1
        if num_frames < 2:
            num_frames = 2
        print(f"  Auto frames: d_seq={d_seq} → {num_frames} time frames")
    else:
        num_frames = args.frames

    if args.mode == "both":
        run_comparison(args.bench, num_frames, args.bt, args.verbose)
    elif args.mode == "ext_d":
        circuit = parse_verilog(args.bench)
        circuit.summary()
        print_sgraph_report(circuit)
        faults = generate_fault_list(circuit)
        engine = ExtendedDAlgorithm(circuit, num_frames=num_frames,
                                     backtrack_limit=args.bt)
        stats = run_algorithm(engine, "ExtD", faults, args.verbose)
        print(f"\nCoverage: {stats['coverage']:.2f}%")
    elif args.mode == "9val":
        circuit = parse_verilog(args.bench)
        circuit.summary()
        print_sgraph_report(circuit)
        faults = generate_fault_list(circuit)
        engine = NineValueAlgorithm(circuit, num_frames=num_frames,
                                      backtrack_limit=args.bt)
        stats = run_algorithm(engine, "9Val", faults, args.verbose)
        print(f"\nCoverage: {stats['coverage']:.2f}%")


if __name__ == "__main__":
    main()
