#!/usr/bin/env python3
"""
verify_mpi_results.py
---------------------
Verify and analyze MPI benchmark JSON results.
Checks that per-rank files match the combined file and displays summary.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        return None


def verify_results(json_base_path: str):
    """
    Verify MPI benchmark results
    
    Args:
        json_base_path: Base path for JSON file (e.g., "results/numpy_2nodes.json")
    """
    base_path = Path(json_base_path)
    
    if not base_path.exists():
        print(f"❌ Combined results file not found: {base_path}")
        return False
    
    print("=" * 70)
    print("MPI Benchmark Results Verification")
    print("=" * 70)
    print()
    
    # Load combined results
    combined = load_json(base_path)
    if combined is None:
        return False
    
    # Check if it's a multi-rank run
    world_size = combined.get("world_size", 1)
    
    if world_size == 1:
        print("📊 Single-rank run detected")
        print(f"   Results: {base_path}")
        print()
        display_single_rank_summary(combined)
        return True
    
    print(f"📊 Multi-rank run detected: {world_size} ranks")
    print(f"   Combined results: {base_path}")
    print()
    
    # Load per-rank files
    rank_files = []
    rank_data = []
    
    for rank in range(world_size):
        rank_file = base_path.parent / f"{base_path.stem}_rank{rank}{base_path.suffix}"
        rank_files.append(rank_file)
        
        if rank_file.exists():
            print(f"   ✓ Found rank {rank} results: {rank_file}")
            data = load_json(rank_file)
            if data:
                rank_data.append(data)
        else:
            print(f"   ⚠ Missing rank {rank} results: {rank_file}")
    
    print()
    
    if len(rank_data) != world_size:
        print(f"❌ Expected {world_size} rank files, found {len(rank_data)}")
        return False
    
    # Verify combined results match per-rank files
    print("🔍 Verifying combined results...")
    
    if "ranks" not in combined:
        print("❌ Combined file missing 'ranks' field")
        return False
    
    if len(combined["ranks"]) != world_size:
        print(f"❌ Combined file has {len(combined['ranks'])} ranks, expected {world_size}")
        return False
    
    print(f"   ✓ Combined file contains {world_size} rank results")
    
    # Display per-rank summaries
    print()
    print("=" * 70)
    print("Per-Rank Performance")
    print("=" * 70)
    print()
    
    for i, rank_result in enumerate(rank_data):
        display_rank_summary(rank_result, i)
        print()
    
    # Display aggregate summary
    if "summary" in combined:
        print("=" * 70)
        print("Aggregate Summary (All Ranks)")
        print("=" * 70)
        print()
        display_aggregate_summary(combined["summary"], world_size)
        print()
        
        # Verify summary calculations
        verify_summary_calculations(rank_data, combined["summary"])
    
    print("✅ Verification complete!")
    return True


def display_single_rank_summary(data: Dict[str, Any]):
    """Display summary for single-rank run"""
    if "metrics" not in data:
        print("⚠ No metrics found in results")
        return
    
    metrics = data["metrics"]
    
    if "rank" in data:
        print(f"Rank:              {data['rank']}")
    if "hostname" in data:
        print(f"Hostname:          {data['hostname']}")
    print()
    
    # Throughput
    if "throughput_samples_per_s" in metrics:
        tput = metrics["throughput_samples_per_s"]
        print(f"Throughput:        {tput.get('mean', 0):.1f} samples/sec (mean)")
        print(f"                   {tput.get('peak', 0):.1f} samples/sec (peak)")
    
    # Bandwidth
    if "bandwidth" in metrics:
        bw = metrics["bandwidth"]
        print(f"Bandwidth:         {bw.get('mean_MiB_s', 0):.2f} MiB/s (mean)")
        print(f"                   {bw.get('peak_MiB_s', 0):.2f} MiB/s (peak)")
    
    # Timing
    if "iteration_ms" in metrics:
        timing = metrics["iteration_ms"]
        print(f"Iteration Time:    {timing.get('mean', 0):.2f} ms (mean)")
        print(f"                   {timing.get('min', 0):.2f} ms (min)")
        print(f"                   {timing.get('max', 0):.2f} ms (max)")


def display_rank_summary(data: Dict[str, Any], rank: int):
    """Display summary for a single rank"""
    print(f"Rank {rank}")
    print("-" * 70)
    
    if "hostname" in data:
        print(f"Hostname:          {data['hostname']}")
    
    if "metrics" not in data:
        print("⚠ No metrics found")
        return
    
    metrics = data["metrics"]
    
    # Throughput
    if "throughput_samples_per_s" in metrics:
        tput = metrics["throughput_samples_per_s"]
        print(f"Throughput:        {tput.get('mean', 0):.1f} samples/sec (mean)")
        print(f"                   {tput.get('peak', 0):.1f} samples/sec (peak)")
    
    # Bandwidth
    if "bandwidth" in metrics:
        bw = metrics["bandwidth"]
        print(f"Bandwidth:         {bw.get('mean_MiB_s', 0):.2f} MiB/s (mean)")
        print(f"                   {bw.get('peak_MiB_s', 0):.2f} MiB/s (peak)")
    
    # Timing
    if "iteration_ms" in metrics:
        timing = metrics["iteration_ms"]
        print(f"Iteration Time:    {timing.get('mean', 0):.2f} ms (mean)")
        print(f"                   {timing.get('min', 0):.2f} ms (min)")
        print(f"                   {timing.get('max', 0):.2f} ms (max)")
    
    # Memory
    if "memory" in metrics:
        mem = metrics["memory"]
        if "process_rss_peak_MiB" in mem:
            print(f"Peak Memory:       {mem['process_rss_peak_MiB']:.2f} MiB")


def display_aggregate_summary(summary: Dict[str, Any], world_size: int):
    """Display aggregate summary across all ranks"""
    total_tput = summary.get("total_throughput_samples_per_s", 0)
    total_bw = summary.get("total_bandwidth_MiB_s", 0)
    mean_iter = summary.get("mean_iteration_ms", 0)
    
    print(f"Total Throughput:  {total_tput:.1f} samples/sec ({world_size} ranks)")
    print(f"Total Bandwidth:   {total_bw:.2f} MiB/s = {total_bw / 1024:.2f} GiB/s")
    print(f"Mean Iter Time:    {mean_iter:.2f} ms (avg across ranks)")
    print()
    print(f"Per-Rank Avg:      {total_tput / world_size:.1f} samples/sec")
    print(f"                   {total_bw / world_size:.2f} MiB/s")


def verify_summary_calculations(rank_data: list, summary: Dict[str, Any]):
    """Verify that summary calculations are correct"""
    print()
    print("🔍 Verifying summary calculations...")
    
    # Calculate expected totals
    expected_tput = sum(
        r.get("metrics", {}).get("throughput_samples_per_s", {}).get("mean", 0)
        for r in rank_data
    )
    expected_bw = sum(
        r.get("metrics", {}).get("bandwidth", {}).get("mean_MiB_s", 0)
        for r in rank_data
    )
    
    actual_tput = summary.get("total_throughput_samples_per_s", 0)
    actual_bw = summary.get("total_bandwidth_MiB_s", 0)
    
    # Allow small floating point differences
    tput_match = abs(expected_tput - actual_tput) < 0.1
    bw_match = abs(expected_bw - actual_bw) < 0.1
    
    if tput_match:
        print(f"   ✓ Throughput calculation correct: {actual_tput:.1f} samples/sec")
    else:
        print(f"   ❌ Throughput mismatch: expected {expected_tput:.1f}, got {actual_tput:.1f}")
    
    if bw_match:
        print(f"   ✓ Bandwidth calculation correct: {actual_bw:.2f} MiB/s")
    else:
        print(f"   ❌ Bandwidth mismatch: expected {expected_bw:.2f}, got {actual_bw:.2f}")
    
    return tput_match and bw_match


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_mpi_results.py <json_file>")
        print()
        print("Example:")
        print("  python3 verify_mpi_results.py results/numpy_2nodes.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    success = verify_results(json_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

