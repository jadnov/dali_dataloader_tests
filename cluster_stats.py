#!/usr/bin/env python3
"""
cluster_stats.py
----------------
Cluster-wide statistics aggregation for distributed DALI benchmarks.

Provides:
- Aggregation of metrics from multiple nodes
- Cluster-wide performance statistics
- Per-node performance breakdowns
- Load balancing analysis
- JSON export for cluster results
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """Metrics from a single node/rank"""
    rank: int
    hostname: str
    gpus: int
    samples_processed: int
    shard_count: int
    
    # Timing metrics
    iteration_times_ms: List[float]
    mean_iteration_ms: float
    std_iteration_ms: float
    min_iteration_ms: float
    max_iteration_ms: float
    median_iteration_ms: float
    
    # Performance metrics
    throughput_samples_per_s: float
    bandwidth_MiB_s: float
    bandwidth_MB_s: float
    
    # Memory metrics
    process_rss_mb: Optional[float] = None
    peak_gpu_allocated_mb: Optional[List[float]] = None


@dataclass
class ClusterMetrics:
    """Aggregated cluster-wide metrics"""
    # Cluster configuration
    total_nodes: int
    total_ranks: int
    total_gpus: int
    unique_hostnames: List[str]
    mpi_library: str
    
    # Aggregated performance
    total_samples_processed: int
    total_throughput_samples_per_s: float
    total_bandwidth_MiB_s: float
    total_bandwidth_MB_s: float
    
    # Cluster-wide timing statistics
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    
    # Load balance metrics
    load_balance_score: float  # 1.0 = perfect balance, lower = worse
    node_efficiency: float     # min_throughput / max_throughput
    throughput_cv: float       # Coefficient of variation for throughput
    
    # Per-node statistics
    node_mean_throughput: float
    node_std_throughput: float
    node_min_throughput: float
    node_max_throughput: float
    
    # Detailed per-node metrics
    per_node_metrics: List[NodeMetrics]


def aggregate_metrics(all_node_metrics: List[Dict[str, Any]], mpi_library: str = "Unknown") -> ClusterMetrics:
    """
    Aggregate metrics from all nodes into cluster-wide statistics.
    
    Args:
        all_node_metrics: List of metrics dictionaries from each rank
        mpi_library: Name of MPI library being used
        
    Returns:
        ClusterMetrics object with aggregated statistics
    """
    if not all_node_metrics:
        raise ValueError("No node metrics provided")
    
    logger.info(f"Aggregating metrics from {len(all_node_metrics)} nodes...")
    
    # Convert to NodeMetrics objects
    node_metrics_list = []
    for metrics in all_node_metrics:
        node_metrics = NodeMetrics(
            rank=metrics['rank'],
            hostname=metrics['hostname'],
            gpus=metrics['gpus'],
            samples_processed=metrics['samples_processed'],
            shard_count=metrics['shard_count'],
            iteration_times_ms=metrics['iteration_times_ms'],
            mean_iteration_ms=metrics['mean_iteration_ms'],
            std_iteration_ms=metrics['std_iteration_ms'],
            min_iteration_ms=metrics['min_iteration_ms'],
            max_iteration_ms=metrics['max_iteration_ms'],
            median_iteration_ms=metrics['median_iteration_ms'],
            throughput_samples_per_s=metrics['throughput_samples_per_s'],
            bandwidth_MiB_s=metrics['bandwidth_MiB_s'],
            bandwidth_MB_s=metrics['bandwidth_MB_s'],
            process_rss_mb=metrics.get('process_rss_mb'),
            peak_gpu_allocated_mb=metrics.get('peak_gpu_allocated_mb')
        )
        node_metrics_list.append(node_metrics)
    
    # Cluster configuration
    total_nodes = len(all_node_metrics)
    total_ranks = total_nodes
    total_gpus = sum(m.gpus for m in node_metrics_list)
    unique_hostnames = sorted(set(m.hostname for m in node_metrics_list))
    
    # Aggregate samples and performance
    total_samples = sum(m.samples_processed for m in node_metrics_list)
    total_throughput = sum(m.throughput_samples_per_s for m in node_metrics_list)
    total_bandwidth_MiB = sum(m.bandwidth_MiB_s for m in node_metrics_list)
    total_bandwidth_MB = sum(m.bandwidth_MB_s for m in node_metrics_list)
    
    # Aggregate all iteration times across all nodes
    all_iteration_times = []
    for m in node_metrics_list:
        all_iteration_times.extend(m.iteration_times_ms)
    
    times_array = np.array(all_iteration_times)
    mean_latency = float(np.mean(times_array))
    std_latency = float(np.std(times_array))
    min_latency = float(np.min(times_array))
    max_latency = float(np.max(times_array))
    median_latency = float(np.median(times_array))
    
    # Per-node throughput statistics
    throughputs = [m.throughput_samples_per_s for m in node_metrics_list]
    node_mean_throughput = float(np.mean(throughputs))
    node_std_throughput = float(np.std(throughputs))
    node_min_throughput = float(np.min(throughputs))
    node_max_throughput = float(np.max(throughputs))
    
    # Load balance metrics
    # Load balance score: How evenly distributed are the iteration times?
    # 1.0 = perfect (no variance), lower = worse balance
    load_balance_score = 1.0 - (std_latency / mean_latency) if mean_latency > 0 else 0.0
    load_balance_score = max(0.0, min(1.0, load_balance_score))  # Clamp to [0, 1]
    
    # Node efficiency: How similar are the node throughputs?
    # 1.0 = perfect (all nodes same speed), lower = worse
    node_efficiency = node_min_throughput / node_max_throughput if node_max_throughput > 0 else 0.0
    
    # Coefficient of variation for throughput
    throughput_cv = node_std_throughput / node_mean_throughput if node_mean_throughput > 0 else 0.0
    
    cluster_metrics = ClusterMetrics(
        total_nodes=total_nodes,
        total_ranks=total_ranks,
        total_gpus=total_gpus,
        unique_hostnames=unique_hostnames,
        mpi_library=mpi_library,
        total_samples_processed=total_samples,
        total_throughput_samples_per_s=total_throughput,
        total_bandwidth_MiB_s=total_bandwidth_MiB,
        total_bandwidth_MB_s=total_bandwidth_MB,
        mean_latency_ms=mean_latency,
        std_latency_ms=std_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        median_latency_ms=median_latency,
        load_balance_score=load_balance_score,
        node_efficiency=node_efficiency,
        throughput_cv=throughput_cv,
        node_mean_throughput=node_mean_throughput,
        node_std_throughput=node_std_throughput,
        node_min_throughput=node_min_throughput,
        node_max_throughput=node_max_throughput,
        per_node_metrics=node_metrics_list
    )
    
    logger.info(f"Cluster aggregation complete:")
    logger.info(f"  Total throughput: {total_throughput:.1f} samples/sec")
    logger.info(f"  Total bandwidth: {total_bandwidth_MiB:.2f} MiB/s")
    logger.info(f"  Load balance: {load_balance_score:.3f}")
    
    return cluster_metrics


def format_cluster_report(cluster_metrics: ClusterMetrics, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a human-readable cluster performance report.
    
    Args:
        cluster_metrics: ClusterMetrics object with aggregated statistics
        config: Optional benchmark configuration dictionary
        
    Returns:
        Formatted report string
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("                        CLUSTER-WIDE BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Cluster Configuration
    lines.append("Cluster Configuration:")
    lines.append(f"  Total Nodes:              {cluster_metrics.total_nodes}")
    lines.append(f"  Total Ranks:              {cluster_metrics.total_ranks}")
    lines.append(f"  Total GPUs:               {cluster_metrics.total_gpus}")
    lines.append(f"  Unique Hostnames:         {len(cluster_metrics.unique_hostnames)}")
    lines.append(f"  Nodes:                    {', '.join(cluster_metrics.unique_hostnames)}")
    lines.append(f"  MPI Library:              {cluster_metrics.mpi_library}")
    lines.append("")
    
    # Benchmark Configuration (if provided)
    if config:
        lines.append("Benchmark Configuration:")
        for key, value in sorted(config.items()):
            if key not in ['json_output', 'verbose']:  # Skip some internal params
                lines.append(f"  {key:24s}: {value}")
        lines.append("")
    
    # Cluster-Wide Performance
    lines.append("Cluster-Wide Performance:")
    lines.append(f"  Total Samples Processed:  {cluster_metrics.total_samples_processed:,}")
    lines.append(f"  Total Throughput:         {cluster_metrics.total_throughput_samples_per_s:,.1f} samples/sec")
    lines.append(f"  Total Bandwidth:          {cluster_metrics.total_bandwidth_MiB_s:,.2f} MiB/s "
                f"({cluster_metrics.total_bandwidth_MiB_s / 1024:.2f} GiB/s)")
    lines.append(f"                            {cluster_metrics.total_bandwidth_MB_s:,.2f} MB/s")
    lines.append(f"  Mean Latency:             {cluster_metrics.mean_latency_ms:.2f} ms")
    lines.append(f"  Median Latency:           {cluster_metrics.median_latency_ms:.2f} ms")
    lines.append(f"  Latency Range:            {cluster_metrics.min_latency_ms:.2f} - {cluster_metrics.max_latency_ms:.2f} ms")
    lines.append(f"  Latency Std Dev:          {cluster_metrics.std_latency_ms:.2f} ms")
    lines.append("")
    
    # Load Balance Analysis
    lines.append("Load Balance Analysis:")
    lines.append(f"  Load Balance Score:       {cluster_metrics.load_balance_score:.3f} "
                f"({'excellent' if cluster_metrics.load_balance_score >= 0.95 else 'good' if cluster_metrics.load_balance_score >= 0.90 else 'fair' if cluster_metrics.load_balance_score >= 0.80 else 'poor'})")
    lines.append(f"  Node Efficiency:          {cluster_metrics.node_efficiency:.3f} "
                f"({cluster_metrics.node_efficiency * 100:.1f}%)")
    lines.append(f"  Throughput CV:            {cluster_metrics.throughput_cv:.3f}")
    lines.append("")
    
    # Per-Node Statistics
    lines.append("Per-Node Performance:")
    lines.append(f"  {'Rank':<6} {'Hostname':<20} {'GPUs':<5} {'Samples':<10} {'Throughput (s/s)':<18} {'Bandwidth (MiB/s)':<18}")
    lines.append(f"  {'-'*6} {'-'*20} {'-'*5} {'-'*10} {'-'*18} {'-'*18}")
    
    for node in cluster_metrics.per_node_metrics:
        lines.append(f"  {node.rank:<6} {node.hostname:<20} {node.gpus:<5} "
                    f"{node.samples_processed:<10,} "
                    f"{node.throughput_samples_per_s:<18,.1f} "
                    f"{node.bandwidth_MiB_s:<18,.2f}")
    
    lines.append("")
    
    # Node Performance Variation
    lines.append("Node Performance Variation:")
    lines.append(f"  Mean Node Throughput:     {cluster_metrics.node_mean_throughput:,.1f} samples/sec")
    lines.append(f"  Std Dev:                  {cluster_metrics.node_std_throughput:,.1f} samples/sec "
                f"({cluster_metrics.throughput_cv * 100:.1f}%)")
    lines.append(f"  Fastest Node:             Rank {max(cluster_metrics.per_node_metrics, key=lambda n: n.throughput_samples_per_s).rank} "
                f"({cluster_metrics.node_max_throughput:,.1f} samples/sec)")
    lines.append(f"  Slowest Node:             Rank {min(cluster_metrics.per_node_metrics, key=lambda n: n.throughput_samples_per_s).rank} "
                f"({cluster_metrics.node_min_throughput:,.1f} samples/sec)")
    lines.append(f"  Speed Ratio (max/min):    {cluster_metrics.node_max_throughput / cluster_metrics.node_min_throughput:.2f}x")
    lines.append("")
    
    # Detailed Per-Node Timing
    lines.append("Per-Node Timing Statistics:")
    lines.append(f"  {'Rank':<6} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Median (ms)':<12}")
    lines.append(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for node in cluster_metrics.per_node_metrics:
        lines.append(f"  {node.rank:<6} "
                    f"{node.mean_iteration_ms:<12.2f} "
                    f"{node.std_iteration_ms:<12.2f} "
                    f"{node.min_iteration_ms:<12.2f} "
                    f"{node.max_iteration_ms:<12.2f} "
                    f"{node.median_iteration_ms:<12.2f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def write_cluster_json(cluster_metrics: ClusterMetrics, config: Dict[str, Any], 
                      output_path: str, benchmark_name: str = "unknown") -> None:
    """
    Write cluster metrics to JSON file.
    
    Args:
        cluster_metrics: ClusterMetrics object with aggregated statistics
        config: Benchmark configuration dictionary
        output_path: Path to output JSON file
        benchmark_name: Name of the benchmark type
    """
    try:
        # Convert NodeMetrics to dictionaries
        per_node_dicts = []
        for node in cluster_metrics.per_node_metrics:
            node_dict = asdict(node)
            per_node_dicts.append(node_dict)
        
        # Build JSON structure
        result = {
            "benchmark": benchmark_name,
            "cluster_config": {
                "total_nodes": cluster_metrics.total_nodes,
                "total_ranks": cluster_metrics.total_ranks,
                "total_gpus": cluster_metrics.total_gpus,
                "unique_hostnames": cluster_metrics.unique_hostnames,
                "mpi_library": cluster_metrics.mpi_library
            },
            "benchmark_config": config,
            "cluster_metrics": {
                "total_samples_processed": cluster_metrics.total_samples_processed,
                "total_throughput_samples_per_s": round(cluster_metrics.total_throughput_samples_per_s, 2),
                "total_bandwidth": {
                    "MiB_s": round(cluster_metrics.total_bandwidth_MiB_s, 2),
                    "GiB_s": round(cluster_metrics.total_bandwidth_MiB_s / 1024, 2),
                    "MB_s": round(cluster_metrics.total_bandwidth_MB_s, 2)
                },
                "latency_ms": {
                    "mean": round(cluster_metrics.mean_latency_ms, 2),
                    "std": round(cluster_metrics.std_latency_ms, 2),
                    "min": round(cluster_metrics.min_latency_ms, 2),
                    "max": round(cluster_metrics.max_latency_ms, 2),
                    "median": round(cluster_metrics.median_latency_ms, 2)
                },
                "load_balance": {
                    "score": round(cluster_metrics.load_balance_score, 3),
                    "node_efficiency": round(cluster_metrics.node_efficiency, 3),
                    "throughput_cv": round(cluster_metrics.throughput_cv, 3)
                },
                "node_throughput_stats": {
                    "mean": round(cluster_metrics.node_mean_throughput, 2),
                    "std": round(cluster_metrics.node_std_throughput, 2),
                    "min": round(cluster_metrics.node_min_throughput, 2),
                    "max": round(cluster_metrics.node_max_throughput, 2)
                }
            },
            "per_node_metrics": per_node_dicts
        }
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Wrote cluster metrics to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to write cluster JSON: {e}")
        raise


def compare_cluster_runs(json_files: List[str]) -> str:
    """
    Compare multiple cluster benchmark runs and generate a comparison report.
    
    Args:
        json_files: List of JSON result file paths
        
    Returns:
        Formatted comparison report string
    """
    if not json_files:
        return "No files provided for comparison"
    
    lines = []
    lines.append("=" * 80)
    lines.append("                    CLUSTER BENCHMARK COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    
    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results.append((Path(json_file).name, data))
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    if not results:
        return "No valid result files found"
    
    # Comparison table
    lines.append(f"{'Run':<30} {'Nodes':<8} {'Throughput (s/s)':<20} {'Bandwidth (GiB/s)':<20} {'Load Balance':<15}")
    lines.append(f"{'-'*30} {'-'*8} {'-'*20} {'-'*20} {'-'*15}")
    
    for filename, data in results:
        nodes = data['cluster_config']['total_nodes']
        throughput = data['cluster_metrics']['total_throughput_samples_per_s']
        bandwidth_gib = data['cluster_metrics']['total_bandwidth']['GiB_s']
        load_balance = data['cluster_metrics']['load_balance']['score']
        
        lines.append(f"{filename:<30} {nodes:<8} {throughput:<20,.1f} {bandwidth_gib:<20.2f} {load_balance:<15.3f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test cluster statistics with synthetic data
    print("Testing cluster statistics aggregation...")
    
    # Create synthetic node metrics
    all_node_metrics = [
        {
            'rank': 0,
            'hostname': 'node-0',
            'gpus': 4,
            'samples_processed': 10000,
            'shard_count': 6,
            'iteration_times_ms': [12.5, 12.3, 12.7, 12.4, 12.6],
            'mean_iteration_ms': 12.5,
            'std_iteration_ms': 0.15,
            'min_iteration_ms': 12.3,
            'max_iteration_ms': 12.7,
            'median_iteration_ms': 12.5,
            'throughput_samples_per_s': 20480,
            'bandwidth_MiB_s': 1280.0,
            'bandwidth_MB_s': 1342.0
        },
        {
            'rank': 1,
            'hostname': 'node-1',
            'gpus': 4,
            'samples_processed': 10000,
            'shard_count': 6,
            'iteration_times_ms': [12.8, 12.6, 13.0, 12.7, 12.9],
            'mean_iteration_ms': 12.8,
            'std_iteration_ms': 0.16,
            'min_iteration_ms': 12.6,
            'max_iteration_ms': 13.0,
            'median_iteration_ms': 12.8,
            'throughput_samples_per_s': 20000,
            'bandwidth_MiB_s': 1250.0,
            'bandwidth_MB_s': 1310.0
        }
    ]
    
    # Aggregate metrics
    cluster_metrics = aggregate_metrics(all_node_metrics, mpi_library="OpenMPI 4.1.4")
    
    # Generate report
    config = {
        'shard': '/test/shards/*/data.bin',
        'batch': 256,
        'workers': 16,
        'iterations': 5
    }
    
    report = format_cluster_report(cluster_metrics, config)
    print(report)
    
    # Write JSON
    write_cluster_json(cluster_metrics, config, "/tmp/test_cluster_metrics.json", "test_benchmark")
    print(f"\nWrote test JSON to /tmp/test_cluster_metrics.json")






