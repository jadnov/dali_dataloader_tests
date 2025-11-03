# MPI Architecture Summary

## Quick Overview

This document provides a visual summary of the MPI integration for DALI DataLoader Tests.

## Component Structure

```
dali_dataloader_tests/
│
├── requirements.txt                     # ← Add mpi4py
│
├── mpi_utils.py                        # ← NEW: MPI coordination utilities
├── cluster_stats.py                    # ← NEW: Cluster statistics aggregation
├── mpi_runner.py                       # ← NEW: Unified MPI launcher
│
├── binary_random_read/
│   ├── dali_random_read_bin.py         # Existing (unchanged)
│   └── dali_random_read_bin_mpi.py     # ← NEW: MPI-enabled version
│
├── numpy_random_read/
│   ├── dali_random_read_numpy.py       # Existing (unchanged)
│   └── dali_random_read_numpy_mpi.py   # ← NEW: MPI-enabled version
│
└── zarr_random_read/
    ├── dali_random_read_zarr.py        # Existing (unchanged)
    └── dali_random_read_zarr_mpi.py    # ← NEW: MPI-enabled version
```

## Execution Flow

### Single-Node (Current)
```
┌────────────────────────────┐
│     User runs script       │
│  python dali_*.py          │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Load all shards          │
│   Use all local GPUs       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Run benchmark            │
│   Collect metrics          │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Display statistics       │
│   Write JSON output        │
└────────────────────────────┘
```

### Multi-Node MPI (New)
```
┌─────────────────────────────────────────────────────┐
│          User launches with mpirun                  │
│  mpirun -np 8 python dali_*_mpi.py                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    MPI spawns 8 ranks        │
        │  (e.g., 8 nodes, 1 per node) │
        └──────┬───────────────────────┘
               │
    ┌──────────┼──────────┬─────────────────┐
    │          │          │                 │
    ▼          ▼          ▼                 ▼
┌───────┐  ┌───────┐  ┌───────┐         ┌───────┐
│Rank 0 │  │Rank 1 │  │Rank 2 │   ...   │Rank 7 │
│Node 0 │  │Node 1 │  │Node 2 │         │Node 7 │
└───┬───┘  └───┬───┘  └───┬───┘         └───┬───┘
    │          │          │                 │
    │ Shards   │ Shards   │ Shards          │ Shards
    │ 0-5      │ 6-11     │ 12-17           │ 42-47
    │          │          │                 │
    │ Local    │ Local    │ Local           │ Local
    │ GPUs     │ GPUs     │ GPUs            │ GPUs
    │ 0-3      │ 0-3      │ 0-3             │ 0-3
    │          │          │                 │
    ▼          ▼          ▼                 ▼
┌───────┐  ┌───────┐  ┌───────┐         ┌───────┐
│ Local │  │ Local │  │ Local │         │ Local │
│Metrics│  │Metrics│  │Metrics│         │Metrics│
└───┬───┘  └───┬───┘  └───┬───┘         └───┬───┘
    │          │          │                 │
    └──────────┼──────────┴─────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │    MPI.Gather()      │
    │  All metrics → Rank 0│
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Rank 0 Only:       │
    │ • Aggregate metrics  │
    │ • Compute cluster    │
    │   statistics         │
    │ • Print summary      │
    │ • Write JSON         │
    └──────────────────────┘
```

## Data Sharding Strategy

### Example: 48 Shards, 8 MPI Ranks

**Contiguous Block Assignment** (Recommended):
```
Rank 0: Shards [0, 1, 2, 3, 4, 5]        → 6 shards
Rank 1: Shards [6, 7, 8, 9, 10, 11]      → 6 shards
Rank 2: Shards [12, 13, 14, 15, 16, 17]  → 6 shards
Rank 3: Shards [18, 19, 20, 21, 22, 23]  → 6 shards
Rank 4: Shards [24, 25, 26, 27, 28, 29]  → 6 shards
Rank 5: Shards [30, 31, 32, 33, 34, 35]  → 6 shards
Rank 6: Shards [36, 37, 38, 39, 40, 41]  → 6 shards
Rank 7: Shards [42, 43, 44, 45, 46, 47]  → 6 shards
```

Implementation:
```python
def assign_shards_to_rank(all_shards, rank, world_size):
    shards_per_rank = len(all_shards) // world_size
    start = rank * shards_per_rank
    end = start + shards_per_rank if rank < world_size - 1 else len(all_shards)
    return all_shards[start:end]
```

## Metrics Flow

### Local Metrics (Per-Rank)
Each rank collects:
```python
{
  "rank": 0,
  "hostname": "node-0",
  "gpus": 4,
  "samples_processed": 10000,
  "shard_count": 6,
  "iteration_times_ms": [12.5, 12.3, 12.7, ...],
  "memory_stats": {
    "process_rss_mb": 2048.5,
    "gpu_allocated_mb": [512, 512, 512, 512]
  }
}
```

### Cluster Metrics (Rank 0 Only)
Rank 0 aggregates:
```python
{
  "cluster_config": {
    "total_nodes": 8,
    "total_gpus": 32,
    "total_ranks": 8
  },
  "cluster_metrics": {
    "total_throughput_samples_per_s": 164010.24,  # Sum across all ranks
    "total_bandwidth_MiB_s": 10250.64,             # Sum across all ranks
    "mean_latency_ms": 12.49,                      # Average across all ranks
    "load_balance_score": 0.96                     # Std / Mean of iteration times
  },
  "per_node_metrics": [
    {"rank": 0, "throughput": 20562.3, ...},
    {"rank": 1, "throughput": 20481.7, ...},
    ...
  ]
}
```

## Key MPI Functions

### mpi_utils.py
```python
from mpi4py import MPI

def init_mpi():
    """Initialize MPI and return comm, rank, size"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def is_master(rank):
    """Check if current rank is master (rank 0)"""
    return rank == 0

def gather_metrics(local_metrics, comm):
    """Gather metrics from all ranks to rank 0"""
    all_metrics = comm.gather(local_metrics, root=0)
    return all_metrics

def barrier(comm):
    """Synchronize all ranks"""
    comm.Barrier()
```

### cluster_stats.py
```python
def aggregate_metrics(all_node_metrics):
    """Compute cluster-wide statistics"""
    # Extract iteration times from all nodes
    all_times = []
    for node_metrics in all_node_metrics:
        all_times.extend(node_metrics['iteration_times_ms'])
    
    # Compute cluster statistics
    cluster_metrics = {
        'total_throughput': sum(node['throughput'] for node in all_node_metrics),
        'mean_latency': np.mean(all_times),
        'std_latency': np.std(all_times),
        'min_latency': np.min(all_times),
        'max_latency': np.max(all_times),
        'load_balance': compute_load_balance(all_node_metrics)
    }
    
    return cluster_metrics
```

## Usage Examples

### Launch Binary Benchmark on 8 Nodes
```bash
mpirun -np 8 -hostfile hosts.txt \
  python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --iterations 10 \
  --json-output cluster_results.json
```

### Launch NumPy Benchmark on 4 Nodes
```bash
mpirun -np 4 -hostfile hosts.txt \
  python numpy_random_read/dali_random_read_numpy_mpi.py \
  --shard "/mnt/weka/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 32 \
  --shuffle \
  --iterations 10
```

### Using Unified MPI Runner
```bash
mpirun -np 8 -hostfile hosts.txt \
  python mpi_runner.py \
  --benchmark zarr \
  --shard "/mnt/weka/shards/*/steps.zarr.pack" \
  --batch 128 \
  --workers 16 \
  --iterations 10
```

## Expected Output

### Per-Node Output (Each Rank)
```
[Rank 0 @ node-0] Starting benchmark...
[Rank 0 @ node-0] Assigned shards: 0-5 (6 shards)
[Rank 0 @ node-0] Using 4 local GPUs
[Rank 0 @ node-0] Iteration 1: 12.45 ms
[Rank 0 @ node-0] Iteration 2: 12.38 ms
...
[Rank 0 @ node-0] Local metrics: 20562 samples/sec
```

### Cluster Summary (Rank 0 Only)
```
======================================================================
                    CLUSTER-WIDE SUMMARY
======================================================================
Cluster Configuration:
  Total Nodes:       8
  Total GPUs:        32
  Total Ranks:       8
  MPI Library:       OpenMPI 4.1.4

Cluster-Wide Performance:
  Total Throughput:  164,010 samples/sec
  Total Bandwidth:   10,250.64 MiB/s (10.01 GiB/s)
  Mean Latency:      12.49 ms
  Load Balance:      0.96 (excellent)
  Node Efficiency:   94.3%

Per-Node Statistics:
  Rank 0 (node-0):   20,562 samples/sec,  1,284 MiB/s
  Rank 1 (node-1):   20,481 samples/sec,  1,279 MiB/s
  Rank 2 (node-2):   20,623 samples/sec,  1,288 MiB/s
  ...
  
Variation Analysis:
  Throughput StdDev: 145.2 samples/sec (0.88%)
  Fastest Node:      Rank 5 (20,892 samples/sec)
  Slowest Node:      Rank 3 (19,734 samples/sec)
  Speed Ratio:       1.06x (good balance)
======================================================================
```

## Benefits Summary

| Feature | Single-Node | MPI Multi-Node |
|---------|-------------|----------------|
| Throughput | Limited by single node | Linear scaling with nodes |
| Data Volume | Limited by node memory/storage | Distributed across cluster |
| Failure Isolation | Single point of failure | Per-node monitoring |
| Storage Testing | Single client stress | Multi-client concurrent stress |
| Production Realism | Local simulation | Real distributed workload |
| Scalability | 1-8 GPUs typical | 8-256+ GPUs |

## Load Balancing Metrics

**Load Balance Score**: Measures how evenly work is distributed
```
load_balance_score = 1.0 - (std_dev(iteration_times) / mean(iteration_times))

• 1.0 = Perfect balance (all nodes identical)
• 0.9+ = Excellent balance
• 0.8-0.9 = Good balance
• < 0.8 = Poor balance (investigate bottlenecks)
```

**Node Efficiency**: Measures slowest vs fastest node
```
node_efficiency = min_throughput / max_throughput

• 1.0 = Perfect (all nodes same speed)
• 0.95+ = Excellent
• 0.90-0.95 = Good
• < 0.90 = Investigate slow nodes
```

## Troubleshooting Common Issues

### Issue: Ranks hang at startup
**Cause**: Network/firewall blocking MPI communication
**Solution**: Check firewall rules, verify passwordless SSH

### Issue: Uneven performance across nodes
**Cause**: Hardware differences, network bottlenecks, storage hotspots
**Solution**: Check per-node metrics, verify identical hardware, test storage performance

### Issue: "No shards found" on some ranks
**Cause**: Shared storage not mounted on all nodes
**Solution**: Verify all nodes can access shard directory

### Issue: Out of memory on some ranks
**Cause**: Uneven shard sizes or memory allocation
**Solution**: Use shard-level partitioning, reduce batch size

---

**Ready to implement? See `MPI_IMPLEMENTATION_PLAN.md` for detailed implementation steps.**






