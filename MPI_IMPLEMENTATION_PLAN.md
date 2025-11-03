# MPI Implementation Plan for DALI DataLoader Tests

## Overview
This document outlines the plan to add MPI (Message Passing Interface) support to the DALI DataLoader Tests project, enabling parallel execution across multiple nodes with cluster-wide summary statistics.

## Goals
1. **Parallel Execution**: Run benchmarks simultaneously across multiple nodes/GPUs in a cluster
2. **Distributed Workload**: Each MPI rank independently runs benchmarks on its assigned data and GPUs
3. **Cluster Statistics**: Aggregate results from all nodes to provide comprehensive cluster-wide metrics
4. **Backward Compatibility**: Keep existing single-node scripts functional
5. **Flexibility**: Support running any benchmark type (binary, numpy, zarr) via MPI

## Architecture Overview

### MPI Topology
```
┌─────────────────────────────────────────────────────────┐
│                    MPI Coordinator                       │
│                      (Rank 0)                           │
│  • Launches MPI job across N nodes                      │
│  • Collects results from all ranks                      │
│  • Computes cluster-wide statistics                     │
│  • Writes unified summary report                        │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐   ┌────────▼───────┐   ┌──────▼────────┐
│   Node 0     │   │    Node 1      │   │   Node N-1    │
│  Rank 0      │   │  Rank 1        │   │  Rank N-1     │
│  • GPU 0-3   │   │  • GPU 0-3     │   │  • GPU 0-3    │
│  • Local     │   │  • Local       │   │  • Local      │
│    benchmark │   │    benchmark   │   │    benchmark  │
│  • Local     │   │  • Local       │   │  • Local      │
│    metrics   │   │    metrics     │   │    metrics    │
└──────────────┘   └────────────────┘   └───────────────┘
```

### Data Flow
1. **Initialization**: Each MPI rank initializes independently
2. **Data Sharding**: Each rank works on its assigned portion of the dataset
3. **Benchmark Execution**: All ranks run benchmarks in parallel
4. **Results Collection**: Rank 0 gathers results from all ranks via `MPI.Gather()`
5. **Aggregation**: Rank 0 computes cluster-wide statistics
6. **Reporting**: Rank 0 writes comprehensive summary with per-node and cluster metrics

## Components to Implement

### 1. MPI Dependencies
**File**: `requirements.txt`
```python
mpi4py  # MPI bindings for Python
```

### 2. MPI Utilities Module
**File**: `mpi_utils.py`

**Purpose**: Provide MPI initialization, coordination, and communication utilities

**Key Functions**:
- `init_mpi()`: Initialize MPI environment, return `comm`, `rank`, `size`
- `is_master()`: Check if current rank is master (rank 0)
- `barrier()`: Synchronize all ranks
- `gather_metrics(local_metrics, comm)`: Gather metrics from all ranks to rank 0
- `broadcast_config(config, comm)`: Broadcast configuration from rank 0 to all ranks
- `setup_rank_logging(rank, log_dir)`: Configure per-rank logging
- `get_rank_hostname()`: Get hostname for current rank (for identifying nodes)

**Key Features**:
- Handle MPI initialization/finalization gracefully
- Provide fallback behavior if MPI is not available (single-node mode)
- Proper error handling and cleanup

### 3. Cluster Statistics Module
**File**: `cluster_stats.py`

**Purpose**: Aggregate and compute cluster-wide statistics from all nodes

**Key Classes/Functions**:
- `ClusterMetrics`: Data class to hold aggregated metrics
  - Per-node metrics (list of individual node results)
  - Cluster-wide aggregates (mean, median, std, min, max, sum)
  - Throughput and bandwidth calculations
  
- `aggregate_metrics(all_node_metrics)`: Compute cluster statistics
  - Aggregate iteration times across all nodes
  - Compute cluster-wide throughput (sum of all nodes)
  - Compute cluster-wide bandwidth (sum of all nodes)
  - Calculate per-node statistics (mean, std, min, max for each node)
  - Calculate cluster statistics (overall mean, std, min, max across all nodes)
  
- `format_cluster_report(cluster_metrics)`: Generate human-readable report
  - Section 1: Cluster Overview (total nodes, total GPUs, total samples processed)
  - Section 2: Cluster-wide Aggregates (total throughput, total bandwidth, mean latency)
  - Section 3: Per-Node Statistics (individual node performance)
  - Section 4: Variation Analysis (coefficient of variation, load balance metrics)

- `write_cluster_json(cluster_metrics, output_path)`: Write JSON summary
  - Structured JSON with cluster and per-node metrics
  - Compatible with existing single-node JSON format
  - Additional cluster-specific fields

### 4. MPI-Enabled Benchmark Wrappers
**Files**: 
- `binary_random_read/dali_random_read_bin_mpi.py`
- `numpy_random_read/dali_random_read_numpy_mpi.py`
- `zarr_random_read/dali_random_read_zarr_mpi.py`

**Purpose**: Wrap existing benchmark scripts with MPI awareness

**Key Modifications**:
1. **MPI Initialization**: Initialize MPI at script start
2. **Rank-Specific Data Sharding**: 
   - Each rank gets a unique subset of shards
   - Use rank ID to determine which shards to process
   - Example: With 50 shards and 5 ranks, rank 0 gets shards 0-9, rank 1 gets shards 10-19, etc.
3. **GPU Assignment**:
   - Each rank uses its local GPUs
   - Example: 4 GPUs per node, each rank uses GPUs 0-3 on its node
4. **Independent Execution**: Each rank runs benchmark independently
5. **Metrics Collection**: Store results in structured format
6. **MPI Gather**: Rank 0 collects results from all ranks
7. **Cluster Summary**: Only rank 0 prints final statistics

**Design Pattern**:
```python
from mpi_utils import init_mpi, is_master, gather_metrics
from cluster_stats import aggregate_metrics, format_cluster_report

def main():
    # Initialize MPI
    comm, rank, world_size = init_mpi()
    
    # Existing argument parsing
    args = parse_args()
    
    # Shard assignment: each rank gets its portion
    all_shards = discover_shards(args.shard)
    my_shards = assign_shards_to_rank(all_shards, rank, world_size)
    
    # Run benchmark on local data
    local_metrics = run_benchmark(my_shards, args)
    
    # Gather results from all ranks to rank 0
    all_metrics = gather_metrics(local_metrics, comm)
    
    # Rank 0 aggregates and reports
    if is_master():
        cluster_metrics = aggregate_metrics(all_metrics)
        print(format_cluster_report(cluster_metrics))
        if args.json_output:
            write_cluster_json(cluster_metrics, args.json_output)
```

### 5. Unified MPI Runner Script
**File**: `mpi_runner.py`

**Purpose**: Universal launcher for any benchmark type via MPI

**Features**:
- Auto-detect benchmark type from shard pattern
- Support all benchmark-specific arguments
- Provide cluster-level configuration (nodes, GPUs per node, etc.)
- Generate unified output format

**Usage Example**:
```bash
# Launch with mpirun/mpiexec
mpirun -np 8 -hostfile hosts.txt python mpi_runner.py \
  --benchmark binary \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --iterations 10 \
  --json-output cluster_results.json
```

**Key Features**:
- Automatic MPI setup and teardown
- Dynamic benchmark selection
- Unified output format across all benchmark types
- Error handling and cleanup across all nodes

### 6. Documentation Updates
**File**: `README.md`

**New Sections to Add**:
1. **MPI Setup and Installation**
   - Installing MPI (OpenMPI, MPICH, Intel MPI)
   - Installing mpi4py
   - Verifying MPI installation

2. **Multi-Node Configuration**
   - Creating hostfile for MPI
   - SSH key setup for passwordless access
   - Network configuration requirements
   - Storage considerations (shared vs. distributed filesystem)

3. **Running MPI Benchmarks**
   - Basic MPI command examples
   - Using different benchmark types
   - Interpreting cluster statistics
   - Troubleshooting common issues

4. **Cluster Deployment Examples**
   - Small cluster (2-4 nodes)
   - Large cluster (8+ nodes)
   - Mixed GPU configurations

## Implementation Details

### Data Sharding Strategy

**Option 1: Shard-Level Partitioning** (Recommended)
- Each rank gets exclusive set of shards
- Clean separation, no coordination needed
- Example: 50 shards, 5 ranks → 10 shards per rank
- Simple implementation: `shards[rank::world_size]` or contiguous blocks

**Option 2: Sample-Level Partitioning** (Alternative)
- All ranks load all shards, but different samples
- More complex, requires coordination
- Better for uneven shard sizes
- Use for fallback if shard count < rank count

### Metrics Aggregation Strategy

**Per-Node Metrics** (collected from each rank):
- Iteration times (mean, std, min, max, median)
- Throughput (samples/sec)
- Bandwidth (MB/s, MiB/s)
- Memory usage (RSS, GPU memory)
- Sample count processed

**Cluster-Wide Metrics** (computed by rank 0):
- **Total Throughput**: Sum of all node throughputs
- **Total Bandwidth**: Sum of all node bandwidths
- **Mean Latency**: Average iteration time across all nodes
- **Load Balance**: Coefficient of variation of iteration times
- **Total Samples**: Sum of samples processed across all nodes
- **Node Efficiency**: Min throughput / Max throughput (closer to 1.0 = better balance)

### JSON Output Format

```json
{
  "benchmark": "binary_random_read",
  "cluster_config": {
    "total_nodes": 8,
    "total_gpus": 32,
    "ranks": 8,
    "mpi_library": "OpenMPI 4.1.0"
  },
  "config": {
    "shard": "/mnt/weka/shards_64k/*/data.bin",
    "batch": 256,
    "workers": 16,
    "iterations": 10
  },
  "cluster_metrics": {
    "total_throughput_samples_per_s": 164010.24,
    "total_bandwidth_MiB_s": 10250.64,
    "mean_latency_ms": 12.49,
    "load_balance_score": 0.96,
    "node_efficiency": 0.93
  },
  "per_node_metrics": [
    {
      "rank": 0,
      "hostname": "node-0",
      "gpus": 4,
      "samples": 10000,
      "iteration_ms": {"mean": 12.45, "std": 0.18},
      "throughput_samples_per_s": 20562.3,
      "bandwidth_MiB_s": 1283.89
    },
    ...
  ],
  "aggregated_stats": {
    "iteration_ms_across_nodes": {
      "mean": 12.49,
      "std": 0.34,
      "min": 11.87,
      "max": 13.12
    }
  }
}
```

## Testing Strategy

### Phase 1: Local Testing
1. Test with `mpirun -np 2` on single machine (2 ranks, same node)
2. Verify metrics collection and aggregation
3. Test with different numbers of ranks (1, 2, 4, 8)

### Phase 2: Multi-Node Testing
1. Test with 2 nodes, 1 rank per node
2. Test with 4 nodes, 1 rank per node
3. Test with varying ranks per node

### Phase 3: Integration Testing
1. Run all three benchmark types (binary, numpy, zarr) via MPI
2. Verify results match single-node runs (accounting for data sharding)
3. Test error handling (rank failure, network issues)

### Phase 4: Performance Validation
1. Verify linear scaling with node count
2. Check load balancing across nodes
3. Measure MPI communication overhead

## Deployment Considerations

### Prerequisites
1. **MPI Library**: OpenMPI 4.x or MPICH 3.x installed on all nodes
2. **Shared Storage**: All nodes must access same shard data (NFS, WEKA, Lustre, etc.)
3. **Network**: High-bandwidth, low-latency interconnect (InfiniBand, RoCE preferred)
4. **SSH Access**: Passwordless SSH between all nodes
5. **Python Environment**: Same Python + dependencies on all nodes (virtual env or container)

### Hostfile Example
```
node-0 slots=4
node-1 slots=4
node-2 slots=4
node-3 slots=4
```

### MPI Launch Examples

**Basic Launch (OpenMPI)**:
```bash
mpirun -np 8 -hostfile hosts.txt \
  python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 --workers 16 --iterations 10
```

**With Binding (for NUMA systems)**:
```bash
mpirun -np 8 -hostfile hosts.txt \
  --bind-to numa --map-by ppr:2:socket \
  python numpy_random_read/dali_random_read_numpy_mpi.py \
  --shard "/mnt/weka/shards_numpy/*/image.npy" \
  --batch 256 --workers 32 --iterations 10
```

**SLURM Integration**:
```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

srun python zarr_random_read/dali_random_read_zarr_mpi.py \
  --shard "/mnt/weka/shards/*/steps.zarr.pack" \
  --batch 256 --workers 16 --iterations 10
```

## Backward Compatibility

**Existing scripts remain unchanged** and continue to work as single-node benchmarks:
- `dali_random_read_bin.py`
- `dali_random_read_numpy.py`
- `dali_random_read_zarr.py`

**New MPI-enabled scripts** are separate:
- `dali_random_read_bin_mpi.py`
- `dali_random_read_numpy_mpi.py`
- `dali_random_read_zarr_mpi.py`

This allows users to:
- Use single-node scripts for local testing/development
- Use MPI scripts for cluster-scale benchmarking
- Gradually adopt MPI without breaking existing workflows

## Benefits of This Approach

1. **Scalability**: Linear scaling with number of nodes (near-ideal for embarrassingly parallel workload)
2. **Flexibility**: Support different cluster sizes, GPU configurations
3. **Comprehensive Metrics**: Per-node and cluster-wide statistics
4. **Production-Ready**: Real-world deployment patterns for ML data loading at scale
5. **Debugging**: Per-node logs help identify bottlenecks
6. **Storage Testing**: Stress-test shared filesystems with concurrent multi-node access

## Next Steps

1. ✅ Review and approve this plan
2. 🔲 Implement MPI dependencies (requirements.txt)
3. 🔲 Implement `mpi_utils.py` module
4. 🔲 Implement `cluster_stats.py` module
5. 🔲 Create MPI-enabled benchmark wrappers (start with binary)
6. 🔲 Test locally with multiple ranks
7. 🔲 Test on multi-node cluster
8. 🔲 Create unified `mpi_runner.py`
9. 🔲 Update documentation (README.md)
10. 🔲 Performance validation and tuning

---

**Questions to Consider:**

1. **MPI Library Preference**: Do you have a preferred MPI implementation (OpenMPI, MPICH, Intel MPI)?
2. **Cluster Environment**: Will this run on SLURM, bare-metal, Kubernetes, or other?
3. **Storage Backend**: What storage system (WEKA, Lustre, NFS, local SSDs)?
4. **Node Configuration**: Typical number of GPUs per node?
5. **Nebius SDK Integration**: Should we integrate with Nebius SDK for cluster management/orchestration?






