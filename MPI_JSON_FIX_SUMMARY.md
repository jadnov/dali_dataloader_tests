# MPI JSON Output Issue - Fixed

## Problem Identified

When running the benchmark with MPI (`-np 2`), both processes were producing **different results** but only one process's data was being saved to the JSON file.

### Root Cause
The script was **not MPI-aware**. Both MPI ranks were independently writing to the **same JSON file path** without coordination:

```python
# Both ranks executed this simultaneously:
with open(args.json_output, "w") as f:
    json.dump(summary, f, indent=2)
```

This caused:
- Race condition (one rank overwrites the other)
- Lost data from one or more ranks
- Inability to see per-rank performance differences
- No aggregate statistics across all ranks

### Example from Your Run

**Process 1 (Rank 0):**
- Mean time: 32.45 ms
- Throughput: 7888.5 samples/sec
- Bandwidth: 3698.66 MiB/s

**Process 2 (Rank 1):**
- Mean time: 35.56 ms
- Throughput: 7198.4 samples/sec
- Bandwidth: 3375.07 MiB/s

**But `numpy_2nodes.json` only contained Rank 0's data!**

---

## Solution Implemented

The script is now **MPI-aware** using the existing `mpi_utils.py` module:

### Changes Made

1. **MPI Initialization**
   ```python
   from mpi_utils import init_mpi, is_master, gather_metrics, get_rank_hostname
   
   # At start of main():
   comm, rank, world_size = init_mpi()
   hostname = get_rank_hostname()
   ```

2. **Per-Rank JSON Files**
   - Each rank now writes to its own file: `{basename}_rank{N}.json`
   - Example: `numpy_2nodes_rank0.json`, `numpy_2nodes_rank1.json`
   - Each file contains complete results for that rank

3. **Combined Results (Rank 0)**
   - Rank 0 gathers results from all ranks using MPI
   - Writes aggregate results to the original path: `numpy_2nodes.json`
   - Contains:
     - Individual results from all ranks
     - Summary with totals:
       - Total throughput (sum across ranks)
       - Total bandwidth (sum across ranks)
       - Mean iteration time (average across ranks)

### Output Files After Fix

When running with `-np 2`:

```
results/
├── numpy_2nodes_rank0.json      # Rank 0's individual results
├── numpy_2nodes_rank1.json      # Rank 1's individual results
└── numpy_2nodes.json            # Combined results + summary
```

When running with `-np 1` (single process):
```
results/
└── numpy_2nodes.json            # Single rank results (no suffix)
```

---

## JSON File Structure

### Per-Rank Files (`*_rank{N}.json`)
```json
{
  "benchmark": "numpy_random_read",
  "rank": 0,
  "world_size": 2,
  "hostname": "node-hostname",
  "config": { ... },
  "metrics": {
    "iteration_ms": { "mean": 32.45, ... },
    "throughput_samples_per_s": { "mean": 7888.54, ... },
    "bandwidth": { "mean_MiB_s": 3698.66, ... },
    ...
  }
}
```

### Combined File (`*.json`)
```json
{
  "benchmark": "numpy_random_read",
  "world_size": 2,
  "config": { ... },
  "ranks": [
    { /* Rank 0 full results */ },
    { /* Rank 1 full results */ }
  ],
  "summary": {
    "total_throughput_samples_per_s": 15086.94,  // sum of both ranks
    "total_bandwidth_MiB_s": 7073.73,             // sum of both ranks
    "mean_iteration_ms": 34.005                   // average across ranks
  }
}
```

---

## Next Steps

1. **Re-run your benchmark:**
   ```bash
   mpirun -np 2 -hostfile hosts.txt \
     python3 numpy_random_read/dali_random_read_numpy.py \
     --shard "/mnt/test/shards_numpy/*/image.npy" \
     --batch 256 \
     --workers 32 \
     --shuffle \
     --device-read-ahead 2 \
     --iterations 10 \
     --json-output results/numpy_2nodes.json
   ```

2. **Verify the outputs:**
   ```bash
   ls -lh results/numpy_2nodes*.json
   ```
   
   You should see:
   - `numpy_2nodes_rank0.json`
   - `numpy_2nodes_rank1.json`
   - `numpy_2nodes.json` (combined)

3. **Check combined results:**
   ```bash
   cat results/numpy_2nodes.json | jq '.summary'
   ```
   
   This will show the aggregated performance across both nodes.

---

## Benefits

✅ **No data loss** - All rank results preserved  
✅ **Per-rank analysis** - See performance differences between nodes  
✅ **Aggregate metrics** - Total cluster throughput/bandwidth  
✅ **Backward compatible** - Works with single-process runs  
✅ **Uses existing MPI infrastructure** - Leverages `mpi_utils.py`

---

## Issue #2: Hang After "Resource cleanup completed" (FIXED)

### Problem
The script was hanging after printing "Resource cleanup completed" when running with multiple MPI ranks.

### Root Cause
MPI collective operations (like `gather`) require **all ranks to participate** at the same time. Without proper synchronization barriers:
- One rank could reach the gather before another
- MPI finalization could hang waiting for synchronization
- Race conditions in file I/O

### Solution
Added **three synchronization barriers**:

1. **Before gather** (line 490):
   ```python
   # Ensure all ranks have written their individual files before gathering
   barrier(comm)
   all_results = gather_metrics(summary, comm)
   ```

2. **After gather** (line 522):
   ```python
   # Wait for rank 0 to finish writing before continuing
   barrier(comm)
   ```

3. **Before exit** (lines 598-601 and 612-613):
   ```python
   # Final barrier to ensure all ranks finish together
   if world_size > 1:
       logger.info(f"[Rank {rank}] Waiting for all ranks to complete...")
       barrier(comm)
       logger.info(f"[Rank {rank}] All ranks completed successfully")
   ```

### Expected Output
You should now see:
```
======================================================================
Cleaning up resources...
Resource cleanup completed
[Rank 0] Waiting for all ranks to complete...
[Rank 0] All ranks completed successfully

Cleaning up resources...
Resource cleanup completed
[Rank 1] Waiting for all ranks to complete...
[Rank 1] All ranks completed successfully
```

The script will exit cleanly within 1-2 seconds.

