# Issue Analysis: Missing MPI Rank Data

## What You Observed

You ran with **2 MPI ranks** but the JSON file only contained data from **1 rank**.

### Logged Output (Both Ranks Ran)

**Rank 0:**
```
Mean:          32.45 ms
Throughput:    7888.5 samples/sec
Bandwidth:     3698.66 MiB/s
```

**Rank 1:**
```
Mean:          35.56 ms
Throughput:    7198.4 samples/sec  
Bandwidth:     3375.07 MiB/s
```

**Total Expected:** ~15,087 samples/sec, ~7,074 MiB/s

### What Was Saved (JSON File)

```json
{
  "metrics": {
    "iteration_ms": { "mean": 32.45 },
    "throughput_samples_per_s": { "mean": 7888.54 },
    "bandwidth": { "mean_MiB_s": 3698.66 }
  }
}
```

**Only Rank 0's data!** ❌

### The Problem

Both ranks wrote to the same file simultaneously without coordination:
- **50% of performance data lost**
- **No visibility into per-node differences** (Rank 1 was ~9% slower)
- **Cannot calculate total cluster throughput**

---

## After the Fix

When you re-run the benchmark, you'll get:

### 3 Output Files

1. **`numpy_2nodes_rank0.json`** - Rank 0's complete results
2. **`numpy_2nodes_rank1.json`** - Rank 1's complete results  
3. **`numpy_2nodes.json`** - Combined with totals:

```json
{
  "world_size": 2,
  "ranks": [
    { /* Full Rank 0 data */ },
    { /* Full Rank 1 data */ }
  ],
  "summary": {
    "total_throughput_samples_per_s": 15086.94,  ← Sum of both!
    "total_bandwidth_MiB_s": 7073.73,            ← Sum of both!
    "mean_iteration_ms": 34.005                  ← Average
  }
}
```

### Key Improvements

✅ **All data preserved** - No more lost metrics  
✅ **Per-rank analysis** - See which node is faster/slower  
✅ **True cluster metrics** - Total throughput across all nodes  
✅ **Debugging capability** - Identify performance imbalances

---

## How to Test

1. **Backup old results:**
   ```bash
   mv results/numpy_2nodes.json results/numpy_2nodes_old.json
   ```

2. **Re-run benchmark:**
   ```bash
   mpirun -np 2 -hostfile hosts.txt \
     /home/liran/venv/bin/python3 \
     /home/liran/dali_dataloader_tests/numpy_random_read/dali_random_read_numpy.py \
     --shard "/mnt/test/shards_numpy/*/image.npy" \
     --batch 256 \
     --workers 32 \
     --shuffle \
     --device-read-ahead 2 \
     --iterations 10 \
     --json-output results/numpy_2nodes.json
   ```

3. **Verify results:**
   ```bash
   python3 verify_mpi_results.py results/numpy_2nodes.json
   ```

4. **Compare:**
   ```bash
   # Old file (only 1 rank):
   cat results/numpy_2nodes_old.json | jq '.metrics.throughput_samples_per_s.mean'
   # Expected: ~7888
   
   # New file (both ranks):
   cat results/numpy_2nodes.json | jq '.summary.total_throughput_samples_per_s'
   # Expected: ~15087 (almost 2x!)
   ```

---

## Example: Performance Analysis

With the fixed output, you can now analyze:

```bash
# Which rank is faster?
cat results/numpy_2nodes_rank0.json | jq '.metrics.throughput_samples_per_s.mean'
cat results/numpy_2nodes_rank1.json | jq '.metrics.throughput_samples_per_s.mean'

# Bandwidth per node
cat results/numpy_2nodes_rank0.json | jq '.metrics.bandwidth.mean_MiB_s'
cat results/numpy_2nodes_rank1.json | jq '.metrics.bandwidth.mean_MiB_s'

# Total cluster performance
cat results/numpy_2nodes.json | jq '.summary'
```

This helps you:
- Identify slow nodes
- Debug network/storage bottlenecks
- Validate scaling efficiency
- Report accurate benchmarks

