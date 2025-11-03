# Hang Fix v2 - Complete Solution

## Problem
Script was hanging after "Resource cleanup completed" when running with multiple MPI ranks.

## Root Causes Identified

### 1. Missing Final Synchronization
After cleanup, ranks weren't synchronized before exit, causing MPI to hang waiting for coordination.

### 2. Double Cleanup Issue  
`cleanup_resources()` was being called twice:
- Once explicitly in the `finally` block
- Once by `atexit` handler
This could cause timing issues.

### 3. Variable Scope in Finally Block
Checking `'comm' in locals()` in the finally block had scope issues.

### 4. No Explicit Exit
Script relied on implicit exit after main() returns, but MPI processes need explicit coordination.

---

## Fixes Applied

### Fix 1: Prevent Double Cleanup
Added `_cleanup_done` flag to make cleanup idempotent:

```python
_cleanup_done = False

def cleanup_resources():
    global _cleanup_done
    if _cleanup_done:
        return
    # ... cleanup code ...
    _cleanup_done = True
```

### Fix 2: Save Variables Before Finally Block
Store comm/rank/world_size at the start of finally block:

```python
finally:
    saved_rank = rank if 'rank' in locals() else '?'
    saved_world_size = world_size if 'world_size' in locals() else 1
    saved_comm = comm if 'comm' in locals() else None
```

### Fix 3: Final Barrier After Cleanup
Add synchronization after cleanup with error handling:

```python
if saved_world_size > 1 and saved_comm is not None:
    logger.info(f"[Rank {saved_rank}] Final barrier before exit...")
    sys.stdout.flush()
    try:
        barrier(saved_comm)
        logger.info(f"[Rank {saved_rank}] Final barrier passed")
    except Exception as e:
        logger.warning(f"[Rank {saved_rank}] Final barrier failed: {e}")
```

### Fix 4: Explicit sys.exit(0)
Force clean exit after main() completes:

```python
if __name__ == "__main__":
    try:
        main()
        sys.exit(0)  # Explicit exit
    except SystemExit:
        raise
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
```

### Fix 5: Flush Output Buffers
Added `sys.stdout.flush()` after critical log messages to ensure output isn't buffered.

---

## Expected Output Now

You should see:

```
======================================================================
Cleaning up resources...
Resource cleanup completed
[Rank 0] Entering finally block...
[Rank 0] Final barrier before exit...
[Rank 0] Final barrier passed
[Rank 0] Cleanup complete, exiting...

Cleaning up resources...
Resource cleanup completed
[Rank 1] Entering finally block...
[Rank 1] Final barrier before exit...
[Rank 1] Final barrier passed
[Rank 1] Cleanup complete, exiting...
```

**Both ranks should exit cleanly within 1-2 seconds.**

---

## Testing

### Quick Test (60s timeout):
```bash
cd /home/liran/dali_dataloader_tests
./run_with_timeout.sh 60
```

### Full Test:
```bash
mpirun -np 2 -hostfile hosts.txt \
  /home/liran/venv/bin/python3 \
  numpy_random_read/dali_random_read_numpy.py \
  --shard "/mnt/test/shards_numpy/*/image.npy" \
  --batch 256 --workers 32 --shuffle \
  --device-read-ahead 2 --iterations 10 \
  --json-output results/numpy_2nodes.json
```

### Verify Results:
```bash
python3 verify_mpi_results.py results/numpy_2nodes.json
```

---

## Debugging If Still Hangs

If it still hangs, run with the timeout script to capture where:

```bash
./run_with_timeout.sh 30
cat /tmp/mpi_benchmark_output.log | tail -50
```

Look for the **last log message** to see where it's stuck:

- ❌ Stuck after "Cleaning up resources..." → Issue in cleanup_resources()
- ❌ Stuck after "Resource cleanup completed" → Issue before finally block
- ❌ Stuck after "Entering finally block" → Issue in finally block execution
- ❌ Stuck after "Final barrier before exit" → MPI barrier deadlock

---

## What Should Work

✅ Script completes benchmark  
✅ Both ranks write individual JSON files  
✅ Rank 0 writes combined JSON with totals  
✅ All ranks synchronize and exit cleanly  
✅ No hanging processes  
✅ Exit code 0

**Total runtime: ~10-15 seconds for 10 iterations**

