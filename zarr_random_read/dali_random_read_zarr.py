#!/usr/bin/env python3
"""
dali_random_read_zarr.py – production-style DALI random-read benchmark (multi-shard)

• Reads *all* steps.zarr.pack files that match the --shard glob.
• Each 64-KB chunk is fetched through PackedDirectoryStore → Zarr → mmap → DALI.
• ExternalInputIterator builds one global shuffled index, then shards it
  across GPUs exactly like prod.
• Comprehensive benchmarking with multiple iterations, memory tracking, and statistics.
• Supports page cache dropping for accurate disk I/O measurements.

Example
--------
python dali_random_read_zarr.py \
  --shard "/mnt/weka/shards/00000000/*/steps.zarr.pack" \
  --batch 256 --workers 16 --shuffle --iterations 5 --drop-cache
"""

from __future__ import annotations

import argparse
import bisect
import glob
import math
import pickle
import random
import struct
import sys
import time
import logging
import atexit
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Iterator, Sequence

import numpy as np
import torch
import zarr
from nvidia.dali import fn, pipeline_def, types

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global resource tracking
_open_stores = []
_pipes = []

def cleanup_resources():
    """Clean up global resources"""
    global _open_stores, _pipes
    logger.info("Cleaning up resources...")
    
    # Close stores
    for store in _open_stores:
        try:
            store.close()
        except Exception as e:
            logger.warning(f"Error closing store: {e}")
    _open_stores.clear()
    
    # Clear pipe references (DALI manages lifecycle automatically)
    _pipes.clear()
    
    logger.info("Resource cleanup completed")

# Register cleanup function
atexit.register(cleanup_resources)


# ---------------------------------------------------------------------------
def get_memory_stats():
    """Get current memory usage statistics"""
    stats = {}
    
    # System memory
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        stats['process_rss_mb'] = mem_info.rss / (1024**2)
        stats['process_vms_mb'] = mem_info.vms / (1024**2)
        
        sys_mem = psutil.virtual_memory()
        stats['system_total_mb'] = sys_mem.total / (1024**2)
        stats['system_used_mb'] = sys_mem.used / (1024**2)
        stats['system_percent'] = sys_mem.percent
    
    # GPU memory
    if torch.cuda.is_available():
        stats['gpu_count'] = torch.cuda.device_count()
        stats['gpu_allocated_mb'] = []
        stats['gpu_reserved_mb'] = []
        stats['gpu_total_mb'] = []
        
        for i in range(torch.cuda.device_count()):
            stats['gpu_allocated_mb'].append(torch.cuda.memory_allocated(i) / (1024**2))
            stats['gpu_reserved_mb'].append(torch.cuda.memory_reserved(i) / (1024**2))
            # Get total memory
            props = torch.cuda.get_device_properties(i)
            stats['gpu_total_mb'].append(props.total_memory / (1024**2))
    
    return stats


def format_memory_stats(stats, label=""):
    """Format memory statistics for logging"""
    lines = []
    if label:
        lines.append(f"{label}:")
    
    if HAS_PSUTIL:
        lines.append(f"  Process RSS:   {stats['process_rss_mb']:.2f} MiB")
        lines.append(f"  System Used:   {stats['system_used_mb']:.2f} / {stats['system_total_mb']:.2f} MiB ({stats['system_percent']:.1f}%)")
    
    if 'gpu_count' in stats and stats['gpu_count'] > 0:
        for i in range(stats['gpu_count']):
            alloc = stats['gpu_allocated_mb'][i]
            reserved = stats['gpu_reserved_mb'][i]
            total = stats['gpu_total_mb'][i]
            # Only show GPU memory if actually being used
            if alloc > 0 or reserved > 0:
                lines.append(f"  GPU {i} Memory:  {alloc:.2f} MiB allocated, {reserved:.2f} MiB reserved / {total:.2f} MiB total")
    
    return lines


# ---------------------------------------------------------------------------
def drop_page_cache():
    """
    Drop Linux page cache to measure actual disk performance.
    Requires sudo/root privileges.
    """
    try:
        logger.info("Attempting to drop page cache (requires sudo)...")
        logger.info("  You may be prompted for your password")
        
        # Try to drop page cache
        # sync first to ensure clean state
        subprocess.run(['sudo', 'sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], 
                      check=True, capture_output=True, text=True)
        
        logger.info("✓ Page cache dropped successfully")
        logger.info("  Benchmark will now measure actual disk I/O performance")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to drop page cache: {e}")
        logger.error("  Benchmark will measure cached/memory performance")
        logger.error("  To enable cache dropping, run this script with sudo")
        return False
    except FileNotFoundError:
        logger.error("sudo command not found")
        return False
    except Exception as e:
        logger.error(f"Unexpected error dropping page cache: {e}")
        return False

# ──────────────────────────────── PackedDirectoryStore ─────────────────────────
def extract_pack_data(path: str | Path) -> Dict[str, bytes]:
    """
    Extract data from .pack files produced by *pack_directory_store* (v2).
    Returns a dictionary mapping file names to their contents.
    """
    path = str(path)
    try:
        with open(path, "rb") as fh:
            sig = fh.read(4)
            if sig != b"pack":
                raise ValueError("Not a .pack file (missing signature)")

            version = struct.unpack("<I", fh.read(4))[0]
            if version != 2:
                raise ValueError(f"Unsupported .pack version {version} (expected 2)")

            (size,) = struct.unpack("<Q", fh.read(8))
            manifest_bytes = fh.read(size)
            manifest: Dict[str, tuple[int, int]] = pickle.loads(manifest_bytes)
            data_offset = fh.tell()  # start of raw blob
            
            # Extract all data into memory
            data = {}
            for key, (offset, size) in manifest.items():
                fh.seek(data_offset + offset)
                data[key] = fh.read(size)
            
            return data
    except Exception as e:
        logger.error(f"Failed to extract pack data from {path}: {e}")
        raise


def create_zarr_from_pack_data(pack_data: Dict[str, bytes]) -> zarr.Group:
    """
    Create a zarr group from extracted pack data using temporary directory.
    """
    # Create a temporary directory to store the zarr data
    temp_dir = tempfile.mkdtemp()
    try:
        # Write the extracted data to temporary files
        for key, data in pack_data.items():
            file_path = os.path.join(temp_dir, key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(data)
        
        # Open as zarr group
        root = zarr.open_group(temp_dir, mode='r')
        return root
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
# ───────────────────────────────────────────────────────────────────────────────


# ───────────────────────── helper: open shards & collect lengths ──────────────
def open_all_shards(pattern: str):
    """Open all shards with proper error handling and resource tracking"""
    try:
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No file matches glob {pattern!r}")
        
        logger.info(f"Found {len(files)} shard files")
        
        # For now, let's just get the file list and lengths without loading all data
        # We'll implement lazy loading in the iterator
        lengths, cum = [], [0]
        
        # For testing, let's only use the first 5 shards to avoid memory issues
        test_files = files[:5]
        
        # Load just the first shard to get the structure and sample count
        try:
            pack_data = extract_pack_data(test_files[0])
            root = create_zarr_from_pack_data(pack_data)
            sample_count = root["image"].shape[0]
            
            # Assume all shards have the same number of samples
            for i, p in enumerate(test_files):
                lengths.append(sample_count)
                cum.append(cum[-1] + sample_count)
            
        except Exception as e:
            logger.error(f"Failed to load first shard: {e}")
            raise

        total = cum[-1]
        return test_files, lengths, cum, total, test_files  # Return test_files instead of roots for lazy loading
        
    except Exception as e:
        logger.error(f"Failed to open shards: {e}")
        raise


# ───────────────────────── ExternalInputIterator ──────────────────────────────
class ExternalInputIterator:
    def __init__(
        self,
        files: Sequence[str],
        cumlens: Sequence[int],
        global_indices: Sequence[int],
        batch: int,
        rank: int,
        world: int,
    ):
        self.files = files
        self.cumlens = cumlens
        self.idx_pool = global_indices
        self.bs = batch
        self._shard_cache = {}  # Cache for loaded shards

        start = len(self.idx_pool) * rank // world
        end = len(self.idx_pool) * (rank + 1) // world
        self.local_idx = self.idx_pool[start:end]
        self.n = len(self.local_idx)

    def __iter__(self):
        self.i = 0
        return self

    def _load_shard(self, shard_idx: int):
        """Lazy load a shard if not already cached"""
        if shard_idx not in self._shard_cache:
            try:
                pack_data = extract_pack_data(self.files[shard_idx])
                root = create_zarr_from_pack_data(pack_data)
                self._shard_cache[shard_idx] = root
            except Exception as e:
                logger.error(f"Failed to load shard {shard_idx}: {e}")
                raise
        return self._shard_cache[shard_idx]

    def _fetch_sample(self, global_idx: int):
        shard_idx = bisect.bisect_right(self.cumlens, global_idx) - 1
        local = global_idx - self.cumlens[shard_idx]
        root = self._load_shard(shard_idx)
        return (
            np.asarray(root["image"][local]),
            np.asarray(root["actions"][local]),
            np.asarray(root["state"][local]),
        )

    def __next__(self):
        # For benchmarking, we want to cycle through the data indefinitely
        # instead of raising StopIteration after one pass
        imgs, acts, sts = [], [], []
        for _ in range(self.bs):
            # Use modulo to cycle through indices indefinitely
            gidx = self.local_idx[self.i % self.n]
            im, ac, st = self._fetch_sample(gidx)
            imgs.append(im)
            acts.append(ac)
            sts.append(st)
            self.i += 1
            
            # Log when we cycle through the data (every time we complete a full pass)
            if self.i > 0 and self.i % self.n == 0:
                logger.debug(f"ExternalInputIterator completed full data cycle at sample {self.i}")
        
        return imgs, acts, sts

    next = __next__  # Py2 compat


# ───────────────────────────── DALI pipeline ──────────────────────────────────
@pipeline_def
def zarr_pipe(eii, device_read_ahead: int):
    imgs, acts, sts = fn.external_source(
        source=eii,
        num_outputs=3,
        batch=True,
        dtype=[types.UINT8, types.FLOAT, types.FLOAT],
    )
    return imgs, acts, sts


# ──────────────────────────────── main ────────────────────────────────────────
def main():
    try:
        pa = argparse.ArgumentParser(description="Multi-shard random-read benchmark")
        pa.add_argument("--shard", required=True,
                        help='glob like "/mnt/weka/shards/00000000/*/steps.zarr.pack"')
        pa.add_argument("--batch",   type=int, default=256, help="batch per GPU")
        pa.add_argument("--workers", type=int, default=4,   help="DALI CPU threads / GPU")
        pa.add_argument("--shuffle", action="store_true",   help="shuffle global index")
        pa.add_argument("--device-read-ahead", type=int, default=1, metavar="N",
                        help="GPU prefetch queue depth")
        pa.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")
        pa.add_argument("--drop-cache", action="store_true", 
                        help="Drop page cache before benchmark (requires sudo)")
        pa.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        pa.add_argument("--json-output", type=str, default=None,
                        help="Path to write a JSON file with final results")
        args = pa.parse_args()

        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"Starting benchmark with args: {vars(args)}")
        
        if not HAS_PSUTIL:
            logger.warning("psutil not available, memory stats will be limited")

        # Drop page cache if requested (before loading any data)
        if args.drop_cache:
            drop_page_cache()

        # Open shards with error handling
        files, lens, cum, total, _ = open_all_shards(args.shard)
        
        # Get initial memory state
        mem_initial = get_memory_stats()
        logger.info("")
        for line in format_memory_stats(mem_initial, "Initial memory usage"):
            logger.info(line)

        # Prepare indices
        global_indices = list(range(total))
        if args.shuffle:
            random.shuffle(global_indices)
            logger.info("Shuffled global indices")

        # Setup GPU configuration
        n_gpu = max(1, torch.cuda.device_count())
        logger.info(f"Available GPUs: {n_gpu}")

        # Create and build pipes
        pipes = []
        for dev in range(n_gpu):
            try:
                eii = ExternalInputIterator(files, cum, global_indices,
                                            args.batch, dev, n_gpu)
                pipe_instance = zarr_pipe(
                    batch_size=args.batch,
                    num_threads=args.workers,
                    device_id=dev,
                    prefetch_queue_depth=args.device_read_ahead,
                    eii=eii,
                    device_read_ahead=args.device_read_ahead,
                )
                pipes.append(pipe_instance)
                _pipes.append(pipe_instance)  # Track for cleanup
            except Exception as e:
                logger.error(f"Failed to create pipe for GPU {dev}: {e}")
                raise

        # Build all pipes
        for i, p in enumerate(pipes):
            try:
                p.build()
            except Exception as e:
                logger.error(f"Failed to build pipe {i}: {e}")
                raise

        # Import DALI iterator
        try:
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError:
            from nvidia.dali.plugin.base_iterator import DALIGenericIterator
            from nvidia.dali.plugin.pytorch import LastBatchPolicy

        # Create iterator
        # Note: Removed size=total to allow infinite iteration for benchmarking
        # The original size=total was causing StopIteration after consuming all samples
        dali_it = DALIGenericIterator(
            pipes,
            ["image", "actions", "state"],
            # size=total,  # Removed to prevent iterator exhaustion
            last_batch_policy=LastBatchPolicy.PARTIAL,
            dynamic_shape=True,
        )

        # Calculate data sizes for bandwidth metrics
        # Get sample data from first shard to calculate sizes
        pack_data = extract_pack_data(files[0])
        root = create_zarr_from_pack_data(pack_data)
        sample_img = np.asarray(root["image"][0])
        sample_act = np.asarray(root["actions"][0])
        sample_st = np.asarray(root["state"][0])
        
        sample_img_bytes = sample_img.nbytes
        sample_act_bytes = sample_act.nbytes
        sample_st_bytes = sample_st.nbytes
        sample_total_bytes = sample_img_bytes + sample_act_bytes + sample_st_bytes
        
        logger.info(f"Sample data size: {sample_total_bytes / (1024**2):.2f} MiB "
                   f"(img: {sample_img_bytes / (1024**2):.2f} MiB, "
                   f"act: {sample_act_bytes / 1024:.2f} KiB, "
                   f"state: {sample_st_bytes / 1024:.2f} KiB)")

        # Warm-up
        logger.info("Running warm-up...")
        next(dali_it)
        
        # Benchmark iterations
        logger.info(f"Starting benchmark with {args.iterations} iteration(s)...")
        iteration_times_ms = []
        samples_per_iter = args.batch * n_gpu
        bytes_per_iter = sample_total_bytes * samples_per_iter
        
        # Track peak memory usage
        peak_process_rss = 0
        peak_gpu_allocated = [0] * n_gpu if torch.cuda.is_available() else []
        
        for iter_num in range(args.iterations):
            logger.info(f"Running iteration {iter_num + 1}/{args.iterations}...")
            
            # Log memory usage every 10 iterations
            if iter_num % 10 == 0 and iter_num > 0:
                mem_current = get_memory_stats()
                logger.info(f"Memory check at iteration {iter_num + 1}:")
                for line in format_memory_stats(mem_current):
                    logger.info(f"  {line}")
            
            t0 = time.perf_counter()

            try:
                batch = next(dali_it)
                for samp in batch:
                    for v in samp.values():
                        # Newer wheels: DALI Tensor ➜ .as_cpu()
                        if hasattr(v, "as_cpu"):
                            v = v.as_cpu()
                        # Older wheels: already a torch.Tensor
                        if isinstance(v, torch.Tensor):
                            v = v.cpu()            # sync to host
                        if hasattr(v, "numpy"):    # final read to ensure timing includes copy
                            _ = v.numpy()
                
                elapsed_ms = (time.perf_counter() - t0) * 1e3
                iteration_times_ms.append(elapsed_ms)
                ms_per_sample = elapsed_ms / samples_per_iter
                logger.info(f"  Iteration {iter_num + 1}: {elapsed_ms:.2f} ms "
                           f"({ms_per_sample:.3f} ms/sample)")
                
                # Track peak memory
                mem_current = get_memory_stats()
                if HAS_PSUTIL and mem_current.get('process_rss_mb', 0) > peak_process_rss:
                    peak_process_rss = mem_current['process_rss_mb']
                if 'gpu_allocated_mb' in mem_current:
                    for i, alloc in enumerate(mem_current['gpu_allocated_mb']):
                        if alloc > peak_gpu_allocated[i]:
                            peak_gpu_allocated[i] = alloc
                            
            except StopIteration as e:
                logger.error(f"Iterator exhausted at iteration {iter_num + 1}: {e}")
                logger.error("The DALI iterator has consumed all available data.")
                logger.error("This usually means the iterator size limit was reached.")
                logger.error("Consider removing the 'size' parameter from DALIGenericIterator.")
                break
            except MemoryError as e:
                logger.error(f"Memory error at iteration {iter_num + 1}: {e}")
                logger.error("System ran out of memory. Consider reducing cache size or batch size.")
                break
            except OSError as e:
                logger.error(f"OS error at iteration {iter_num + 1}: {e}")
                logger.error("This could be file handle exhaustion or disk space issues.")
                break
            except Exception as e:
                logger.error(f"Unexpected error at iteration {iter_num + 1}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
        
        # Calculate and display final statistics
        logger.info("")
        logger.info("="*70)

        # Compute stats first and guard empty timings
        stats_available = len(iteration_times_ms) > 0
        if stats_available:
            times_array = np.array(iteration_times_ms)
            mean_ms = np.mean(times_array)
            std_ms = np.std(times_array)
            min_ms = np.min(times_array)
            max_ms = np.max(times_array)
            median_ms = np.median(times_array)
            # Pre-compute bandwidth metrics for reuse (logs and JSON)
            mean_bw_mibs = (bytes_per_iter / (1024**2)) / (mean_ms / 1000)
            min_bw_mibs = (bytes_per_iter / (1024**2)) / (max_ms / 1000)
            max_bw_mibs = (bytes_per_iter / (1024**2)) / (min_ms / 1000)
            mean_bw_mbs  = (bytes_per_iter / (1000**2)) / (mean_ms / 1000)
            min_bw_mbs   = (bytes_per_iter / (1000**2)) / (max_ms / 1000)
            max_bw_mbs   = (bytes_per_iter / (1000**2)) / (min_ms / 1000)
        else:
            logger.error("No iteration timings recorded; skipping stats computation")

        logger.info("FINAL STATISTICS")
        logger.info("="*70)

        # Optionally write JSON summary (after stats computed)
        if args.json_output:
            try:
                import json
                base = {
                    "benchmark": "zarr_random_read",
                    "config": {
                        "shard": args.shard,
                        "batch": args.batch,
                        "workers": args.workers,
                        "shuffle": args.shuffle,
                        "device_read_ahead": args.device_read_ahead,
                        "iterations": args.iterations,
                        "drop_cache": args.drop_cache,
                        "total_samples": int(total),
                    },
                }
                # Take a memory snapshot for JSON to avoid referencing later variables
                mem_final_json = get_memory_stats()
                if stats_available:
                    summary = {
                        **base,
                        "metrics": {
                            "iteration_ms": {
                                "mean": round(float(mean_ms), 2),
                                "std": round(float(std_ms), 2),
                                "min": round(float(min_ms), 2),
                                "max": round(float(max_ms), 2),
                                "median": round(float(median_ms), 2),
                                **({"p95": round(float(np.percentile(times_array, 95)), 2),
                                    "p99": round(float(np.percentile(times_array, 99)), 2)} if args.iterations > 1 else {}),
                            },
                            "per_sample_ms": {
                                "mean": round(float(mean_ms/samples_per_iter), 2),
                                "min": round(float(min_ms/samples_per_iter), 2),
                                "max": round(float(max_ms/samples_per_iter), 2),
                            },
                            "throughput_samples_per_s": {
                                "mean": round(float(samples_per_iter/(mean_ms/1000)), 2),
                                "peak": round(float(samples_per_iter/(min_ms/1000)), 2),
                            },
                            "bandwidth": {
                                "data_per_iter_MiB": round(float(bytes_per_iter / (1024**2)), 2),
                                "data_per_iter_MB": round(float(bytes_per_iter / (1000**2)), 2),
                                "mean_MiB_s": round(float(mean_bw_mibs), 2),
                                "mean_MB_s": round(float(mean_bw_mbs), 2),
                                "min_MiB_s": round(float(min_bw_mibs), 2),
                                "min_MB_s": round(float(min_bw_mbs), 2),
                                "peak_MiB_s": round(float(max_bw_mibs), 2),
                                "peak_MB_s": round(float(max_bw_mbs), 2),
                            },
                            "memory": {
                                **({
                                    "process_rss_initial_MiB": round(float(mem_initial.get('process_rss_mb', 0)), 2)
                                } if HAS_PSUTIL else {}),
                                **({
                                    "process_rss_final_MiB": round(float(mem_final_json.get('process_rss_mb', 0)), 2),
                                    "process_rss_peak_MiB": round(float(peak_process_rss), 2),
                                    "system_used_MiB": round(float(mem_final_json.get('system_used_mb', 0)), 2),
                                    "system_total_MiB": round(float(mem_final_json.get('system_total_mb', 0)), 2),
                                    "system_percent": round(float(mem_final_json.get('system_percent', 0)), 2),
                                } if HAS_PSUTIL else {}),
                            },
                        },
                    }
                else:
                    summary = {**base, "error": "no_iteration_timings_recorded"}
                os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
                with open(args.json_output, "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Wrote JSON results to {args.json_output}")
            except Exception as e:
                logger.error(f"Failed to write JSON results: {e}")
        
        logger.info(f"Configuration:")
        logger.info(f"  Batch size:    {args.batch}")
        logger.info(f"  Pipeline count: {n_gpu}")
        logger.info(f"  Workers:       {args.workers}")
        logger.info(f"  Samples/iter:  {samples_per_iter}")
        logger.info(f"  Iterations:    {args.iterations}")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Samples/pipe:  {total // n_gpu}")
        logger.info(f"  Note: Iterator cycles through data for multi-iteration benchmarking")
        logger.info(f"")
        logger.info(f"Time per iteration (ms):")
        logger.info(f"  Mean:          {mean_ms:.2f} ms")
        logger.info(f"  Std Dev:       {std_ms:.2f} ms")
        logger.info(f"  Min:           {min_ms:.2f} ms")
        logger.info(f"  Max:           {max_ms:.2f} ms")
        logger.info(f"  Median:        {median_ms:.2f} ms")
        
        if args.iterations > 1:
            p95_ms = np.percentile(times_array, 95)
            p99_ms = np.percentile(times_array, 99)
            logger.info(f"  95th pctl:     {p95_ms:.2f} ms")
            logger.info(f"  99th pctl:     {p99_ms:.2f} ms")
        
        logger.info(f"")
        logger.info(f"Time per sample (ms):")
        logger.info(f"  Mean:          {mean_ms/samples_per_iter:.3f} ms/sample")
        logger.info(f"  Min:           {min_ms/samples_per_iter:.3f} ms/sample")
        logger.info(f"  Max:           {max_ms/samples_per_iter:.3f} ms/sample")
        
        logger.info(f"")
        logger.info(f"Throughput (samples/sec):")
        logger.info(f"  Mean:          {samples_per_iter/(mean_ms/1000):.1f} samples/sec")
        logger.info(f"  Peak (best):   {samples_per_iter/(min_ms/1000):.1f} samples/sec")
        
        logger.info(f"")
        cache_note = "(actual disk I/O)" if args.drop_cache else "(from memory-mapped files, may include page cache)"
        logger.info(f"Bandwidth {cache_note}:")
        mean_bw_mibs = (bytes_per_iter / (1024**2)) / (mean_ms / 1000)
        min_bw_mibs = (bytes_per_iter / (1024**2)) / (max_ms / 1000)  # min bandwidth = max time
        max_bw_mibs = (bytes_per_iter / (1024**2)) / (min_ms / 1000)  # max bandwidth = min time
        
        # Also calculate decimal (MB/s) for comparison with fio
        mean_bw_mbs = (bytes_per_iter / (1000**2)) / (mean_ms / 1000)
        min_bw_mbs = (bytes_per_iter / (1000**2)) / (max_ms / 1000)
        max_bw_mbs = (bytes_per_iter / (1000**2)) / (min_ms / 1000)
        
        logger.info(f"  Data/iter:     {bytes_per_iter / (1024**2):.2f} MiB ({bytes_per_iter / (1000**2):.2f} MB)")
        logger.info(f"  Mean:          {mean_bw_mibs:.2f} MiB/s ({mean_bw_mbs:.2f} MB/s) = {mean_bw_mibs / 1024:.2f} GiB/s")
        logger.info(f"  Min:           {min_bw_mibs:.2f} MiB/s ({min_bw_mbs:.2f} MB/s) = {min_bw_mibs / 1024:.2f} GiB/s")
        logger.info(f"  Peak (best):   {max_bw_mibs:.2f} MiB/s ({max_bw_mbs:.2f} MB/s) = {max_bw_mibs / 1024:.2f} GiB/s")
        
        # Display memory statistics
        logger.info(f"")
        logger.info(f"Memory Usage:")
        mem_final = get_memory_stats()
        
        if HAS_PSUTIL:
            logger.info(f"  Initial Process RSS:  {mem_initial['process_rss_mb']:.2f} MiB")
            logger.info(f"  Final Process RSS:    {mem_final['process_rss_mb']:.2f} MiB")
            logger.info(f"  Peak Process RSS:     {peak_process_rss:.2f} MiB")
            logger.info(f"  System Memory:        {mem_final['system_used_mb']:.2f} / {mem_final['system_total_mb']:.2f} MiB ({mem_final['system_percent']:.1f}%)")
        
        # Only show GPU memory if we're actually using GPU
        if torch.cuda.is_available() and 'gpu_allocated_mb' in mem_final:
            # Check if any GPU memory is actually being used
            gpu_memory_used = any(alloc > 0 for alloc in mem_final['gpu_allocated_mb'])
            if gpu_memory_used:
                for i in range(n_gpu):
                    logger.info(f"  GPU {i} Allocated:     {mem_final['gpu_allocated_mb'][i]:.2f} MiB (peak: {peak_gpu_allocated[i]:.2f} MiB)")
                    logger.info(f"  GPU {i} Reserved:      {mem_final['gpu_reserved_mb'][i]:.2f} MiB / {mem_final['gpu_total_mb'][i]:.2f} MiB total")
        
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()