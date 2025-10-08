#!/usr/bin/env python3
"""
dali_random_read_fixed.py
-------------------------
Random-read benchmark: one 64 KiB read per sample,
no mmap, no compression, optional O_DIRECT.
"""
from __future__ import annotations
import argparse, bisect, glob, json, os, random, sys, time, logging, atexit
from pathlib import Path

import numpy as np
import torch
from nvidia.dali import fn, pipeline_def, types

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except ImportError:
    from nvidia.dali.plugin.base_iterator import DALIGenericIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global resource tracking
_open_file_descriptors = []
_pipes = []

def cleanup_resources():
    """Clean up global resources"""
    global _open_file_descriptors, _pipes
    logger.info("Cleaning up resources...")
    
    # Close file descriptors
    for fd in _open_file_descriptors:
        try:
            os.close(fd)
        except Exception as e:
            logger.warning(f"Error closing file descriptor {fd}: {e}")
    _open_file_descriptors.clear()
    
    # Stop and clear pipes
    for pipe in _pipes:
        try:
            pipe.stop()
        except Exception as e:
            logger.warning(f"Error stopping pipe: {e}")
    _pipes.clear()
    
    logger.info("Resource cleanup completed")

# Register cleanup function
atexit.register(cleanup_resources)

# ── open all shards ──────────────────────────────────────────────────────────
def open_shards(pattern: str, direct: bool):
    """Open shards with proper error handling and resource tracking"""
    try:
        dirs = sorted(Path(p).parent for p in glob.glob(pattern))
        if not dirs:
            raise FileNotFoundError(f"No matches for pattern: {pattern!r}")
        
        logger.info(f"Found {len(dirs)} shard directories")
        
        flag = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        fds, rec_bytes, cum = [], 0, [0]
        
        for i, d in enumerate(dirs):
            try:
                logger.debug(f"Opening shard {i+1}/{len(dirs)}: {d}")
                
                # Check for required files
                meta_path = d / "meta.json"
                data_path = d / "data.bin"
                
                if not meta_path.exists():
                    raise FileNotFoundError(f"Missing metadata file: {meta_path}")
                if not data_path.exists():
                    raise FileNotFoundError(f"Missing data file: {data_path}")
                
                # Load metadata
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                rec_bytes = meta["record_bytes"]
                records = meta["records"]
                
                # Open file descriptor
                fd = os.open(data_path, flag)
                fds.append(fd)
                _open_file_descriptors.append(fd)  # Track for cleanup
                
                cum.append(cum[-1] + records)
                logger.debug(f"Opened shard {d}: {records} records, {rec_bytes} bytes each")
                
            except Exception as e:
                logger.error(f"Failed to open shard {d}: {e}")
                # Clean up any opened file descriptors
                for fd in fds:
                    try:
                        os.close(fd)
                        if fd in _open_file_descriptors:
                            _open_file_descriptors.remove(fd)
                    except:
                        pass
                raise
        
        logger.info(f"Successfully opened {len(fds)} shards with {cum[-1]} total records")
        return fds, rec_bytes, cum
        
    except Exception as e:
        logger.error(f"Failed to open shards: {e}")
        raise

# ── iterator ─────────────────────────────────────────────────────────────────
class ExtIter:
    def __init__(self, fds, rec_bytes, cum, idx, bs, rank, world):
        self.fds, self.rec_bytes, self.cum = fds, rec_bytes, cum
        s = len(idx)*rank//world; e = len(idx)*(rank+1)//world
        self.idx = idx[s:e];  self.bs = bs;  self.n = len(self.idx); self.i = 0

    def __iter__(self): self.i = 0; return self

    def _fetch(self, g):
        shard = bisect.bisect_right(self.cum, g)-1
        offset = (g - self.cum[shard]) * self.rec_bytes
        buf = os.pread(self.fds[shard], self.rec_bytes, offset)
        return np.frombuffer(buf, np.uint8)              # 64 KiB → (65536,)

    def __next__(self):
        if self.i >= self.n: raise StopIteration
        batch = [self._fetch(self.idx[(self.i+k)%self.n]) for k in range(self.bs)]
        self.i += self.bs
        return batch        

    next = __next__

# ── DALI pipeline ────────────────────────────────────────────────────────────
@pipeline_def
def pipe(eii):
    tensor = fn.external_source(source=eii, batch=True, dtype=types.UINT8)
    return tensor

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--shard", required=True,
            help='glob like "/mnt/weka/shards_64k/*/data.bin"')
        ap.add_argument("--batch", type=int, default=256)
        ap.add_argument("--workers", type=int, default=4)
        ap.add_argument("--shuffle", action="store_true")
        ap.add_argument("--direct", action="store_true", help="O_DIRECT bypass cache")
        ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args = ap.parse_args()

        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"Starting benchmark with args: {vars(args)}")

        # Open shards with error handling
        fds, rec_bytes, cum = open_shards(args.shard, args.direct)
        total = cum[-1]
        logger.info(f"📦 {len(fds)} shards → {total} samples  "
                   f"(record = {rec_bytes//1024} KiB, O_DIRECT={args.direct})")

        # Prepare indices
        idx = list(range(total))
        if args.shuffle: 
            random.shuffle(idx)
            logger.info("Shuffled global indices")

        # Setup GPU configuration
        n_gpu = max(1, torch.cuda.device_count())
        logger.info(f"Using {n_gpu} GPU(s)")

        # Create and build pipes
        pipes = []
        for gpu in range(n_gpu):
            try:
                eii = ExtIter(fds, rec_bytes, cum, idx, args.batch, gpu, n_gpu)
                pipe_instance = pipe(batch_size=args.batch,
                                   num_threads=args.workers,
                                   device_id=gpu,
                                   eii=eii)
                pipes.append(pipe_instance)
                _pipes.append(pipe_instance)  # Track for cleanup
            except Exception as e:
                logger.error(f"Failed to create pipe for GPU {gpu}: {e}")
                raise

        # Build all pipes
        for i, p in enumerate(pipes):
            try:
                p.build()
                logger.debug(f"Built pipe {i+1}/{len(pipes)}")
            except Exception as e:
                logger.error(f"Failed to build pipe {i}: {e}")
                raise

        # Create iterator
        it = DALIGenericIterator(pipes, ["record"], size=total,
                                 last_batch_policy=LastBatchPolicy.PARTIAL)

        # Warm-up
        logger.info("Running warm-up...")
        next(it)
        
        # Benchmark
        logger.info("Starting benchmark...")
        t0 = time.perf_counter()
        
        for sample in next(it):             # one timed batch
            v = sample["record"].cpu()
            _ = v.numpy()  # Ensure data is fully read
        
        ms = (time.perf_counter()-t0)*1e3
        logger.info(f"✅ {args.batch*n_gpu} samples in {ms:.1f} ms  "
                   f"→ {ms/(args.batch*n_gpu):.3f} ms/sample")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()