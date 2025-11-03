#!/usr/bin/env python3
"""
mpi_utils.py
------------
MPI utilities for distributed DALI benchmarking across multiple nodes.

Provides:
- MPI initialization and coordination
- Metrics gathering from all ranks
- Configuration broadcasting
- Rank-specific logging
- Hostname and node information
"""

import os
import socket
import logging
from typing import Optional, Tuple, Any, Dict
from pathlib import Path

# Try to import MPI, but allow fallback to single-node mode
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


def init_mpi() -> Tuple[Optional[Any], int, int]:
    """
    Initialize MPI environment and return communicator, rank, and world size.
    
    Returns:
        Tuple[comm, rank, size] where:
            - comm: MPI communicator (or None if MPI not available)
            - rank: Current rank (0 if MPI not available)
            - size: Total number of ranks (1 if MPI not available)
    
    If MPI is not available, returns (None, 0, 1) for single-node operation.
    """
    if not HAS_MPI:
        logging.warning("MPI not available, running in single-node mode")
        return None, 0, 1
    
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return comm, rank, size
    except Exception as e:
        logging.error(f"Failed to initialize MPI: {e}")
        logging.warning("Falling back to single-node mode")
        return None, 0, 1


def is_master(rank: int = 0) -> bool:
    """
    Check if current process is the master rank (rank 0).
    
    Args:
        rank: Current rank number
        
    Returns:
        True if rank is 0, False otherwise
    """
    return rank == 0


def get_rank_hostname() -> str:
    """
    Get the hostname of the current rank's node.
    
    Returns:
        Hostname as a string
    """
    return socket.gethostname()


def get_mpi_library_info() -> Dict[str, Any]:
    """
    Get information about the MPI library being used.
    
    Returns:
        Dictionary with MPI library information
    """
    if not HAS_MPI:
        return {
            "available": False,
            "library": "None",
            "version": "N/A"
        }
    
    try:
        version = MPI.Get_version()
        library_version = MPI.Get_library_version()
        
        # Try to parse library name from version string
        library_name = "Unknown"
        if "Open MPI" in library_version or "OpenMPI" in library_version:
            library_name = "OpenMPI"
        elif "MPICH" in library_version:
            library_name = "MPICH"
        elif "Intel" in library_version:
            library_name = "Intel MPI"
        
        return {
            "available": True,
            "library": library_name,
            "version": f"{version[0]}.{version[1]}",
            "full_version": library_version.split('\n')[0]  # First line only
        }
    except Exception as e:
        return {
            "available": True,
            "library": "Unknown",
            "version": "Unknown",
            "error": str(e)
        }


def barrier(comm: Optional[Any]) -> None:
    """
    Synchronize all ranks at a barrier.
    
    Args:
        comm: MPI communicator (None for single-node mode)
    """
    if comm is not None:
        comm.Barrier()


def gather_metrics(local_metrics: Dict[str, Any], comm: Optional[Any], root: int = 0) -> Optional[list]:
    """
    Gather metrics from all ranks to the root rank.
    
    Args:
        local_metrics: Dictionary containing metrics from current rank
        comm: MPI communicator (None for single-node mode)
        root: Rank to gather data to (default: 0)
        
    Returns:
        List of metrics from all ranks (only on root rank), None on other ranks
        In single-node mode, returns list with single element
    """
    if comm is None:
        # Single-node mode: return list with single element
        return [local_metrics]
    
    try:
        all_metrics = comm.gather(local_metrics, root=root)
        return all_metrics
    except Exception as e:
        logging.error(f"Failed to gather metrics: {e}")
        return None


def broadcast_config(config: Optional[Dict[str, Any]], comm: Optional[Any], root: int = 0) -> Dict[str, Any]:
    """
    Broadcast configuration from root rank to all other ranks.
    
    Args:
        config: Configuration dictionary (only meaningful on root rank)
        comm: MPI communicator (None for single-node mode)
        root: Rank broadcasting the data (default: 0)
        
    Returns:
        Configuration dictionary on all ranks
    """
    if comm is None:
        # Single-node mode: just return the config
        return config if config is not None else {}
    
    try:
        config = comm.bcast(config, root=root)
        return config
    except Exception as e:
        logging.error(f"Failed to broadcast config: {e}")
        return {}


def setup_rank_logging(rank: int, log_dir: Optional[str] = None, verbose: bool = False) -> None:
    """
    Configure per-rank logging with appropriate formatting.
    
    Args:
        rank: Current rank number
        log_dir: Directory for log files (None for no file logging)
        verbose: Enable verbose (DEBUG level) logging
    """
    hostname = get_rank_hostname()
    
    # Create formatter with rank and hostname
    log_format = f'%(asctime)s - [Rank {rank} @ {hostname}] - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir_path / f"rank_{rank}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def assign_shards_to_rank(all_shards: list, rank: int, world_size: int) -> list:
    """
    Assign a subset of shards to the current rank using contiguous block assignment.
    
    This ensures each rank gets an approximately equal number of shards in a
    contiguous block, which can help with filesystem caching and prefetching.
    
    Args:
        all_shards: List of all available shard paths
        rank: Current rank number
        world_size: Total number of ranks
        
    Returns:
        List of shard paths assigned to this rank
    """
    if world_size <= 0:
        raise ValueError(f"Invalid world_size: {world_size}")
    
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}")
    
    total_shards = len(all_shards)
    
    if total_shards == 0:
        return []
    
    # Calculate shards per rank (base allocation)
    shards_per_rank = total_shards // world_size
    remainder = total_shards % world_size
    
    # Distribute remainder shards to first 'remainder' ranks
    if rank < remainder:
        start_idx = rank * (shards_per_rank + 1)
        end_idx = start_idx + shards_per_rank + 1
    else:
        start_idx = remainder * (shards_per_rank + 1) + (rank - remainder) * shards_per_rank
        end_idx = start_idx + shards_per_rank
    
    return all_shards[start_idx:end_idx]


def get_local_gpu_count() -> int:
    """
    Get the number of GPUs available on the current node.
    
    Returns:
        Number of local GPUs (0 if none available or error)
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except Exception as e:
        logging.warning(f"Failed to detect GPU count: {e}")
        return 0


def print_mpi_info(comm: Optional[Any], rank: int, world_size: int) -> None:
    """
    Print MPI configuration information (only from rank 0).
    
    Args:
        comm: MPI communicator
        rank: Current rank
        world_size: Total number of ranks
    """
    if not is_master(rank):
        return
    
    mpi_info = get_mpi_library_info()
    
    print("=" * 70)
    print("MPI Configuration:")
    print(f"  MPI Available:     {mpi_info['available']}")
    print(f"  MPI Library:       {mpi_info.get('library', 'N/A')}")
    print(f"  MPI Version:       {mpi_info.get('version', 'N/A')}")
    print(f"  Total Ranks:       {world_size}")
    print(f"  World Size:        {world_size}")
    
    # Gather hostnames from all ranks
    if comm is not None:
        hostname = get_rank_hostname()
        all_hostnames = comm.gather(hostname, root=0)
        
        if all_hostnames:
            unique_hosts = sorted(set(all_hostnames))
            print(f"  Unique Nodes:      {len(unique_hosts)}")
            print(f"  Nodes:             {', '.join(unique_hosts)}")
            
            # Count ranks per node
            from collections import Counter
            host_counts = Counter(all_hostnames)
            print(f"  Ranks per Node:    ", end="")
            print(", ".join([f"{host}: {count}" for host, count in sorted(host_counts.items())]))
        
        # Gather GPU counts
        local_gpu_count = get_local_gpu_count()
        all_gpu_counts = comm.gather(local_gpu_count, root=0)
        
        if all_gpu_counts:
            total_gpus = sum(all_gpu_counts)
            print(f"  Total GPUs:        {total_gpus}")
            print(f"  GPUs per Rank:     {', '.join([f'Rank {i}: {c}' for i, c in enumerate(all_gpu_counts)])}")
    else:
        print(f"  Unique Nodes:      1 (single-node mode)")
        hostname = get_rank_hostname()
        print(f"  Node:              {hostname}")
        local_gpus = get_local_gpu_count()
        print(f"  Local GPUs:        {local_gpus}")
    
    print("=" * 70)


def finalize_mpi() -> None:
    """
    Finalize MPI (should be called at program exit if MPI was used).
    
    Note: mpi4py typically handles finalization automatically, but this
    can be called explicitly if needed.
    """
    if HAS_MPI and MPI.Is_initialized() and not MPI.Is_finalized():
        try:
            MPI.Finalize()
        except Exception as e:
            logging.warning(f"Error during MPI finalization: {e}")


# Context manager for MPI operations
class MPIContext:
    """
    Context manager for MPI operations.
    
    Usage:
        with MPIContext() as (comm, rank, size):
            # Your MPI code here
            pass
    """
    
    def __init__(self, log_dir: Optional[str] = None, verbose: bool = False):
        """
        Initialize MPI context.
        
        Args:
            log_dir: Directory for per-rank log files
            verbose: Enable verbose logging
        """
        self.log_dir = log_dir
        self.verbose = verbose
        self.comm = None
        self.rank = 0
        self.size = 1
    
    def __enter__(self):
        """Enter context: initialize MPI"""
        self.comm, self.rank, self.size = init_mpi()
        setup_rank_logging(self.rank, self.log_dir, self.verbose)
        return self.comm, self.rank, self.size
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: finalize MPI if needed"""
        # MPI finalization is handled automatically by mpi4py
        pass


if __name__ == "__main__":
    # Test MPI utilities
    print("Testing MPI utilities...")
    
    comm, rank, size = init_mpi()
    setup_rank_logging(rank, verbose=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initialized MPI: rank={rank}, size={size}")
    logger.info(f"Hostname: {get_rank_hostname()}")
    logger.info(f"Local GPUs: {get_local_gpu_count()}")
    
    # Test barrier
    logger.info("Testing barrier...")
    barrier(comm)
    logger.info("Barrier passed")
    
    # Test metrics gathering
    local_metrics = {
        "rank": rank,
        "hostname": get_rank_hostname(),
        "test_value": rank * 100
    }
    
    all_metrics = gather_metrics(local_metrics, comm)
    
    if is_master(rank):
        logger.info("Gathered metrics from all ranks:")
        for i, metrics in enumerate(all_metrics):
            logger.info(f"  Rank {i}: {metrics}")
    
    # Test shard assignment
    if is_master(rank):
        all_shards = [f"shard_{i:03d}" for i in range(50)]
        logger.info(f"Total shards: {len(all_shards)}")
        
        for r in range(size):
            assigned = assign_shards_to_rank(all_shards, r, size)
            logger.info(f"  Rank {r}: {len(assigned)} shards: {assigned[:3]}...")
    
    # Print MPI info
    print_mpi_info(comm, rank, size)
    
    logger.info("Test complete")






