#!/usr/bin/env python3
"""Test MPI initialization in the benchmark script"""

import sys
from pathlib import Path

print("=== MPI Initialization Test ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Script dir: {Path(__file__).parent}", flush=True)
print("", flush=True)

# Test 1: Import mpi_utils
print("Test 1: Importing mpi_utils...", flush=True)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from mpi_utils import init_mpi, get_rank_hostname
    print("✓ mpi_utils imported successfully", flush=True)
except Exception as e:
    print(f"✗ Failed to import mpi_utils: {e}", flush=True)
    sys.exit(1)

# Test 2: Initialize MPI
print("", flush=True)
print("Test 2: Initializing MPI...", flush=True)
try:
    comm, rank, world_size = init_mpi()
    hostname = get_rank_hostname()
    print(f"✓ MPI initialized successfully", flush=True)
    print(f"   Rank: {rank}", flush=True)
    print(f"   World Size: {world_size}", flush=True)
    print(f"   Hostname: {hostname}", flush=True)
except Exception as e:
    print(f"✗ Failed to initialize MPI: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if running under mpirun
print("", flush=True)
print("Test 3: Environment check...", flush=True)
mpi_env_vars = ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_RANK']
for var in mpi_env_vars:
    import os
    value = os.environ.get(var, 'not set')
    print(f"   {var}: {value}", flush=True)

print("", flush=True)
print(f"=== Test Complete: Rank {rank} of {world_size} ===", flush=True)

