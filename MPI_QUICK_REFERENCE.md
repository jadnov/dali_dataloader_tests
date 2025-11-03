# MPI Quick Reference Guide

## Installation

### Install MPI Library

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
```

**From Source (Latest OpenMPI):**
```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -xzf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
```

### Install mpi4py

```bash
# After MPI is installed
pip install mpi4py

# Verify installation
python -c "from mpi4py import MPI; print(MPI.Get_version())"
```

## Basic MPI Commands

### Launch MPI Job

**Local (single machine, multiple processes):**
```bash
mpirun -np 4 python script.py
# -np 4: Launch 4 MPI processes
```

**Multi-node (using hostfile):**
```bash
mpirun -np 8 -hostfile hosts.txt python script.py
```

**Multi-node (explicit hosts):**
```bash
mpirun -np 8 -H node1,node2,node3,node4 python script.py
```

### Hostfile Format

**hosts.txt:**
```
node-0 slots=2
node-1 slots=2
node-2 slots=2
node-3 slots=2
```

**Explanation:**
- `slots=2`: Maximum 2 MPI processes per node
- Total processes: Up to 8 (2 per node × 4 nodes)

### Binding and Mapping

**Bind to NUMA nodes:**
```bash
mpirun -np 8 --bind-to numa --map-by ppr:2:socket python script.py
```

**Bind to CPU cores:**
```bash
mpirun -np 8 --bind-to core --map-by core python script.py
```

**Display binding:**
```bash
mpirun -np 4 --report-bindings python script.py
```

## DALI MPI Benchmark Commands

### Binary Random Read (8 Nodes)

```bash
mpirun -np 8 -hostfile hosts.txt \
  python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 10 \
  --drop-cache \
  --json-output results/binary_8nodes.json
```

### NumPy Random Read (4 Nodes)

```bash
mpirun -np 4 -hostfile hosts.txt \
  python numpy_random_read/dali_random_read_numpy_mpi.py \
  --shard "/mnt/weka/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 32 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 10 \
  --json-output results/numpy_4nodes.json
```

### Zarr Random Read (16 Nodes)

```bash
mpirun -np 16 -hostfile hosts.txt \
  python zarr_random_read/dali_random_read_zarr_mpi.py \
  --shard "/mnt/weka/shards/*/steps.zarr.pack" \
  --batch 128 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 1 \
  --iterations 10 \
  --json-output results/zarr_16nodes.json
```

### Unified MPI Runner

```bash
# Auto-detect benchmark type from shard pattern
mpirun -np 8 -hostfile hosts.txt \
  python mpi_runner.py \
  --benchmark binary \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --iterations 10
```

## SLURM Integration

### SLURM Job Script

**job_mpi_benchmark.sh:**
```bash
#!/bin/bash
#SBATCH --job-name=dali_mpi_bench
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/dali_mpi_%j.out
#SBATCH --error=logs/dali_mpi_%j.err

# Load modules
module load openmpi/4.1.4
module load cuda/12.0

# Activate Python environment
source /path/to/venv/bin/activate

# Run benchmark
srun python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 32 \
  --iterations 10 \
  --json-output results/binary_slurm_${SLURM_JOB_ID}.json
```

**Submit:**
```bash
sbatch job_mpi_benchmark.sh
```

**Monitor:**
```bash
squeue -u $USER
tail -f logs/dali_mpi_*.out
```

## Environment Variables

### OpenMPI Tuning

```bash
# Increase message buffer size
export OMPI_MCA_btl_tcp_sndbuf=16777216
export OMPI_MCA_btl_tcp_rcvbuf=16777216

# Use InfiniBand if available
export OMPI_MCA_btl=self,openib

# Verbose output for debugging
export OMPI_MCA_verbose=1
```

### DALI Environment Variables

```bash
# DALI CPU threads
export DALI_NUM_THREADS=16

# DALI prefetch queue depth
export DALI_PREFETCH_QUEUE_DEPTH=2

# CUDA visible devices (per-rank)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Debugging MPI Issues

### Test MPI Installation

```bash
# Test local
mpirun -np 4 python -c "from mpi4py import MPI; print('Rank', MPI.COMM_WORLD.rank, 'of', MPI.COMM_WORLD.size)"

# Test multi-node
mpirun -np 8 -hostfile hosts.txt hostname
```

### Verbose MPI Execution

```bash
mpirun -np 8 -hostfile hosts.txt --verbose --debug-devel python script.py
```

### Check Connectivity

```bash
# Test SSH to all nodes
for node in node-{0..7}; do
  echo "Testing $node..."
  ssh $node hostname
done

# Test MPI connectivity
mpirun -np 8 -hostfile hosts.txt --display-map hostname
```

### Log Per-Rank Output

```bash
# Redirect stdout/stderr per rank
mpirun -np 8 -hostfile hosts.txt \
  --output-filename logs/rank \
  python script.py

# Creates logs/rank.0, logs/rank.1, ..., logs/rank.7
```

## Performance Tuning

### Optimal Process Placement

**1 Rank per Node (Recommended for GPU workloads):**
```bash
mpirun -np 8 -hostfile hosts.txt -npernode 1 python script.py
```

**2 Ranks per Node (For dual-socket systems):**
```bash
mpirun -np 16 -hostfile hosts.txt -npernode 2 \
  --bind-to numa --map-by ppr:1:socket \
  python script.py
```

### Network Optimization

**Use InfiniBand (if available):**
```bash
export OMPI_MCA_btl=self,openib
export OMPI_MCA_btl_openib_if_include=mlx5_0
```

**Use High-Speed Ethernet:**
```bash
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=eth1  # Specify fast network interface
```

### Storage Optimization

**For shared filesystems (NFS, WEKA, Lustre):**
```bash
# Ensure all nodes mount the same path
mpirun -np 8 -hostfile hosts.txt ls -la /mnt/weka/shards_64k/

# Test concurrent read performance
mpirun -np 8 -hostfile hosts.txt \
  dd if=/mnt/weka/test.bin of=/dev/null bs=1M count=1024
```

## Common MPI Patterns in Code

### Initialize MPI

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} of {size}")
```

### Gather Data to Root

```python
# Each rank has local data
local_data = {'rank': rank, 'value': rank * 10}

# Gather all data to rank 0
all_data = comm.gather(local_data, root=0)

if rank == 0:
    print(f"Gathered data from all ranks: {all_data}")
```

### Broadcast Configuration

```python
# Rank 0 has configuration
if rank == 0:
    config = {'batch': 256, 'workers': 16}
else:
    config = None

# Broadcast to all ranks
config = comm.bcast(config, root=0)

print(f"Rank {rank} received config: {config}")
```

### Synchronize Ranks

```python
# All ranks do their work
local_result = do_computation()

# Wait for all ranks to finish
comm.Barrier()

# Now all ranks are synchronized
if rank == 0:
    print("All ranks have finished computation")
```

### Reduce (Sum across ranks)

```python
# Each rank computes a local sum
local_sum = sum(range(rank * 100, (rank + 1) * 100))

# Sum across all ranks
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Global sum: {global_sum}")
```

## Monitoring and Profiling

### Resource Usage Per Node

```bash
# In separate terminal, SSH to each node and monitor
ssh node-0 'nvidia-smi dmon -s u -c 100'  # GPU utilization
ssh node-0 'iostat -x 5'                   # Disk I/O
ssh node-0 'iftop -i eth0'                 # Network traffic
```

### MPI Performance Profiling

**Using MPIP (MPI Profiler):**
```bash
# Install MPIP
# Build with profiling support

# Run with profiling
mpirun -np 8 -hostfile hosts.txt \
  env LD_PRELOAD=/path/to/libmpiP.so \
  python script.py
```

**Using Score-P:**
```bash
# Profile MPI communication
scorep mpirun -np 8 python script.py

# Analyze results
scorep-score scorep_*
```

## Cluster Setup Checklist

### Prerequisites

- [ ] MPI library installed on all nodes (same version)
- [ ] Python environment replicated on all nodes
- [ ] Passwordless SSH configured between all nodes
- [ ] Shared storage mounted at same path on all nodes
- [ ] Network connectivity verified (low latency preferred)
- [ ] GPU drivers and CUDA installed on all nodes
- [ ] Firewall rules allow MPI communication (ports 1024-65535)
- [ ] Same user account on all nodes

### Quick Verification

```bash
# 1. Test SSH
for node in node-{0..7}; do ssh $node hostname; done

# 2. Test MPI
mpirun -np 8 -hostfile hosts.txt hostname

# 3. Test Python/MPI
mpirun -np 8 -hostfile hosts.txt python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

# 4. Test shared storage
mpirun -np 8 -hostfile hosts.txt ls /mnt/weka/

# 5. Test GPU access
mpirun -np 8 -hostfile hosts.txt nvidia-smi -L
```

## Example Workflows

### Scaling Study (1, 2, 4, 8 Nodes)

```bash
# 1 Node
mpirun -np 1 python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" --batch 256 --iterations 10 \
  --json-output results/binary_1node.json

# 2 Nodes
mpirun -np 2 -hostfile hosts.txt python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" --batch 256 --iterations 10 \
  --json-output results/binary_2nodes.json

# 4 Nodes
mpirun -np 4 -hostfile hosts.txt python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" --batch 256 --iterations 10 \
  --json-output results/binary_4nodes.json

# 8 Nodes
mpirun -np 8 -hostfile hosts.txt python binary_random_read/dali_random_read_bin_mpi.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" --batch 256 --iterations 10 \
  --json-output results/binary_8nodes.json

# Analyze scaling efficiency
python analyze_scaling.py results/binary_*nodes.json
```

### Storage System Comparison

```bash
# Test NFS
mpirun -np 8 -hostfile hosts.txt python script.py --shard "/mnt/nfs/shards/*/data.bin"

# Test WEKA
mpirun -np 8 -hostfile hosts.txt python script.py --shard "/mnt/weka/shards/*/data.bin"

# Test Lustre
mpirun -np 8 -hostfile hosts.txt python script.py --shard "/mnt/lustre/shards/*/data.bin"
```

## Troubleshooting

### "Cannot find mpirun"
```bash
# Add MPI to PATH
export PATH=/usr/lib64/openmpi/bin:$PATH
# Or
module load openmpi
```

### "Permission denied" on SSH
```bash
# Setup passwordless SSH
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
for node in node-{0..7}; do
  ssh-copy-id $node
done
```

### "No shards found"
```bash
# Verify shared storage is mounted
mpirun -np 8 -hostfile hosts.txt ls -la /mnt/weka/shards_64k/
```

### Hanging at initialization
```bash
# Check firewall
sudo iptables -L
# Allow MPI ports
sudo iptables -A INPUT -p tcp --dport 1024:65535 -j ACCEPT
```

### Different results on different nodes
```bash
# Verify same Python environment
mpirun -np 8 -hostfile hosts.txt python --version
mpirun -np 8 -hostfile hosts.txt pip list | grep mpi4py
```

---

**For detailed implementation, see:** `MPI_IMPLEMENTATION_PLAN.md`

**For architecture details, see:** `MPI_ARCHITECTURE_SUMMARY.md`






