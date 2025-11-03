#!/bin/bash
#
# test_mpi_fix.sh - Test the MPI-aware benchmark with proper synchronization
#

set -e  # Exit on error

echo "=================================="
echo "Testing MPI-Aware Benchmark Fix"
echo "=================================="
echo ""

# Clean up old results
echo "📁 Cleaning up old results..."
rm -f results/numpy_2nodes*.json
echo ""

# Run the benchmark with 2 MPI ranks
echo "🚀 Running benchmark with 2 MPI ranks..."
echo "   Command: mpirun -np 2 -hostfile hosts.txt python3 numpy_random_read/dali_random_read_numpy.py ..."
echo ""

timeout 120 mpirun -np 2 -hostfile hosts.txt \
  /home/liran/venv/bin/python3 \
  /home/liran/dali_dataloader_tests/numpy_random_read/dali_random_read_numpy.py \
  --shard "/mnt/test/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 32 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 10 \
  --json-output results/numpy_2nodes.json

echo ""
echo "✅ Benchmark completed successfully!"
echo ""

# Verify the results
echo "🔍 Verifying results..."
echo ""

if [ -f "results/numpy_2nodes_rank0.json" ]; then
    echo "✓ Found results/numpy_2nodes_rank0.json"
else
    echo "✗ Missing results/numpy_2nodes_rank0.json"
    exit 1
fi

if [ -f "results/numpy_2nodes_rank1.json" ]; then
    echo "✓ Found results/numpy_2nodes_rank1.json"
else
    echo "✗ Missing results/numpy_2nodes_rank1.json"
    exit 1
fi

if [ -f "results/numpy_2nodes.json" ]; then
    echo "✓ Found results/numpy_2nodes.json (combined)"
else
    echo "✗ Missing results/numpy_2nodes.json"
    exit 1
fi

echo ""
echo "📊 Running detailed verification..."
python3 verify_mpi_results.py results/numpy_2nodes.json

echo ""
echo "=================================="
echo "✅ All tests passed!"
echo "=================================="

