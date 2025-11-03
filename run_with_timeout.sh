#!/bin/bash
#
# run_with_timeout.sh - Run MPI benchmark with timeout and debugging
#

TIMEOUT=${1:-60}  # Default 60 seconds

echo "=================================="
echo "Running MPI Benchmark with ${TIMEOUT}s timeout"
echo "=================================="
echo ""

# Clean old results
echo "🧹 Cleaning old results..."
rm -f results/numpy_2nodes*.json
echo ""

# Run with timeout
echo "🚀 Starting benchmark..."
echo ""

timeout --kill-after=5s ${TIMEOUT}s mpirun -np 2 -hostfile hosts.txt \
  /home/liran/venv/bin/python3 \
  /home/liran/dali_dataloader_tests/numpy_random_read/dali_random_read_numpy.py \
  --shard "/mnt/test/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 32 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 10 \
  --json-output results/numpy_2nodes.json 2>&1 | tee /tmp/mpi_benchmark_output.log

EXIT_CODE=$?

echo ""
echo "=================================="

if [ $EXIT_CODE -eq 124 ]; then
    echo "❌ TIMEOUT after ${TIMEOUT} seconds!"
    echo ""
    echo "📋 Last 30 lines of output:"
    tail -30 /tmp/mpi_benchmark_output.log
    echo ""
    echo "💡 Checking for hung processes..."
    pgrep -af "dali_random_read_numpy.py" || echo "   No processes found"
    exit 124
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    echo ""
    echo "📊 Checking results..."
    ls -lh results/numpy_2nodes*.json 2>/dev/null | wc -l | xargs echo "   Found files:"
    ls results/numpy_2nodes*.json 2>/dev/null || echo "   No result files found!"
else
    echo "❌ Benchmark failed with exit code: $EXIT_CODE"
    echo ""
    echo "📋 Last 50 lines of output:"
    tail -50 /tmp/mpi_benchmark_output.log
fi

echo "=================================="

