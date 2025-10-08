# DALI DataLoader Tests

This project is a comprehensive benchmarking suite for testing NVIDIA DALI (Data Loading Library) performance with different data formats and storage configurations. The project is designed to test random-read performance across various data storage patterns commonly used in machine learning workflows.

## Project Structure Overview

```
dali_dataloader_tests/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── binary_random_read/         # Fixed-size binary record tests (64KB)
├── numpy_random_read/          # NumPy array random-read tests
└── zarr_random_read/           # Zarr compressed array tests
```

## Key Technical Features

### 1. **Multi-Format Testing**
- **NumPy**: Memory-mapped arrays with structured data (fastest for small random reads)
- **Binary**: Raw fixed-size 64KB records (pure I/O testing, minimal overhead)
- **Zarr**: Compressed arrays with custom packing (production-like with compression)

### 2. **Performance Optimization**
- **Multi-GPU support**: Automatic data sharding across multiple GPUs
- **Configurable workers**: CPU thread pool control for parallel data loading
- **Device read-ahead**: GPU prefetch queue depth for hiding latency
- **O_DIRECT**: Optional flag to bypass OS page cache for true disk performance measurement

### 3. **Realistic Workloads**
- **Random access patterns**: Simulates training data loading with shuffled indices
- **Large datasets**: 50+ shards, 500MB+ each (~25GB+ total)
- **Mixed data types**: Images (uint8), actions (float32), and state data (float32)
- **Compression**: Tests real-world storage efficiency trade-offs

## Usage Examples

### 1. **NumPy Random-Read Testing**
```bash
cd numpy_random_read

# Create 50 shards with synthetic data (default: 4000 steps, 2 cameras, 256x320 resolution)
python create_synthetic_data.py --num-shards 50

# Benchmark random reads from all shards
python dali_random_read_numpy.py \
  --shard "/mnt/weka/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 2
```

### 2. **Binary (64KB) Random-Read Testing**
```bash
cd binary_random_read

# Create 45 shards with 64KB fixed-size records (~500MB each)
python create_synthetic_data.py --num-shards 45

# Benchmark with O_DIRECT for true disk I/O performance
python dali_random_read_bin.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --direct

# Or test with OS page cache
python dali_random_read_bin.py \
  --shard "/mnt/weka/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle
```

### 3. **Zarr Compressed Testing**
```bash
cd zarr_random_read

# Create 50 shards with Zarr compressed format (~0.7 compression ratio)
python create_synthetic_data.py --num-shards 50

# Benchmark multi-shard random reads
python dali_random_read_zarr.py \
  --shard "/mnt/weka/shards2/00000000/*/steps.zarr.pack" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 2
```

## Project Purpose

This project serves as a comprehensive benchmarking suite for:

1. **Storage System Evaluation**: Tests different storage backends (e.g., WEKA filesystem, NFS, local storage)
2. **DALI Performance Tuning**: Optimizes data loading pipelines for machine learning training workflows
3. **Format Comparison**: Compares NumPy, binary, and Zarr performance characteristics for different access patterns
4. **Production Readiness**: Validates data loading performance under realistic conditions with multi-GPU setups

The project is particularly valuable for teams working with large-scale machine learning datasets that need to optimize data loading performance across different storage formats and configurations.

## Common Command-Line Options

All benchmark scripts support the following options:

- `--shard`: Glob pattern for shard files (e.g., `/mnt/weka/shards/*/data.bin`)
- `--batch`: Batch size per GPU (default: 256)
- `--workers`: Number of CPU worker threads (default: 4)
- `--shuffle`: Shuffle global indices before distributing to GPUs
- `--device-read-ahead`: GPU prefetch queue depth (default: 1, NumPy and Zarr only)
- `--direct`: Enable O_DIRECT for bypassing OS cache (binary tests only)
- `--verbose` or `-v`: Enable verbose logging for debugging

## Performance Considerations

- **NumPy**: Best for small to medium random reads with structured data. Memory-mapped files provide efficient caching.
- **Binary**: Best for testing pure storage I/O performance with fixed-size records. Use `--direct` to measure disk performance without cache effects.
- **Zarr**: Best for production scenarios with compression. Balances storage efficiency with read performance.

## Installation

```bash
pip install -r requirements.txt
```

Ensure you have:
- CUDA 12.0+ installed
- Compatible NVIDIA GPU(s)
- Sufficient storage space for test data (~25GB+ per full test suite)