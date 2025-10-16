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
python3 create_synthetic_data.py --num-shards 50

# Basic benchmark with default settings
python3 dali_random_read_numpy.py \
  --shard "/mnt/test/shards_numpy/*/image.npy"

# Full benchmark with all parameters
python3 dali_random_read_numpy.py \
  --shard "/mnt/test/shards_numpy/*/image.npy" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 5 \
  --drop-cache \
  --verbose
```

### 2. **Binary (64KB) Random-Read Testing**
```bash
cd binary_random_read

# Create 45 shards with 64KB fixed-size records (~500MB each)
python3 create_synthetic_data.py --num-shards 45

# Basic benchmark with default settings
python3 dali_random_read_bin.py \
  --shard "/mnt/test/shards_64k/*/data.bin"

# Benchmark with O_DIRECT for true disk I/O performance
python3 dali_random_read_bin.py \
  --shard "/mnt/test/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --direct \
  --device-read-ahead 2 \
  --iterations 5 \
  --drop-cache \
  --verbose

# Test with OS page cache (no O_DIRECT)
python3 dali_random_read_bin.py \
  --shard "/mnt/test/shards_64k/*/data.bin" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --iterations 3
```

### 3. **Zarr Compressed Testing**
```bash
cd zarr_random_read

# Create 50 shards with Zarr compressed format (~0.7 compression ratio)
python3 create_synthetic_data.py --num-shards 50

# Basic benchmark with default settings
python3 dali_random_read_zarr.py \
  --shard "/mnt/test/shards2/00000000/*/steps.pack"

# Full benchmark with all parameters
python3 dali_random_read_zarr.py \
  --shard "/mnt/test/shards2/00000000/*/steps.pack" \
  --batch 256 \
  --workers 16 \
  --shuffle \
  --device-read-ahead 2 \
  --iterations 5 \
  --drop-cache \
  --verbose
```

## Complete CLI Parameters Reference

### NumPy Random-Read Script (`dali_random_read_numpy.py`)
```bash
python3 dali_random_read_numpy.py \
  --shard "/path/to/shards/*/image.npy" \     # Required: Glob pattern for .npy files
  --batch 256 \                               # Batch size per GPU (default: 256)
  --workers 4 \                               # CPU worker threads (default: 4)
  --shuffle \                                 # Shuffle global indices (optional)
  --device-read-ahead 2 \                     # GPU prefetch queue depth (default: 2)
  --iterations 1 \                            # Number of benchmark iterations (default: 1)
  --drop-cache \                              # Drop page cache before benchmark (requires sudo)
  --verbose                                   # Enable verbose logging (optional)
```

### Binary Random-Read Script (`dali_random_read_bin.py`)
```bash
python3 dali_random_read_bin.py \
  --shard "/path/to/shards/*/data.bin" \      # Required: Glob pattern for .bin files
  --batch 256 \                               # Batch size per GPU (default: 256)
  --workers 4 \                               # CPU worker threads (default: 4)
  --shuffle \                                 # Shuffle global indices (optional)
  --direct \                                  # Enable O_DIRECT for bypassing OS cache (optional)
  --device-read-ahead 2 \                     # GPU prefetch queue depth (default: 2)
  --iterations 1 \                            # Number of benchmark iterations (default: 1)
  --drop-cache \                              # Drop page cache before benchmark (requires sudo)
  --verbose                                   # Enable verbose logging (optional)
```

### Zarr Random-Read Script (`dali_random_read_zarr.py`)
```bash
python3 dali_random_read_zarr.py \
  --shard "/path/to/shards/*/steps.pack" \    # Required: Glob pattern for .pack files
  --batch 256 \                               # Batch size per GPU (default: 256)
  --workers 4 \                               # CPU worker threads (default: 4)
  --shuffle \                                 # Shuffle global indices (optional)
  --device-read-ahead 1 \                     # GPU prefetch queue depth (default: 1)
  --iterations 1 \                            # Number of benchmark iterations (default: 1)
  --drop-cache \                              # Drop page cache before benchmark (requires sudo)
  --verbose                                   # Enable verbose logging (optional)
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

### Required Parameters
- `--shard`: Glob pattern for shard files (e.g., `/mnt/weka/shards/*/data.bin`)

### Performance Tuning Parameters
- `--batch`: Batch size per GPU (default: 256)
- `--workers`: Number of CPU worker threads (default: 4)
- `--device-read-ahead`: GPU prefetch queue depth (default: 1 for Zarr, 2 for NumPy/Binary)
- `--shuffle`: Shuffle global indices before distributing to GPUs

### Benchmarking Parameters
- `--iterations`: Number of benchmark iterations to run (default: 1)
- `--drop-cache`: Drop page cache before benchmark (requires sudo privileges)
- `--direct`: Enable O_DIRECT for bypassing OS cache (binary tests only)

### Debugging Parameters
- `--verbose` or `-v`: Enable verbose logging for debugging

### Parameter-Specific Notes
- **NumPy/Zarr scripts**: Support `--device-read-ahead` for GPU prefetch optimization
- **Binary script**: Supports `--direct` flag for O_DIRECT I/O testing
- **All scripts**: Support `--drop-cache` for measuring actual disk performance vs cached performance

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