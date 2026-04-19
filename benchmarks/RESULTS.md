# Poema FMA Benchmark Results

## Configuration
- Hardware: CPU (benchmark)
- PyTorch: 2.8.0+cu126
- NumPy: 2.4.3

## Polynomial Evaluation (Φ-FMA vs NumPy vs PyTorch)

### Degree 10

| Batch Size | NumPy (ms) | Poema Φ-FMA (ms) | PyTorch Direct (ms) | Poema Error | Speedup vs NumPy |
|------------|------------|------------------|---------------------|-------------|------------------|
| 1000 | 0.034 | 0.045 | 0.121 | 1.39e-17 | 0.75x |
| 10000 | 0.230 | 0.084 | 0.692 | 1.39e-17 | 2.74x |
| 100000 | 3.797 | 0.552 | 2.971 | 1.39e-17 | 6.87x |

### Degree 50

| Batch Size | NumPy (ms) | Poema Φ-FMA (ms) | PyTorch Direct (ms) | Poema Error | Speedup vs NumPy |
|------------|------------|------------------|---------------------|-------------|------------------|
| 1000 | 0.075 | 0.234 | 0.663 | 1.39e-17 | 0.32x |
| 10000 | 0.299 | 0.348 | 4.024 | 1.39e-17 | 0.86x |
| 100000 | 3.234 | 1.478 | 17.873 | 1.39e-17 | 2.19x |

### Degree 100

| Batch Size | NumPy (ms) | Poema Φ-FMA (ms) | PyTorch Direct (ms) | Poema Error | Speedup vs NumPy |
|------------|------------|------------------|---------------------|-------------|------------------|
| 1000 | 0.150 | 0.443 | 1.274 | 1.39e-17 | 0.34x |
| 10000 | 0.548 | 0.691 | 8.190 | 1.39e-17 | 0.79x |
| 100000 | 5.449 | 2.690 | 34.515 | 1.39e-17 | 2.03x |

## Notes
- Error is measured against NumPy polyval (reference implementation)
- Speedup > 1.0 means Poema is faster than NumPy
- Φ-FMA uses Horner evaluation with certified error bounds
