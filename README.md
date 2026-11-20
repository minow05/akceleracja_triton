# Triton vs C++ (CUDA) GEMM Benchmark

Performance study comparing **OpenAI Triton** against **C++ (CUDA)** for half-precision matrix multiplication (GEMM: $C = A \times B$) across sizes $512 \times 512 \times 512$, $2048 \times 2048 \times 2048$, and $8192 \times 8192 \times 8192$.

---

## 🚀 Repository Structure

```tree
akceleracja_triton/
├── kernels/
│   ├── __init__.py
│   └── gemm.cu             # C++ (CUDA) kernel with 2D shared memory tiling
├── compare_gemm.py         # C++ vs Triton benchmark runner and CSV exporter
├── results.csv             # Exported latency, throughput, memory, and error metrics
├── requirements.txt        # Dependencies
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 📊 Benchmark Results (NVIDIA GeForce RTX 5060 Ti)

| Size ($M \times N \times K$) | Memory Footprint | C++ (CUDA) Time | C++ TFLOPS | Triton Time | Triton TFLOPS | Speedup (Triton vs C++) |
|---|---|---|---|---|---|---|
| **$512 \times 512 \times 512$** | 1.5 MB | 0.14 ms | 1.96 TF | **0.06 ms** | **4.22 TF** | **2.15x** |
| **$2048 \times 2048 \times 2048$** | 24.0 MB | 8.46 ms | 2.03 TF | **0.49 ms** | **35.29 TF** | **17.38x** |
| **$8192 \times 8192 \times 8192$** | 384.0 MB | 573.94 ms | 1.92 TF | **42.35 ms** | **25.96 TF** | **13.55x** |

---

## 🛠 Running the Benchmark

```powershell
.\.venv\Scripts\python.exe compare_gemm.py
```

The script compiles `kernels/gemm.cu` via `nvcc` on first execution, runs both implementations, and outputs the comparison table and `results.csv`.
