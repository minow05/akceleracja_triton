# Triton vs C++ (CUDA) GEMM Benchmark

Concise comparison of **OpenAI Triton** against **C++ (CUDA)** and **PyTorch (cuBLAS)** for half-precision matrix multiplication across sizes $512 \times 512 \times 512$, $2048 \times 2048 \times 2048$, and $8192 \times 8192 \times 8192$.

---

## 🚀 Repository Structure

```tree
akceleracja_triton/
├── kernels/
│   ├── __init__.py
│   └── gemm.cu             # C++ (CUDA) kernel with 2D shared memory tiling
├── compare_gemm.py         # Triton vs C++ benchmark runner and CSV exporter
├── results.csv             # Exported performance and memory metrics
├── requirements.txt        # Dependencies
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 📊 Benchmark Results (RTX 5060 Ti)

| Size | Memory Footprint | C++ (CUDA) Time | C++ TFLOPS | Triton Time | Triton TFLOPS | PyTorch Time | Speedup (Triton vs C++) |
|---|---|---|---|---|---|---|---|
| **512x512** | 1.5 MB | 0.17 ms | 1.57 TF | **0.06 ms** | **4.55 TF** | 0.04 ms | **2.89x** |
| **2048x2048** | 24.0 MB | 8.97 ms | 1.92 TF | **0.41 ms** | **42.15 TF** | 0.42 ms | **21.99x** |
| **8192x8192** | 384.0 MB | 603.13 ms | 1.82 TF | **37.69 ms** | **29.17 TF** | 23.67 ms | **16.00x** |

---

## 🛠 Running the Benchmark

```powershell
.\.venv\Scripts\python.exe compare_gemm.py
```
Outputs console summary and generates `results.csv`.
