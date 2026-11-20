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
├── report_bench.py         # One-shot data collection for the report -> report_data/
├── run_report.ps1          # pull -> report_bench.py -> commit + push report_data/
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

The script compiles `kernels/gemm.cu` via `nvcc` (natively for the installed GPU, e.g. `-arch=sm_120`) on first execution, runs both implementations, and outputs the comparison table and `results.csv`.

---

## 📑 Collecting data for the report (one command)

Open PowerShell **as Administrator** (Nsight Compute needs access to GPU performance counters) in the repo root and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_report.ps1
```

It does `git pull`, runs `report_bench.py` (~10–20 min), then commits and pushes `report_data/`.
Optional smoke test first: `.\.venv\Scripts\python.exe report_bench.py --quick --no-ncu`.

`report_bench.py` collects:

| Output | Content |
|---|---|
| `env.json`, `env.txt`, `packages.txt` | GPU/CPU/OS, driver, CUDA, PyTorch, Triton versions, `nvidia-smi`, `nvcc --version` |
| `bench_runs.csv`, `bench_summary.csv` | cuBLAS vs C++ (CUDA) vs Triton baseline vs Triton tuned, sizes 256–8192, 3 runs, median with L2 flush, mean ± std, TFLOPS, % of cuBLAS |
| `correctness.csv` | Error of every implementation vs an FP32 reference (plus the FP16 rounding lower bound) |
| `ablation.csv`, `tuned_config.json` | Triton sweep over tile sizes, `num_warps`, `num_stages` |
| `group_m.csv` | Effect of `GROUP_M` program re-ordering (L2 reuse) on large matrices |
| `launch_overhead.csv` | Eager loop vs CUDA Graph replay for small sizes |
| `gpu_state.csv` | Temperature, clocks and power after each run |
| `isa/` | PTX / SASS of every kernel + instruction counts (`HMMA` = Tensor Cores, `FFMA` = CUDA cores), registers, shared memory |
| `ncu_2048.ncu-rep`, `ncu_2048_details.csv` | Nsight Compute profile of each kernel at 2048³ |
| `log.txt` | Full console output |
