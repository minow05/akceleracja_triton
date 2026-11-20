# Performance Report Guide: C++ (CUDA) vs OpenAI Triton GEMM

This document provides a structured guide and technical reference for writing a report on the matrix multiplication (GEMM: $C = A \times B$) benchmark comparing C++ (CUDA) with OpenAI Triton.

---

## 1. Experimental Setup & Objectives

- **Core Problem**: Half-precision general matrix multiplication (FP16 inputs, FP32 accumulator, FP16 output), the foundational workload of Attention mechanisms and linear layers in modern neural networks.
- **Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (Blackwell Architecture, SM 12.0, 16 GB GDDR7 VRAM).
- **Evaluated Matrix Dimensions**:
  - Small: $512 \times 512 \times 512$ (1.5 MB total footprint)
  - Medium: $2048 \times 2048 \times 2048$ (24.0 MB total footprint — typical LLM layer size)
  - Large: $8192 \times 8192 \times 8192$ (384.0 MB total footprint)

---

## 2. Empirical Benchmark Data (from `results.csv`)

| Matrix Size ($M \times N \times K$) | Footprint | C++ (CUDA) Latency | C++ TFLOPS | Triton Latency | Triton TFLOPS | Speedup (Triton vs C++) |
|---|---|---|---|---|---|---|
| **$512 \times 512 \times 512$** | 1.5 MB | 0.14 ms | 1.96 TF | **0.06 ms** | **4.22 TF** | **2.15x** |
| **$2048 \times 2048 \times 2048$** | 24.0 MB | 8.46 ms | 2.03 TF | **0.49 ms** | **35.29 TF** | **17.38x** |
| **$8192 \times 8192 \times 8192$** | 384.0 MB | 573.94 ms | 1.92 TF | **42.35 ms** | **25.96 TF** | **13.55x** |

---

## 3. Analysis of the C++ (CUDA) Kernel

### Applied Optimization Techniques
- **2D Shared Memory Tiling (`__shared__`)**: Divided matrices into $16 \times 16$ tiles cached in fast on-chip SRAM to reduce redundant high-latency DRAM (VRAM) traffic.
- **Cooperative Tile Loading**: All threads in a block participate in fetching elements into shared memory before arithmetic operations.
- **Barrier Synchronization (`__syncthreads()`)**: Enforced to prevent race conditions during tile loading and tile consumption.
- **Loop Unrolling (`#pragma unroll`)**: Instructed the compiler to unroll the inner reduction loop over `TILE_DIM`.

### Why C++ (CUDA) Hit a Performance Ceiling (~2.0 TFLOPS)
1. **Scalar CUDA Core Execution**: Calculations were performed via scalar floating-point multiply-accumulate operations (`acc += tile_A * tile_B`) mapped to standard CUDA ALU cores. Specialized hardware **Tensor Cores remained completely unutilized**.
2. **Small Tile Dimensions ($16 \times 16$)**: Limited Instruction-Level Parallelism (ILP) and insufficient register reuse compared to modern hardware capabilities.
3. **Synchronization Latency**: Explicit `__syncthreads()` calls stalled warps, waiting for the slowest thread in the block at every tile iteration.

---

## 4. Why OpenAI Triton Outperformed C++ by 17x

1. **Hardware Tensor Core Utilization**: The Triton call `acc = tl.dot(a, b, acc)` was compiled directly into native hardware Tensor Core instructions (e.g. HMMA), multiplying matrix blocks in single-cycle matrix pipelines.
2. **Coarse-Grained Hierarchical Tiling ($128 \times 128 \times 32$)**: Triton divided the workload into significantly larger tile blocks, maximizing the compute-to-memory-access ratio and register locality.
3. **Compiler-Managed Software Pipelining**: Triton automatically overlaps memory transactions and computation; while current tiles are being processed in registers, subsequent tiles are asynchronously prefetched from memory.

---

## 5. Suggested Report Outline

1. **Introduction**: Importance of GEMM acceleration in deep learning and LLM inference.
2. **Environment & Methodology**: Specifications of the RTX 5060 Ti GPU, software toolchains, and evaluated matrix sizes.
3. **C++ (CUDA) Implementation**: Shared memory tiling design, thread grid configuration, and identified bottlenecks.
4. **Triton Implementation**: Block-level programming model, `tl.dot` semantics, and compiler-driven optimizations.
5. **Results & Comparative Analysis**: Evaluation of latency, TFLOPS, memory consumption, and scaling trends.
6. **Conclusion**: Productivity versus performance trade-offs; how Triton achieves expert-level GPU performance with concise, maintainable code.
