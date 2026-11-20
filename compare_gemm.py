import sys
import os
import csv
import ctypes
import subprocess
import torch
import triton
import triton.language as tl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Build / load C++ CUDA kernel (compiled natively for the installed GPU, e.g. sm_120)
_cc_major, _cc_minor = torch.cuda.get_device_capability(0)
CUDA_ARCH = f"sm_{_cc_major}{_cc_minor}"
cu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gemm.cu")
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", f"gemm_{CUDA_ARCH}.dll")

if not os.path.exists(dll_path) or os.path.getmtime(dll_path) < os.path.getmtime(cu_path):
    subprocess.run(
        ["nvcc", "-O3", f"-arch={CUDA_ARCH}", "-shared", cu_path, "-o", dll_path],
        check=True,
    )

cuda_dll = ctypes.CDLL(dll_path)
launch_gemm = cuda_dll.launch_gemm
launch_gemm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,
]


def gemm_cpp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stream = torch.cuda.current_stream().cuda_stream
    launch_gemm(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K, stream)
    return c


# Streamlined Triton GEMM kernel
@triton.jit
def triton_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16))


def gemm_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    # The kernel has no boundary masks: dimensions must be multiples of the block sizes.
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, \
        "triton_gemm_kernel requires M, N divisible by 128 and K divisible by 32"
    grid =(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    triton_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def measure_kernel(func, a, b, iters=5, warmup=2):
    for _ in range(warmup):
        _ = func(a, b)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = func(a, b)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ms, peak_mem_mb, out


def run_benchmark():
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    sizes = [
        (512, 512, 512),
        (2048, 2048, 2048),
        (8192, 8192, 8192),
    ]

    records = []

    for M, N, K in sizes:
        print(f"\nEvaluating M={M}, N={N}, K={K} (FP16)...")
        flops = 2.0 * M * N * K
        tensor_bytes = (M * K + K * N + M * N) * 2
        footprint_mb = tensor_bytes / (1024 ** 2)

        torch.manual_seed(0)
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        iters = 3 if M == 8192 else 10
        warmup = 1 if M == 8192 else 3

        # C++ CUDA
        cpp_ms, cpp_mem, cpp_out = measure_kernel(gemm_cpp, a, b, iters=iters, warmup=warmup)
        cpp_tflops = (flops / (cpp_ms * 1e-3)) * 1e-12

        # Triton
        triton_ms, triton_mem, triton_out = measure_kernel(gemm_triton, a, b, iters=iters, warmup=warmup)
        triton_tflops = (flops / (triton_ms * 1e-3)) * 1e-12

        # Numerical comparison between C++ and Triton
        diff = torch.max(torch.abs(cpp_out - triton_out)).item()
        speedup_vs_cpp = cpp_ms / triton_ms

        records.append({
            "size": f"{M}x{N}x{K}",
            "matrix_footprint_mb": round(footprint_mb, 2),
            "cpp_time_ms": round(cpp_ms, 3),
            "cpp_tflops": round(cpp_tflops, 2),
            "cpp_peak_mem_mb": round(cpp_mem, 2),
            "triton_time_ms": round(triton_ms, 3),
            "triton_tflops": round(triton_tflops, 2),
            "triton_peak_mem_mb": round(triton_mem, 2),
            "max_diff_cpp_vs_triton": round(diff, 4),
            "speedup_triton_vs_cpp": round(speedup_vs_cpp, 2),
        })

        del a, b, cpp_out, triton_out
        torch.cuda.empty_cache()

    # Save to CSV
    csv_filename = "results.csv"
    csv_fields = [
        "size", "matrix_footprint_mb",
        "cpp_time_ms", "cpp_tflops", "cpp_peak_mem_mb",
        "triton_time_ms", "triton_tflops", "triton_peak_mem_mb",
        "max_diff_cpp_vs_triton", "speedup_triton_vs_cpp",
    ]

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved benchmark metrics to {csv_filename}")
    print("\n" + "=" * 70)
    print(f"{'Size':<16} | {'C++ (CUDA)':<18} | {'Triton':<18} | {'Speedup'}")
    print("-" * 70)
    for r in records:
        cpp_str = f"{r['cpp_time_ms']:.2f} ms ({r['cpp_tflops']:.1f} TF)"
        tri_str = f"{r['triton_time_ms']:.2f} ms ({r['triton_tflops']:.1f} TF)"
        print(f"{r['size']:<16} | {cpp_str:<18} | {tri_str:<18} | {r['speedup_triton_vs_cpp']:.1f}x")
    print("=" * 70)

    print("\n--- CSV OUTPUT CONTENT ---")
    with open(csv_filename, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    run_benchmark()
