"""
One-shot data collection for the report (C++ CUDA vs Triton GEMM).

Usage (from the repo root, ideally in an *Administrator* PowerShell so Nsight Compute works):
    .\\.venv\\Scripts\\python.exe report_bench.py            # full run (~10-20 min)
    .\\.venv\\Scripts\\python.exe report_bench.py --quick    # smoke test (~1-2 min)
    .\\.venv\\Scripts\\python.exe report_bench.py --no-ncu   # skip Nsight Compute profiling

Everything is written to report_data/:
    env.json / env.txt / packages.txt   hardware + software versions, nvidia-smi, nvcc
    bench_runs.csv                      every measurement (run x size x implementation)
    bench_summary.csv                   mean/std over runs, TFLOPS, speedup vs C++ (CUDA)
    correctness.csv                     error of every implementation vs an FP32 reference
    launch_overhead.csv                 eager loop vs CUDA Graph replay for small sizes
    ablation.csv                        Triton tile / num_warps / num_stages sweep
    group_m.csv                         effect of the GROUP_M (L2-friendly) program ordering
    gpu_state.csv                       temperature / clocks / power after every run
    isa/                                PTX + SASS of all kernels, instruction counts
    ncu_2048.ncu-rep, ncu_2048_details.csv   Nsight Compute profile (if available)
    log.txt                             full console output
"""
import argparse
import csv
import glob
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import compare_gemm as cg  # noqa: E402  (builds/loads kernels/gemm_<arch>.dll)

OUT = os.path.join(ROOT, "report_data")
ISA = os.path.join(OUT, "isa")

torch.backends.cuda.matmul.allow_tf32 = False  # FP32 reference must be real FP32
torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------------------
# Tuned Triton kernel: same algorithm as the baseline + GROUP_M program re-ordering
# (improves L2 reuse for large matrices) + configurable num_warps / num_stages.
# --------------------------------------------------------------------------------------
@triton.jit
def triton_gemm_grouped_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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


def cfg_name(cfg):
    return "BM{BM}_BN{BN}_BK{BK}_G{G}_w{W}_s{S}".format(**cfg)


def launch_grouped(a, b, cfg):
    M, K = a.shape
    _, N = b.shape
    assert M % cfg["BM"] == 0 and N % cfg["BN"] == 0 and K % cfg["BK"] == 0
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    k = triton_gemm_grouped_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_K=cfg["BK"], GROUP_M=cfg["G"],
        num_warps=cfg["W"], num_stages=cfg["S"],
    )
    return c, k


def make_grouped(cfg):
    return lambda a, b: launch_grouped(a, b, cfg)[0]


def launch_baseline(a, b):
    """Same launch as compare_gemm.gemm_triton, but also returns the compiled kernel."""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    k = cg.triton_gemm_kernel[(M // 128, N // 128)](
        a, b, c, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    )
    return c, k


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.s = sys.stdout

    def write(self, x):
        self.s.write(x)
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.s.flush()
        self.f.flush()

    def __getattr__(self, name):
        return getattr(self.s, name)


def sh(args, timeout=300):
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return -1, f"<failed to run {args}: {e}>"


def write_csv(name, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> report_data/{name} ({len(rows)} rows)")


def tflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def make_inputs(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((n, n), device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16, generator=g)
    return a, b


_FLUSH = None


def l2_flush_buffer():
    global _FLUSH
    if _FLUSH is None:
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0
        nbytes = max(64 * 2**20, 4 * l2)
        _FLUSH = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")
    return _FLUSH


def bench(fn, target_ms=300.0, min_reps=5, max_reps=300):
    """Median kernel time in ms. L2 is flushed before every repetition; each repetition is
    timed separately with CUDA events (same idea as triton.testing.do_bench)."""
    flush = l2_flush_buffer()
    fn()
    torch.cuda.synchronize()
    # estimate
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(3):
        fn()
    e.record()
    torch.cuda.synchronize()
    est = max(s.elapsed_time(e) / 3, 1e-3)
    # warmup ~25 ms
    for _ in range(max(1, min(50, int(25 / est)))):
        fn()
    n = int(max(min_reps, min(max_reps, target_ms / est)))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(st.elapsed_time(en) for st, en in zip(starts, ends))

    def q(p):
        return t[min(len(t) - 1, int(p * (len(t) - 1) + 0.5))]

    return {"median_ms": q(0.5), "p20_ms": q(0.2), "p80_ms": q(0.8),
            "min_ms": t[0], "max_ms": t[-1], "reps": n}


def gpu_state():
    base = "temperature.gpu,clocks.sm,clocks.mem,power.draw"
    for fields in (base + ",clocks_event_reasons.active", base):
        code, out = sh(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
        if code == 0:
            return out.strip()
    return out.strip()


def cfg_to_arg(cfg):
    return ",".join(str(cfg[k]) for k in ("BM", "BN", "BK", "G", "W", "S"))


def arg_to_cfg(arg):
    return dict(zip(("BM", "BN", "BK", "G", "W", "S"), (int(x) for x in arg.split(","))))


def impls(tuned_cfg):
    d = {
        "cuda_tiled": cg.gemm_cpp,
        "triton_baseline": cg.gemm_triton,
    }
    if tuned_cfg is not None:
        d["triton_tuned"] = make_grouped(tuned_cfg)
    return d


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
def section_env():
    print("\n[1] Environment")
    p = torch.cuda.get_device_properties(0)
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{p.major}.{p.minor}",
        "cuda_arch_used_for_nvcc": cg.CUDA_ARCH,
        "sm_count": p.multi_processor_count,
        "total_memory_gb": round(p.total_memory / 2**30, 2),
    }
    for attr in ("L2_cache_size", "memory_clock_rate", "memory_bus_width",
                 "max_threads_per_multi_processor", "shared_memory_per_block_optin",
                 "shared_memory_per_multiprocessor", "regs_per_multiprocessor", "warp_size"):
        if hasattr(p, attr):
            info[attr] = getattr(p, attr)
    if sys.platform == "win32":
        _, cpu = sh(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        _, ram = sh(["powershell", "-NoProfile", "-Command",
                     "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"])
        info["cpu"] = cpu.strip() or info["cpu"]
        info["ram_gb"] = ram.strip()
    _, q = sh(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,"
               "power.limit,power.max_limit,pcie.link.gen.max,pcie.link.width.max",
               "--format=csv"])
    info["nvidia_smi_query"] = q.strip()
    with open(os.path.join(OUT, "env.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    parts = ["=== env.json ===", json.dumps(info, indent=2)]
    for title, args in [
        ("nvidia-smi", ["nvidia-smi"]),
        ("nvidia-smi -q (clocks/power/performance)",
         ["nvidia-smi", "-q", "-d", "CLOCK,POWER,PERFORMANCE"]),
        ("nvcc --version", ["nvcc", "--version"]),
    ]:
        parts += [f"\n=== {title} ===", sh(args)[1]]
    with open(os.path.join(OUT, "env.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    with open(os.path.join(OUT, "packages.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pkgs) + "\n")
    for k in ("gpu", "compute_capability", "sm_count", "torch", "torch_cuda", "triton", "cpu"):
        print(f"  {k}: {info.get(k)}")


def count(pattern, text):
    return len(re.findall(pattern, text))


PTX_PATTERNS = {
    "mma.sync": r"\bmma\.sync\b", "wgmma": r"\bwgmma\b", "ldmatrix": r"\bldmatrix\b",
    "cp.async": r"\bcp\.async\b", "fma.rn.f32": r"\bfma\.rn\.f32\b", "bar.sync": r"\bbar\.sync\b",
}
SASS_PATTERNS = {
    "HMMA": r"\bHMMA\b", "FFMA": r"\bFFMA\b", "LDGSTS (async copy)": r"\bLDGSTS\b",
    "LDSM (ldmatrix)": r"\bLDSM\b", "LDS": r"\bLDS\b", "LDG": r"\bLDG\b", "BAR": r"\bBAR\b",
}


def cuobjdump_exe():
    exe = shutil.which("cuobjdump")
    if exe is None and shutil.which("nvcc"):
        cand = os.path.join(os.path.dirname(shutil.which("nvcc")), "cuobjdump.exe")
        exe = cand if os.path.exists(cand) else None
    return exe


def dump_triton_isa(name, compiled, summary):
    if compiled is None:
        summary[name] = {"error": "kernel launch did not return a CompiledKernel"}
        return
    asm = compiled.asm
    entry = {}
    for key in ("ttgir", "ptx"):
        if key in asm:
            with open(os.path.join(ISA, f"{name}.{key}.txt"), "w", encoding="utf-8") as f:
                f.write(asm[key])
    ptx = asm.get("ptx", "")
    entry["ptx_static_counts"] = {k: count(p, ptx) for k, p in PTX_PATTERNS.items()}
    md = compiled.metadata
    for attr in ("num_warps", "num_stages", "shared"):
        v = md.get(attr) if isinstance(md, dict) else getattr(md, attr, None)
        entry[attr] = v
    for attr in ("n_regs", "n_spills"):
        entry[attr] = getattr(compiled, attr, None)
    exe = cuobjdump_exe()
    if "cubin" in asm and exe:
        tmp = os.path.join(ISA, f"_{name}.cubin")
        with open(tmp, "wb") as f:
            f.write(asm["cubin"])
        _, sass = sh([exe, "-sass", tmp])
        os.remove(tmp)
        with open(os.path.join(ISA, f"{name}.sass.txt"), "w", encoding="utf-8") as f:
            f.write(sass)
        entry["sass_static_counts"] = {k: count(p, sass) for k, p in SASS_PATTERNS.items()}
    summary[name] = entry


def dump_cuda_isa(summary):
    cu = os.path.join(ROOT, "kernels", "gemm.cu")
    arch = cg.CUDA_ARCH
    entry = {"nvcc_flags": f"-O3 -arch={arch}"}
    ptx_path = os.path.join(ISA, "cuda_tiled.ptx.txt")
    code, out = sh(["nvcc", "-O3", f"-arch={arch}", "-ptx", cu, "-o", ptx_path])
    if code == 0 and os.path.exists(ptx_path):
        ptx = open(ptx_path, encoding="utf-8").read()
        entry["ptx_static_counts"] = {k: count(p, ptx) for k, p in PTX_PATTERNS.items()}
    else:
        entry["ptx_error"] = out
    cubin = os.path.join(ISA, "_cuda_tiled.cubin")
    code, out = sh(["nvcc", "-O3", f"-arch={arch}", "-cubin", "-Xptxas", "-v", cu, "-o", cubin])
    entry["ptxas_resource_usage"] = out.strip()
    exe = cuobjdump_exe()
    if code == 0 and exe:
        _, sass = sh([exe, "-sass", cubin])
        with open(os.path.join(ISA, "cuda_tiled.sass.txt"), "w", encoding="utf-8") as f:
            f.write(sass)
        entry["sass_static_counts"] = {k: count(p, sass) for k, p in SASS_PATTERNS.items()}
    if os.path.exists(cubin):
        os.remove(cubin)
    summary["cuda_tiled"] = entry


def section_isa(tuned_cfg):
    print("\n[ISA] PTX / SASS dumps (proof of Tensor Core usage)")
    os.makedirs(ISA, exist_ok=True)
    summary = {}
    a, b = make_inputs(2048)
    _, k = launch_baseline(a, b)
    dump_triton_isa("triton_baseline", k, summary)
    if tuned_cfg is not None:
        _, k = launch_grouped(a, b, tuned_cfg)
        dump_triton_isa("triton_tuned", k, summary)
        summary["triton_tuned"]["config"] = tuned_cfg
    dump_cuda_isa(summary)
    with open(os.path.join(ISA, "isa_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name, e in summary.items():
        print(f"  {name}: SASS {e.get('sass_static_counts')} | regs={e.get('n_regs')} "
              f"shared={e.get('shared')}")
    print("  -> report_data/isa/")


def section_ablation(sizes, quick):
    print("\n[2] Triton ablation (tile sizes, num_warps, num_stages; GROUP_M=8)")
    tiles = [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]
    bks, warps, stages = [32, 64], [4, 8], [2, 3, 4]
    if quick:
        tiles, bks, warps, stages = [(128, 128), (128, 256)], [32, 64], [4, 8], [3]
    cfgs = [dict(BM=bm, BN=bn, BK=bk, G=8, W=w, S=s)
            for (bm, bn) in tiles for bk in bks for w in warps for s in stages]
    rows, scores = [], {}
    inputs = {n: make_inputs(n) for n in sizes}
    for i, cfg in enumerate(cfgs):
        name = cfg_name(cfg)
        per_size = {}
        for n in sizes:
            a, b = inputs[n]
            row = {"config": name, **cfg, "size": n}
            try:
                fn = make_grouped(cfg)
                r = bench(lambda: fn(a, b), target_ms=100, min_reps=5)
                row.update(status="ok", median_ms=round(r["median_ms"], 4),
                           tflops=round(tflops(n, n, n, r["median_ms"]), 2))
                per_size[n] = row["tflops"]
            except Exception as e:  # noqa: BLE001  (e.g. OutOfResources: shared memory)
                row.update(status=f"failed: {type(e).__name__}: {(str(e).splitlines() or [""])[0][:120]}")
            rows.append(row)
        ok = len(per_size) == len(sizes)
        if ok:
            scores[name] = (math.exp(sum(math.log(v) for v in per_size.values()) / len(sizes)), cfg)
        desc = " | ".join(f"{n}: {per_size.get(n, 'FAIL')}" for n in sizes)
        print(f"  [{i + 1:3d}/{len(cfgs)}] {name:<28} TFLOPS {desc}")
    write_csv("ablation.csv", rows)
    del inputs
    torch.cuda.empty_cache()
    if not scores:
        print("  !! no ablation config succeeded")
        return None
    best_name, (score, best) = max(scores.items(), key=lambda kv: kv[1][0])
    print(f"  best config (geo-mean TFLOPS over {sizes}): {best_name} = {score:.2f} TF")
    return dict(best)


def section_group_m(tuned_cfg, sizes):
    print("\n[3] GROUP_M study (L2-friendly program ordering) for the best tile config")
    rows = []
    for n in sizes:
        a, b = make_inputs(n)
        for g in [1, 2, 4, 8, 16]:
            cfg = dict(tuned_cfg, G=g)
            fn = make_grouped(cfg)
            r = bench(lambda: fn(a, b), target_ms=150)
            t = tflops(n, n, n, r["median_ms"])
            rows.append({"size": n, "GROUP_M": g, "config": cfg_name(cfg),
                         "median_ms": round(r["median_ms"], 4), "tflops": round(t, 2)})
            print(f"  n={n:<5} GROUP_M={g:<3} {r['median_ms']:9.3f} ms  {t:6.2f} TF")
        del a, b
        torch.cuda.empty_cache()
    write_csv("group_m.csv", rows)


def section_main(sizes, runs, tuned_cfg):
    print(f"\n[4] Main benchmark: {runs} runs x {len(sizes)} sizes")
    rows, states = [], []
    fns = impls(tuned_cfg)
    for run in range(1, runs + 1):
        for n in sizes:
            a, b = make_inputs(n, seed=run)
            for name, fn in fns.items():
                r = bench(lambda: fn(a, b))
                t = tflops(n, n, n, r["median_ms"])
                rows.append({"run": run, "size": n, "impl": name,
                             **{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()},
                             "tflops": round(t, 3)})
                print(f"  run {run} n={n:<5} {name:<16} {r['median_ms']:10.4f} ms  "
                      f"{t:7.2f} TF  (reps={r['reps']})")
            del a, b
            torch.cuda.empty_cache()
        st = gpu_state()
        states.append({"run": run, "time": time.strftime("%H:%M:%S"),
                       "temp_C,sm_clock,mem_clock,power[,clock_event_reasons]": st})
        print(f"  GPU state after run {run}: {st}")
    write_csv("bench_runs.csv", rows)
    write_csv("gpu_state.csv", states)

    summary = []
    for n in sizes:
        by_impl = {}
        for name in fns:
            ms = [r["median_ms"] for r in rows if r["size"] == n and r["impl"] == name]
            by_impl[name] = (statistics.mean(ms), statistics.stdev(ms) if len(ms) > 1 else 0.0)
        for name, (mean, std) in by_impl.items():
            summary.append({
                "size": n, "impl": name, "runs": runs,
                "mean_ms": round(mean, 4), "std_ms": round(std, 4),
                "cv_percent": round(100 * std / mean, 2),
                "tflops": round(tflops(n, n, n, mean), 2),
                "speedup_vs_cuda_tiled": round(by_impl["cuda_tiled"][0] / mean, 2),
                "footprint_mb": round(3 * n * n * 2 / 2**20, 2),
            })
    write_csv("bench_summary.csv", summary)
    print("\n  Summary (mean of per-run medians):")
    print(f"  {'size':<6} " + " ".join(f"{k:>18}" for k in fns))
    for n in sizes:
        cells = [next(s for s in summary if s["size"] == n and s["impl"] == k) for k in fns]
        print(f"  {n:<6} " + " ".join(f"{c['mean_ms']:9.3f}ms {c['tflops']:5.1f}TF" for c in cells))


def section_correctness(sizes, tuned_cfg):
    print("\n[5] Correctness vs FP32 reference (a.float() @ b.float(), TF32 disabled)")
    rows = []
    for n in sizes:
        a, b = make_inputs(n)
        ref = a.float() @ b.float()
        ref_norm = torch.linalg.norm(ref).item()
        floor = (ref.half().float() - ref)  # error of just rounding the exact result to FP16
        rows.append({"size": n, "impl": "fp16_rounding_of_reference (lower bound)",
                     "max_abs_err": round(floor.abs().max().item(), 5),
                     "mean_abs_err": round(floor.abs().mean().item(), 6),
                     "rel_frobenius_err": f"{torch.linalg.norm(floor).item() / ref_norm:.3e}",
                     "max_abs_ref": round(ref.abs().max().item(), 2)})
        for name, fn in impls(tuned_cfg).items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = fn(a, b)
            torch.cuda.synchronize()
            peak_extra = torch.cuda.max_memory_allocated() - before - out.numel() * out.element_size()
            d = out.float() - ref
            rows.append({"size": n, "impl": name,
                         "max_abs_err": round(d.abs().max().item(), 5),
                         "mean_abs_err": round(d.abs().mean().item(), 6),
                         "rel_frobenius_err": f"{torch.linalg.norm(d).item() / ref_norm:.3e}",
                         "max_abs_ref": round(ref.abs().max().item(), 2),
                         "output_mb": round(out.numel() * 2 / 2**20, 2),
                         "extra_workspace_mb": round(max(0, peak_extra) / 2**20, 2)})
            del out, d
        print(f"  n={n}: " + ", ".join(f"{r['impl'].split(' ')[0]}={r['rel_frobenius_err']}"
                                      for r in rows if r["size"] == n))
        del a, b, ref, floor
        torch.cuda.empty_cache()
    write_csv("correctness.csv", rows)


def section_launch_overhead(sizes, tuned_cfg):
    print("\n[6] Launch overhead: eager back-to-back loop vs CUDA Graph replay (no L2 flush)")
    rows, R = [], 20
    for n in sizes:
        a, b = make_inputs(n)
        for name, fn in impls(tuned_cfg).items():
            row = {"size": n, "impl": name, "calls_per_measurement": R}
            for _ in range(3):
                fn(a, b)
            torch.cuda.synchronize()
            eager = []
            for _ in range(10):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(R):
                    fn(a, b)
                e.record()
                torch.cuda.synchronize()
                eager.append(s.elapsed_time(e) / R)
            row["eager_ms"] = round(statistics.median(eager), 5)
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(R):
                        fn(a, b)
                g.replay()
                torch.cuda.synchronize()
                gt = []
                for _ in range(10):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    g.replay()
                    e.record()
                    torch.cuda.synchronize()
                    gt.append(s.elapsed_time(e) / R)
                row["graph_ms"] = round(statistics.median(gt), 5)
                row["launch_overhead_us"] = round(1000 * (row["eager_ms"] - row["graph_ms"]), 2)
                del g
            except Exception as e:  # noqa: BLE001
                row["graph_ms"] = f"failed: {type(e).__name__}"
            rows.append(row)
            print(f"  n={n:<5} {name:<16} eager {row['eager_ms']:.4f} ms  graph {row['graph_ms']}")
        del a, b
        torch.cuda.empty_cache()
    write_csv("launch_overhead.csv", rows)


def find_ncu():
    exe = shutil.which("ncu")
    if exe:
        return exe
    cands = sorted(glob.glob(r"C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.exe"))
    cands += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\nsight-compute*\ncu.exe"))
    return cands[-1] if cands else None


def section_ncu(tuned_cfg):
    print("\n[7] Nsight Compute profile (n=2048, one launch per implementation)")
    status_path = os.path.join(OUT, "ncu_status.txt")
    ncu = find_ncu()
    if ncu is None:
        open(status_path, "w").write("ncu not found (install Nsight Compute / add to PATH)\n")
        print("  !! ncu not found - skipped")
        return
    rep = os.path.join(OUT, "ncu_2048")
    args = [ncu, "--set", "full", "--nvtx", "--profile-from-start", "off", "-f", "-o", rep,
            sys.executable, os.path.abspath(__file__), "--ncu-target"]
    if tuned_cfg:
        args += ["--tuned", cfg_to_arg(tuned_cfg)]
    print("  running:", " ".join(args[:8]), "...")
    code, out = sh(args, timeout=1800)
    with open(status_path, "w", encoding="utf-8") as f:
        f.write(f"exit code: {code}\n\n{out}")
    if code != 0 or not os.path.exists(rep + ".ncu-rep"):
        hint = ""
        if "ERR_NVGPUCTRPERM" in out:
            hint = ("\n  -> no permission for GPU performance counters: run PowerShell as "
                    "Administrator, or NVIDIA Control Panel > Developer > Manage GPU Performance "
                    "Counters > 'Allow access to all users'.")
        print(f"  !! ncu failed (exit {code}), see report_data/ncu_status.txt{hint}")
        return
    code, details = sh([ncu, "--import", rep + ".ncu-rep", "--csv", "--page", "details"])
    with open(os.path.join(OUT, "ncu_2048_details.csv"), "w", encoding="utf-8") as f:
        f.write(details)
    print("  -> report_data/ncu_2048.ncu-rep, ncu_2048_details.csv")


def ncu_target(tuned_cfg):
    """Child process run under ncu: one profiled launch of each implementation."""
    a, b = make_inputs(2048)
    fns = impls(tuned_cfg)
    for fn in fns.values():  # compile / warm up outside the profiled region
        fn(a, b)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, fn in fns.items():
        torch.cuda.nvtx.range_push(name)
        fn(a, b)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short smoke test")
    ap.add_argument("--no-ncu", action="store_true", help="skip Nsight Compute")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--ncu-target", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tuned", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.ncu_target:
        ncu_target(arg_to_cfg(args.tuned) if args.tuned else None)
        return 0

    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT, "log.txt"))
    t0 = time.time()

    if args.quick:
        sizes, abl_sizes, runs = [512, 2048], [2048], args.runs or 1
        grp_sizes, small = [2048], [512]
    else:
        sizes = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
        abl_sizes, runs = [2048, 8192], args.runs or 3
        grp_sizes, small = [2048, 4096, 8192], [256, 512, 1024, 2048]
    print(f"report_bench.py  quick={args.quick}  sizes={sizes}  runs={runs}")

    failed = []

    def run(name, f, *a):
        try:
            return f(*a)
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"\n!! section '{name}' failed:\n{traceback.format_exc()}")
            return None

    run("env", section_env)
    tuned = run("ablation", section_ablation, abl_sizes, args.quick)
    if tuned is not None:
        with open(os.path.join(OUT, "tuned_config.json"), "w") as f:
            json.dump(tuned, f, indent=2)
        run("group_m", section_group_m, tuned, grp_sizes)
    run("isa", section_isa, tuned)
    run("main", section_main, sizes, runs, tuned)
    run("correctness", section_correctness, sizes, tuned)
    run("launch_overhead", section_launch_overhead, small, tuned)
    if not args.no_ncu:
        run("ncu", section_ncu, tuned)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Failed sections: {failed or 'none'}")
    print("Results are in report_data/ - commit and push that folder.")
    return 1 if "main" in failed else 0


if __name__ == "__main__":
    sys.exit(main())
