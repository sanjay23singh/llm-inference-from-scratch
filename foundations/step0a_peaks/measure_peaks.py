"""Step 0a: measure the two ceilings of THIS machine: achievable GFLOP/s and GB/s.

Roofline needs exactly two numbers. We measure them instead of trusting spec sheets.
  compute ceiling  = big square matmul (everything reused, nothing waits on memory)
  bandwidth ceiling = big copy / reduction (almost no math, everything waits on memory)

Output: results/peaks.json
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

OUT = Path(__file__).parent / "results"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def bench(fn, dev, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]  # median


def peak_gflops(dev, dtype, n):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    t = bench(lambda: a @ b, dev)
    return 2 * n**3 / t / 1e9


def peak_gbs(dev, dtype, mb=512):
    n = mb * 2**20 // torch.empty(0, dtype=dtype).element_size()
    x = torch.randn(n, device=dev, dtype=dtype)
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    t_copy = bench(lambda: y.copy_(x), dev)          # reads N bytes + writes N bytes
    t_read = bench(lambda: x.sum(), dev)             # reads N bytes
    return max(2 * nbytes / t_copy / 1e9, nbytes / t_read / 1e9)


def chip_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "unknown"


def main():
    devices = [("cpu", torch.float32, 2048)]
    if torch.backends.mps.is_available():
        devices += [("mps", torch.float32, 4096), ("mps", torch.float16, 4096)]
    res = {"chip": chip_name(), "torch": torch.__version__, "ceilings": []}
    for dev, dtype, n in devices:
        gf, bw = peak_gflops(dev, dtype, n), peak_gbs(dev, dtype)
        row = {"device": dev, "dtype": str(dtype).split(".")[-1], "gflops": gf, "gbs": bw,
               "ridge_flop_per_byte": gf / bw}
        res["ceilings"].append(row)
        print(f"{dev:4s} {row['dtype']:8s} compute {gf:9.0f} GFLOP/s | bandwidth {bw:6.1f} GB/s | "
              f"ridge point {gf / bw:6.1f} FLOP/byte")
    OUT.mkdir(exist_ok=True)
    (OUT / "peaks.json").write_text(json.dumps(res, indent=2))
    print("saved", OUT / "peaks.json")


if __name__ == "__main__":
    main()
