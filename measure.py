import gc
import os
import threading
import time

import numpy as np
import pandas as pd
import pynvml
import torch

from equations import flops, memory
from models import SmallCNN


BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 42


def make_grid():
    rng = np.random.default_rng(SEED)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b not in BASE_B and b & (b - 1)]
    extra_s = sorted(rng.choice(s_pool, 4, replace=False).tolist())
    extra_b = sorted(rng.choice(b_pool, 3, replace=False).tolist())
    all_s = sorted(BASE_S + extra_s)
    all_b = sorted(BASE_B + extra_b)
    grid = [
        (S, B, S in extra_s or B in extra_b)
        for S in all_s
        for B in all_b
    ]
    print("extra S:", extra_s)
    print("extra B:", extra_b)
    return grid


def warmup(model, x, n=5):
    with torch.inference_mode():
        for _ in range(n):
            model(x)
    torch.cuda.synchronize()


def measure_latency(model, x, n=20):
    times = []
    with torch.inference_mode():
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def measure_memory(model, x):
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize()
    value = torch.cuda.max_memory_allocated()
    del y
    return int(value)


class PowerSampler:
    def __init__(self, interval=0.01):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.samples.append(
                (time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000)
            )
            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()


def measure_energy(model, x, latency_s):
    repeats = int(np.clip(np.ceil(0.5 / max(latency_s, 1e-4)), 5, 500))
    sampler = PowerSampler()
    torch.cuda.synchronize()
    sampler.start()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sampler.stop()
    powers = [p for t, p in sampler.samples if t0 <= t <= t1]
    if not powers:
        return np.nan
    return float(np.mean(powers) * (t1 - t0) / repeats)


def main():
    memory_fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "1.0"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda")
    model = SmallCNN().to(device).eval()
    props = torch.cuda.get_device_properties(0)

    print("GPU:", props.name)
    print("total memory:", props.total_memory)
    print("memory fraction:", memory_fraction)

    rows = []

    for i, (S, B, is_validation) in enumerate(make_grid(), 1):
        print(f"{i}/132 S={S} B={B}")
        x = None

        try:
            gc.collect()
            torch.cuda.empty_cache()

            x = torch.randn(B, 3, S, S, device=device)
            warmup(model, x)

            latency_s = measure_latency(model, x)
            memory_bytes = measure_memory(model, x)
            energy_j = measure_energy(model, x, latency_s)

            row = {
                "S": S,
                "B": B,
                "latency_s": latency_s,
                "memory_bytes": memory_bytes,
                "energy_j": energy_j,
                "oom": False,
                "is_validation": is_validation,
                "flops_pred": float(flops(S, B)),
                "memory_pred_bytes": float(memory(S, B)),
                "memory_limit_bytes": props.total_memory * memory_fraction,
            }

        except torch.cuda.OutOfMemoryError:
            print("OOM")
            row = {
                "S": S,
                "B": B,
                "latency_s": np.nan,
                "memory_bytes": np.nan,
                "energy_j": np.nan,
                "oom": True,
                "is_validation": is_validation,
                "flops_pred": float(flops(S, B)),
                "memory_pred_bytes": float(memory(S, B)),
                "memory_limit_bytes": props.total_memory * memory_fraction,
            }

        finally:
            if x is not None:
                del x
            gc.collect()
            torch.cuda.empty_cache()

        rows.append(row)
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv("results/measurements.csv", index=False)

    print("done")


if __name__ == "__main__":
    main()
