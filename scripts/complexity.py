"""Parameters, FLOPs, GPU/CPU latency and throughput for every model (Tier A / B tables).

python scripts/complexity.py --models micropathmamba mambavision_t vmamba_t convnextv2_t swin_t \
    efficientvit_b1 mobilenetv4_m mobilenetv3_s resnet50 --out results/complexity.csv

FLOPs use fvcore, which cannot see inside custom CUDA selective-scan kernels. For the Mamba
teachers, report the FLOPs given by their authors (and say so in the table footnote).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model  # noqa: E402


@torch.no_grad()
def latency(m, x, n=200, warm=50):
    for _ in range(warm):
        m(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        m(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--out", default="results/complexity.csv")
a = ap.parse_args()

rows = []
for name in a.models:
    m = build_model(name, 9, pretrained=False).eval()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    try:
        from fvcore.nn import FlopCountAnalysis
        fl = FlopCountAnalysis(m, torch.randn(1, 3, a.img, a.img)).unsupported_ops_warnings(False)
        gflops = fl.total() / 1e9          # fvcore counts one fused multiply-add as one "flop" (= MACs)
    except Exception as e:  # noqa: BLE001
        gflops = float("nan"); print(name, "flops failed:", e)
    r = dict(model=name, params_M=round(params, 3), gmacs_fvcore=round(gflops, 3),
             int8_weights_MB=round(params, 2), fp32_weights_MB=round(params * 4, 2))
    if torch.cuda.is_available():
        g = m.cuda()
        r["gpu_ms_bs1"] = round(latency(g, torch.randn(1, 3, a.img, a.img, device="cuda")), 3)
        ms = latency(g, torch.randn(64, 3, a.img, a.img, device="cuda"), n=30, warm=10)
        r["gpu_img_per_s_bs64"] = round(64 / ms * 1e3, 1)
        m = g.cpu()
    torch.set_num_threads(1)
    r["cpu1t_ms_bs1"] = round(latency(m, torch.randn(1, 3, a.img, a.img), n=20, warm=5), 1)
    rows.append(r); print(r, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
