"""Tier B: try to export a Mamba teacher to ONNX for the ST Edge AI deployability analysis.

MambaVision calls mamba-ssm's CUDA `selective_scan_fn`, which ONNX cannot export.
We swap it for mamba-ssm's pure-PyTorch `selective_scan_ref` (same maths, Python loop
over tokens). The tracer then unrolls the loop, giving a very large but valid graph.
That graph, its size, and what ST Edge AI does with it ARE the Tier-B result.

python scripts/export_teacher_onnx.py --model mambavision_t --ckpt runs/mambavision_t/f0_s1/best.pt \
    --out deploy/teachers/mambavision_t_fp32.onnx
Then:  stedgeai analyze --model deploy/teachers/mambavision_t_fp32.onnx --target stm32n6 --st-neural-art
(see stm32/README.md). For VMamba use the repo's pure-PyTorch scan option (forward_type with a
"torch" implementation) before exporting.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mambavision_t")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()

if a.model.startswith("mambavision"):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    import mambavision.models.mamba_vision as mv
    mv.selective_scan_fn = selective_scan_ref          # pure-PyTorch scan, exportable

m = build_model(a.model, 9, pretrained=False)
m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
m.eval()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(m, torch.randn(1, 3, 224, 224), a.out, input_names=["input"], output_names=["logits"],
                  opset_version=a.opset, do_constant_folding=True, dynamo=False)

import onnx  # noqa: E402
g = onnx.load(a.out, load_external_data=False).graph
ops = Counter(n.op_type for n in g.node)
print(f"nodes: {len(g.node)}  | file: {Path(a.out).stat().st_size / 1e6:.1f} MB")
print("top operators:", ops.most_common(15))
