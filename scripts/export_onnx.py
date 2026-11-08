"""Export a trained student to a static-shape ONNX model and check it against PyTorch.

python scripts/export_onnx.py --model micropathmamba --ckpt runs/.../best.pt --out deploy/student_fp32.onnx
"""
import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="micropathmamba")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="deploy/student_fp32.onnx")
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")

m = build_model(a.model, 9, pretrained=False)
m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
m.eval()
x = torch.randn(1, 3, a.img, a.img)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(m, x, a.out, input_names=["input"], output_names=["logits"], opset_version=a.opset,
                  do_constant_folding=True, dynamo=False)

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

model = onnx.load(a.out)
onnx.checker.check_model(model)
ops = sorted({n.op_type for n in model.graph.node})
print("ONNX operators used:", ops)
bad = {"Loop", "Scan", "CumSum", "If", "NonZero"} & set(ops)
print("dynamic/control-flow ops:", bad or "none (good for ST Edge AI)")

sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = m(x).numpy()
got = sess.run(None, {"input": x.numpy()})[0]
print(f"max |torch - onnx| = {np.abs(ref - got).max():.2e}")
