"""Static INT8 post-training quantization (QDQ, per-channel) for ST Edge AI / STM32N6.

python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8.onnx \
    --folds folds.csv --fold 0 --n-calib 512 [--keep-decay-float]

--keep-decay-float leaves the SSM decay sub-graph (Sub/Abs/Exp on the 49x49 matrix) in float.
Run both variants and report the accuracy/latency trade-off in the paper (ablation).
The quantized model's accuracy is then measured with:
python evaluate.py --onnx deploy/student_int8.onnx --data CRC-VAL-HE-7K --out results/student_int8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PatchDataset, eval_tf  # noqa: E402


class Reader(CalibrationDataReader):
    def __init__(self, df, img, n):
        ds = PatchDataset(df.sample(n=min(n, len(df)), random_state=0), eval_tf(img))
        self.items = iter([{"input": ds[i][0].unsqueeze(0).numpy()} for i in range(len(ds))])

    def get_next(self):
        return next(self.items, None)


ap = argparse.ArgumentParser()
ap.add_argument("--fp32", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--folds", default="folds.csv")
ap.add_argument("--fold", type=int, default=0, help="calibrate on the TRAINING part of this fold")
ap.add_argument("--n-calib", type=int, default=512)
ap.add_argument("--img", type=int, default=224)
ap.add_argument("--keep-decay-float", action="store_true")
a = ap.parse_args()

pre = a.fp32.replace(".onnx", "_pre.onnx")
quant_pre_process(a.fp32, pre)
exclude = []
if a.keep_decay_float:
    g = onnx.load(pre).graph
    exclude = [n.name for n in g.node if "mixer" in n.name and n.op_type in {"Sub", "Abs", "Exp", "Neg"}]
    print(f"keeping {len(exclude)} decay nodes in float")

folds = pd.read_csv(a.folds)
quantize_static(pre, a.out, Reader(folds[folds.fold != a.fold], a.img, a.n_calib),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude)
print("wrote", a.out, f"({Path(a.out).stat().st_size / 1e6:.2f} MB)")
