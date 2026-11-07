"""Evaluate a PyTorch checkpoint or an ONNX (FP32/INT8) model on a Kather-style folder.

python evaluate.py --model micropathmamba --ckpt runs/.../best.pt --data CRC-VAL-HE-7K --out results/x
python evaluate.py --onnx deploy/student_int8.onnx --data CRC-VAL-HE-7K --out results/x_int8
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import CLASSES, PatchDataset, all_metrics, eval_tf, list_folder, predict, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def onnx_predict(path, loader):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    P, Y = [], []
    for x, y in loader:
        for xi in x:   # static batch 1, as deployed
            logits = sess.run(None, {"input": xi.unsqueeze(0).numpy()})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            P.append(e / e.sum(1, keepdims=True))
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def plot_cm(cm, path):
    cm = np.array(cm); norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASSES, rotation=45); ax.set_yticks(range(9), CLASSES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if norm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--ckpt"); ap.add_argument("--onnx")
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--img", type=int, default=224); ap.add_argument("--bs", type=int, default=128)
a = ap.parse_args()

dl = DataLoader(PatchDataset(list_folder(a.data), eval_tf(a.img)), a.bs, num_workers=8)
if a.onnx:
    probs, ys = onnx_predict(a.onnx, dl)
else:
    from models import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(a.model, 9, pretrained=False)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    probs, ys = predict(m.to(dev), dl, dev)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "preds.npz", probs=probs, labels=ys)
res = all_metrics(probs, ys)
save_json(res, out / "metrics.json")
plot_cm(res["confusion_matrix"], out / "confusion_matrix.png")
print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
