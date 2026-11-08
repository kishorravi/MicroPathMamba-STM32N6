"""Shared data loading, transforms and metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import metrics as skm
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_folder(root: str | Path) -> pd.DataFrame:
    """Kather-style folder: root/<CLASS>/*.tif -> DataFrame(path, label)."""
    rows = []
    for ci, c in enumerate(CLASSES):
        d = Path(root) / c
        if not d.exists():
            raise FileNotFoundError(f"missing class folder {d}")
        rows += [(str(p), ci) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows, columns=["path", "label"])


def train_tf(img=224):
    return T.Compose([
        T.RandomResizedCrop(img, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def eval_tf(img=224):
    return T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(MEAN, STD)])


class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.labels, self.tf = df["path"].tolist(), df["label"].tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins=15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def all_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(1)
    cm = skm.confusion_matrix(labels, pred, labels=range(len(CLASSES)))
    tn = cm.sum() - (cm.sum(0) + cm.sum(1) - np.diag(cm))
    fp = cm.sum(0) - np.diag(cm)
    out = {
        "accuracy": skm.accuracy_score(labels, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(labels, pred),
        "precision_macro": skm.precision_score(labels, pred, average="macro", zero_division=0),
        "recall_macro": skm.recall_score(labels, pred, average="macro", zero_division=0),
        "f1_macro": skm.f1_score(labels, pred, average="macro", zero_division=0),
        "f1_weighted": skm.f1_score(labels, pred, average="weighted", zero_division=0),
        "specificity_macro": float(np.mean(tn / np.maximum(tn + fp, 1))),
        "mcc": skm.matthews_corrcoef(labels, pred),
        "kappa": skm.cohen_kappa_score(labels, pred),
        "auc_macro_ovr": skm.roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        "ece": ece_score(probs, labels),
        "per_class_f1": dict(zip(CLASSES, skm.f1_score(labels, pred, average=None, zero_division=0).round(4).tolist())),
        "confusion_matrix": cm.tolist(),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def get_device() -> torch.device:
    """CUDA (NVIDIA) > MPS (Apple Silicon GPU) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        P.append(torch.softmax(logits.float(), 1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
