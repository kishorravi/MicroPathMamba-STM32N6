"""
Train any model on one fold of NCT-CRC-HE-100K, optionally with knowledge distillation,
then test the best checkpoint on the external CRC-VAL-HE-7K set.

Teacher / baseline:
  python train.py --model mambavision_t --fold 0 --seed 1 --data NCT-CRC-HE-100K --ext CRC-VAL-HE-7K
Student with distillation from a trained teacher:
  python train.py --model micropathmamba --fold 0 --seed 1 --teacher mambavision_t \
      --teacher-ckpt runs/mambavision_t/f0_s1/best.pt --data ... --ext ...
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import PatchDataset, all_metrics, eval_tf, list_folder, predict, save_json, train_tf
from models import build_model


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.7, ls=0.1):
    ce = F.cross_entropy(s_logits, y, label_smoothing=ls)
    kl = F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean") * T * T
    return (1 - alpha) * ce + alpha * kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="NCT-CRC-HE-100K root")
    ap.add_argument("--ext", default=None, help="CRC-VAL-HE-7K root (external test)")
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--kd-T", type=float, default=4.0)
    ap.add_argument("--kd-alpha", type=float, default=0.7)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.model}{'_kd-' + a.teacher if a.teacher else ''}"
    out = Path(a.out) / tag / f"f{a.fold}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(a.folds)
    tr, va = folds[folds.fold != a.fold], folds[folds.fold == a.fold]
    dl_tr = DataLoader(PatchDataset(tr, train_tf(a.img)), a.bs, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(PatchDataset(va, eval_tf(a.img)), a.bs * 2, num_workers=a.workers, pin_memory=True)

    model = build_model(a.model, 9, pretrained=not a.no_pretrained).to(dev)
    teacher = None
    if a.teacher:
        teacher = build_model(a.teacher, 9, pretrained=False)
        teacher.load_state_dict(torch.load(a.teacher_ckpt, map_location="cpu"))
        teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")

    best, wait, hist = 1e9, 0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = kd_loss(logits.float(), t_logits.float(), y, a.kd_T, a.kd_alpha)
                else:
                    loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * x.size(0)
        probs, ys = predict(model, dl_va, dev)
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(ys)))
        vacc = float((probs.argmax(1) == ys).mean())
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(dl_tr.dataset), val_loss=vloss, val_acc=vacc,
                         sec=time.time() - t0))
        print(hist[-1], flush=True)
        if vloss < best:
            best, wait = vloss, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            wait += 1
            if wait >= a.patience:
                break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    probs, ys = predict(model, dl_va, dev)
    np.savez_compressed(out / "val_preds.npz", probs=probs, labels=ys)
    res = {"internal_cv": all_metrics(probs, ys)}
    if a.ext:
        ext = list_folder(a.ext)
        dl_ext = DataLoader(PatchDataset(ext, eval_tf(a.img)), a.bs * 2, num_workers=a.workers)
        p2, y2 = predict(model, dl_ext, dev)
        np.savez_compressed(out / "ext_preds.npz", probs=p2, labels=y2)
        res["external_7k"] = all_metrics(p2, y2)
    save_json({"args": vars(a), **res}, out / "metrics.json")
    print({k: round(v["accuracy"], 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
