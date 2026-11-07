"""Statistics for the paper.

1) Mean +- SD over folds x seeds for every model (internal CV and external 7K)
2) Bootstrap 95 % CI on external accuracy and macro-F1
3) Exact McNemar test of every model against a reference model on the same 7K patches

python scripts/stats.py --runs runs --ref micropathmamba_kd-mambavision_t --out results/stats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--ref", required=True, help="run folder name of the reference model")
ap.add_argument("--out", default="results/stats")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

rows, ext = [], {}
for mf in sorted(Path(a.runs).glob("*/f*_s*/metrics.json")):
    j = json.loads(mf.read_text())
    model, split = mf.parts[-3], mf.parts[-2]
    r = {"model": model, "run": split}
    for part in ("internal_cv", "external_7k"):
        if part in j:
            for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc", "kappa", "auc_macro_ovr", "ece"):
                r[f"{part}.{k}"] = j[part][k]
    rows.append(r)
    pz = mf.parent / "ext_preds.npz"
    if pz.exists():
        d = np.load(pz); ext.setdefault(model, []).append((d["probs"].argmax(1), d["labels"]))

df = pd.DataFrame(rows)
summary = df.drop(columns="run").groupby("model").agg(["mean", "std"]).round(4)
summary.to_csv(out / "summary_mean_sd.csv")
print(summary.filter(like="accuracy"))

boot_rows = []
for model, runs in ext.items():
    pred, y = runs[0]          # first fold/seed model; repeat for others if needed
    n = len(y); accs, f1s = [], []
    for _ in range(a.boot):
        i = rng.integers(0, n, n)
        accs.append((pred[i] == y[i]).mean()); f1s.append(f1_score(y[i], pred[i], average="macro"))
    boot_rows.append(dict(model=model, acc=(pred == y).mean(), acc_lo=np.percentile(accs, 2.5),
                          acc_hi=np.percentile(accs, 97.5), f1_lo=np.percentile(f1s, 2.5),
                          f1_hi=np.percentile(f1s, 97.5)))
pd.DataFrame(boot_rows).round(4).to_csv(out / "bootstrap_ci_external.csv", index=False)

if a.ref in ext:
    ref_pred, y = ext[a.ref][0]
    mc = []
    for model, runs in ext.items():
        if model == a.ref:
            continue
        p, _ = runs[0]
        b = int(((ref_pred == y) & (p != y)).sum()); c = int(((ref_pred != y) & (p == y)).sum())
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        mc.append(dict(model=model, ref_right_other_wrong=b, ref_wrong_other_right=c, p_value=pval))
    pd.DataFrame(mc).to_csv(out / "mcnemar_vs_ref.csv", index=False)
    print(pd.DataFrame(mc))
