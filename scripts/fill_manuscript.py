"""Turn experiment outputs into the manuscript's tables and abstract numbers.

Reads
  runs/<model>/f<fold>_s<seed>/metrics.json     (written by train.py)
  results/complexity.csv                        (scripts/complexity.py)
  results/stats/mcnemar_vs_ref.csv              (scripts/stats.py, optional)
  results/stm32_measurements.json               (you fill this from ST Edge AI + power measurements;
                                                 template: results_template/stm32_measurements.json)
Writes
  paper/tables/tierA.tex, tierB.tex, tierC.tex  (\\input by manuscript.tex)
  paper/results_macros.tex                      (\\resStudentExtAcc etc. used in the abstract)
  paper/results_summary.md                      (plain-text summary to send back for the write-up)

Anything not yet measured stays a red XX, so you can run this after every new experiment.
python scripts/fill_manuscript.py --runs runs --results results --paper paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {  # run-folder name -> label in the paper, in table order
    "vmamba_t": "VMamba-T", "mambavision_t": "MambaVision-T", "mambavision_t2": "MambaVision-T2",
    "mambavision_s": "MambaVision-S", "convnextv2_t": "ConvNeXt V2-T", "swin_t": "Swin-T",
    "efficientvit_b1": "EfficientViT-B1", "resnet50": "ResNet-50", "mobilenetv4_m": "MobileNetV4-Conv-M",
    "mobilenetv3_s": "MobileNetV3-Small", "micropathmamba_tiny": "MicroPathMamba-tiny (from scratch)",
    "micropathmamba_cnn": "MicroPathMamba-CNN (no SSM)", "micropathmamba": "MicroPathMamba (no KD)",
}
MAMBA_TEACHERS = ["vmamba_t", "mambavision_t", "mambavision_t2", "mambavision_s"]
# fvcore cannot count selective-scan kernels: use the authors' published GMACs for these (dagger in table)
PUBLISHED_GMACS = {"vmamba_t": 4.9, "mambavision_t": 4.4, "mambavision_t2": 5.1, "mambavision_s": 7.5}
XX = r"\XX"


def fmt(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return XX
    return f"{v * 100:.{d}f}" if pct else f"{v:.{d}f}"


def pm(mean, sd, d=2, pct=True):
    if mean is None or np.isnan(mean):
        return XX
    k = 100 if pct else 1
    return f"{mean * k:.{d}f}" if (sd is None or np.isnan(sd)) else f"{mean * k:.{d}f}$\\pm${sd * k:.{d}f}"


def load_runs(runs: Path) -> pd.DataFrame:
    rows = []
    for mf in sorted(runs.glob("*/f*_s*/metrics.json")):
        j = json.loads(mf.read_text())
        r = {"model": mf.parts[-3], "run": mf.parts[-2]}
        for part, tag in (("internal_cv", "cv"), ("external_7k", "ext")):
            for k in ("accuracy", "f1_macro", "mcc", "ece", "auc_macro_ovr", "balanced_accuracy"):
                r[f"{tag}_{k}"] = j.get(part, {}).get(k, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def agg(df, model, col):
    d = df[df.model == model][col].dropna()
    if d.empty:
        return np.nan, np.nan, 0
    return d.mean(), (d.std(ddof=1) if len(d) > 1 else np.nan), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    runs, res, paper = Path(a.runs), Path(a.results), Path(a.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    df = load_runs(runs) if runs.exists() else pd.DataFrame(columns=["model"])
    kd = sorted({m for m in df.model.unique() if m.startswith("micropathmamba_kd-")}) if len(df) else []
    student = kd[0] if kd else "micropathmamba_kd-mambavision_t"
    order = list(NAMES) + [student]
    names = {**NAMES, student: r"\textbf{MicroPathMamba (KD)}"}

    cx = pd.read_csv(res / "complexity.csv").set_index("model") if (res / "complexity.csv").exists() else pd.DataFrame()
    mc_path = res / "stats" / "mcnemar_vs_ref.csv"
    mc = pd.read_csv(mc_path).set_index("model") if mc_path.exists() else pd.DataFrame()
    st_path = res / "stm32_measurements.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}

    # ---------------- Tier A ----------------
    ext_means = {m: agg(df, m, "ext_accuracy")[0] for m in order} if len(df) else {}
    best = max((v for v in ext_means.values() if not np.isnan(v)), default=None)
    lines = []
    for m in order:
        base = m.split("_kd-")[0]
        p = cx.loc[base, "params_M"] if base in cx.index else np.nan
        g = PUBLISHED_GMACS.get(base, cx.loc[base, "gmacs_fvcore"] if base in cx.index else np.nan)
        gtxt = (f"{g:.2f}" + (r"$^{\dagger}$" if base in PUBLISHED_GMACS else "")) if not np.isnan(g) else XX
        cv = pm(*agg(df, m, "cv_accuracy")[:2]) if len(df) else XX
        ea, es, n = agg(df, m, "ext_accuracy") if len(df) else (np.nan, np.nan, 0)
        ext = pm(ea, es)
        if best is not None and not np.isnan(ea) and abs(ea - best) < 1e-12:
            ext = r"\textbf{" + ext + "}"
        f1 = pm(*agg(df, m, "ext_f1_macro")[:2], d=3, pct=False) if len(df) else XX
        mcc = pm(*agg(df, m, "ext_mcc")[:2], d=3, pct=False) if len(df) else XX
        ece = pm(*agg(df, m, "ext_ece")[:2], d=3, pct=False) if len(df) else XX
        pv = (f"{mc.loc[m, 'p_value']:.3g}" if m in mc.index else ("ref." if m == student else XX))
        if m == student or m == "micropathmamba_tiny":
            lines.append(r"\midrule")
        lines.append(f"{names[m]} & {fmt(p)} & {gtxt} & {cv} & {ext} & {f1} & {mcc} & {ece} & {pv} & {n or XX} \\\\")
    (paper / "tables" / "tierA.tex").write_text(
        "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lrrcccccrc}\n\\toprule\n"
        "Model & Params (M) & GMACs & CV Acc (\\%) & Ext Acc (\\%) & Ext macro-F1 & Ext MCC & Ext ECE & $p$ vs student & runs \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")

    # ---------------- Tier B ----------------
    dep = st.get("deployability", {})
    rows = []
    for m in ["vmamba_t", "mambavision_t", "mobilenetv3_s", "mobilenetv4_m", "micropathmamba"]:
        d = dep.get(m, {})
        w, act = d.get("weights_mb"), d.get("activations_mb")
        fits = XX if w is None or act is None else ("yes" if (act + (w if d.get("weights_internal", True) else 0)) <= 4.2 else "no")
        rows.append(f"{NAMES.get(m, m).replace(' (scratch)', '')} & {d.get('status') or XX} & {fmt(w)} & {fmt(act)} & {fits} & "
                    f"{fmt(d.get('npu_epochs_pct'), 0)} & {d.get('fallback_ops') or XX} \\\\")
    (paper / "tables" / "tierB.tex").write_text(
        "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Model & Status & Weights (MB) & Activ. (MB) & Fits 4.2\\,MB & NPU epochs (\\%) & Fallback ops \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")

    # ---------------- Tier C ----------------
    dev = st.get("on_device", {})
    rows = []
    for m, label in [("mobilenetv3_s", "MobileNetV3-Small"), ("mobilenetv4_s", "MobileNetV4-Conv-S"),
                     ("micropathmamba_cnn", "MicroPathMamba-CNN"), ("micropathmamba_int8", "MicroPathMamba (INT8)"),
                     ("micropathmamba_decayfp", "MicroPathMamba (decay FP)")]:
        d = dev.get(m, {})
        i8 = d.get("int8_acc")
        if i8 is None and d.get("int8_metrics") and Path(d["int8_metrics"]).exists():
            i8 = json.loads(Path(d["int8_metrics"]).read_text())["accuracy"]
        f32 = d.get("fp32_acc")
        delta = None if (i8 is None or f32 is None) else (i8 - f32)
        lat = XX if d.get("latency_ms") is None else (f"{d['latency_ms']:.1f}" + (f"$\\pm${d['latency_sd_ms']:.1f}" if d.get("latency_sd_ms") is not None else ""))
        rows.append(f"{label} & {fmt(f32, 2, True)} & {fmt(i8, 2, True)} & {fmt(delta, 2, True)} & {lat} & "
                    f"{fmt(d.get('energy_mj'))} & {fmt(d.get('ram_mb'))} & {fmt(d.get('agreement_pct'), 1)} \\\\")
    (paper / "tables" / "tierC.tex").write_text(
        "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lccccccc}\n\\toprule\n"
        "Model & FP32 Acc (\\%) & INT8 Acc (\\%) & $\\Delta$ (pp) & Latency (ms) & Energy (mJ) & RAM (MB) & Agree (\\%) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")

    # ---------------- macros for abstract / text ----------------
    teachers = {t: ext_means.get(t, np.nan) for t in MAMBA_TEACHERS}
    teachers = {k: v for k, v in teachers.items() if not np.isnan(v)}
    best_t = max(teachers, key=teachers.get) if teachers else None
    s_acc, _, _ = agg(df, student, "ext_accuracy") if len(df) else (np.nan, 0, 0)
    s_f1, _, _ = agg(df, student, "ext_f1_macro") if len(df) else (np.nan, 0, 0)
    sd = dev.get("micropathmamba_int8", {})
    mv = dep.get("mambavision_t", {})
    macros = {
        "resStudentExtAcc": fmt(s_acc, 2, True), "resStudentExtFone": fmt(s_f1, 3),
        "resTeacherName": NAMES.get(best_t, XX) if best_t else XX,
        "resTeacherExtAcc": fmt(teachers.get(best_t, np.nan), 2, True) if best_t else XX,
        "resStudentIntEightAcc": fmt(sd.get("int8_acc"), 2, True),
        "resStudentLatency": fmt(sd.get("latency_ms"), 1), "resStudentEnergy": fmt(sd.get("energy_mj"), 2),
        "resStudentRAM": fmt(sd.get("ram_mb"), 2), "resMambaWeightsMB": fmt(mv.get("weights_mb"), 1),
        "resMambaFallbackPct": fmt(None if mv.get("npu_epochs_pct") is None else 100 - mv["npu_epochs_pct"], 0),
        "resStudentParams": fmt(cx.loc["micropathmamba", "params_M"], 2) if "micropathmamba" in cx.index else XX,
        "resStudentIntEightMB": fmt(cx.loc["micropathmamba", "int8_weights_MB"], 1) if "micropathmamba" in cx.index else XX,
        "resSTVersion": st.get("st_edge_ai_version") or XX, "resNPUMHz": str(st.get("npu_mhz") or XX),
    }
    (paper / "results_macros.tex").write_text(
        "% generated by scripts/fill_manuscript.py - do not edit by hand\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    # ---------------- plain summary for the write-up ----------------
    summ = ["# Results summary (auto-generated)", ""]
    if len(df):
        t = df.groupby("model")[["cv_accuracy", "ext_accuracy", "ext_f1_macro", "ext_mcc", "ext_ece"]].agg(["mean", "std", "count"]).round(4)
        summ += ["## Per-model (mean, std, n)", "```", t.to_string(), "```"]
    summ += ["", "## Macros", "```", json.dumps(macros, indent=1), "```"]
    if st:
        summ += ["", "## STM32 measurements", "```", json.dumps(st, indent=1), "```"]
    (paper / "results_summary.md").write_text("\n".join(summ))
    filled = sum(v != XX for v in macros.values())
    print(f"runs found: {len(df)} | models: {df.model.nunique() if len(df) else 0} | macros filled: {filled}/{len(macros)}")
    print("wrote paper/tables/tierA.tex, tierB.tex, tierC.tex, paper/results_macros.tex, paper/results_summary.md")


if __name__ == "__main__":
    main()
