# From GPU to Microcontroller: Vision Mamba for On-Device Colorectal Histopathology on STM32N6

Code for the three-tier study:

| Tier | Question | Output |
|---|---|---|
| **A** | How do VMamba-T and MambaVision T/T2/S compare with size-matched CNN/Transformer baselines? | Accuracy, MCC, AUC, ECE (internal 3-fold CV + external CRC-VAL-HE-7K), params, GFLOPs, GPU latency |
| **B** | Can these models be deployed on the STM32N6 at all? | ST Edge AI analysis: memory, NPU vs CPU operator mapping, failure modes |
| **C** | Can a small Mamba keep the accuracy and run on the NPU? | **MicroPathMamba** (~0.95 M params, ~1 MB INT8), distilled from the best teacher; measured on-board latency, energy, RAM, INT8 drop |

## The student: MicroPathMamba

- **Stages 1–4:** MobileNetV2-style inverted residual convolutions with BatchNorm and ReLU6, which map to the NPU.
- **Stage 5:** two MambaVision-style blocks running on 7 × 7 = 49 tokens. The selective scan is written in **matrix (state-space-duality) form**:

  `Y = (exp(-|c_i − c_j|) ⊙ C Bᵀ) (Δ ⊙ X)`, with `c = (Δ·A) @ triu(1)`

  This uses only MatMul / Mul / Sub / Abs / Exp / Sigmoid. There is no CUDA kernel, Loop, Scan or CumSum.
- **Proof:** `tests/test_ssd_equivalence.py` shows the matrix form equals a forward + backward selective scan (relative error ≈ 1e-15). Run it with NumPy alone.
- **Quantisation-friendly choices:** BatchNorm instead of LayerNorm, and `Δ = Δmax·sigmoid(·)` instead of softplus.

## Setup

```bash
conda create -n mambapath python=3.10 -y && conda activate mambapath
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # mamba-ssm builds CUDA kernels: Linux + NVIDIA GPU
python tests/test_ssd_equivalence.py     # sanity check
python -m models.micropathmamba          # prints parameter count and output shape
```

Datasets (Kather et al., 2019, Zenodo record 1214456):

- `NCT-CRC-HE-100K` for training and internal 3-fold CV;
- `CRC-VAL-HE-7K` for external testing only.

Optionally, `NCT-CRC-HE-100K-NONORM` (not stain-normalised) supports a stain-robustness experiment.

## Run

`run_all.sh` lists every run in order. Main entry points:

```bash
python scripts/make_folds.py --data NCT-CRC-HE-100K --out folds.csv --seed 123
python train.py --model mambavision_t --fold 0 --seed 1 --data NCT-CRC-HE-100K --ext CRC-VAL-HE-7K
python train.py --model micropathmamba --fold 0 --seed 1 --teacher mambavision_t \
    --teacher-ckpt runs/mambavision_t/f0_s1/best.pt --data NCT-CRC-HE-100K --ext CRC-VAL-HE-7K
python scripts/export_onnx.py --model micropathmamba --ckpt runs/.../best.pt --out deploy/student_fp32.onnx
python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8.onnx
python evaluate.py --onnx deploy/student_int8.onnx --data CRC-VAL-HE-7K --out results/student_int8
python scripts/complexity.py --models micropathmamba mambavision_t ... --out results/complexity.csv
python scripts/stats.py --runs runs --ref micropathmamba_kd-mambavision_t
```

The on-board steps (ST Edge AI analyse/generate/validate, energy measurement) are in `stm32/README.md`.

## Compute budget

The full grid is 10 Tier-A models × 3 folds × 3 seeds, plus 3 student variants. On one RTX 4090 that is roughly 100–150 GPU hours. If time is short:

- run all models with seed 1 only;
- add seeds 2–3 for the top teacher, the student and two baselines.

## Honest-reporting notes (reviewers check these)

- **Patch-level CV.** NCT-CRC-HE-100K has no patient IDs, so internal CV is patch-level and optimistic. Lead with the external 7K results.
- **FLOPs for Mamba teachers.** fvcore cannot count custom selective-scan kernels. Use the authors' published FLOPs for the teachers and say so.
- **Failed deployments.** Report Tier-B failures (out of memory, unsupported operators) as results, not as missing data.
- **Toolchain.** Keep the ST Edge AI Core version, NPU/CPU clocks and memory placement in every hardware table.
