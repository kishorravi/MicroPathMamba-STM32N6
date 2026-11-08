# Experiment guide: from zero to a results-filled manuscript

## Quick path (what we actually run)

The Mac and Kaggle split the work:

| Where | What | How |
|---|---|---|
| **Kaggle** (free T4 GPU) | All training, including the MambaVision teacher | Upload `kaggle/micropathmamba_kaggle.ipynb` to Kaggle, set GPU T4 x2 and Internet on, then Save & Run All. Download `kaggle_results.zip` when it finishes. |
| **Mac** | Checks, ONNX/INT8, accuracy of the INT8 model, manuscript tables, STM32 | Unzip the Kaggle results into this folder, then run `caffeinate -i bash scripts/mac_run.sh`. |
| **STM32N6570-DK** | Latency, energy and memory on the board | Follow `stm32/README.md`, then fill in `results/stm32_measurements.json`. |

The phase sections below describe the full plan in detail.

Every result lands in `runs/` or `results/`. After **each** phase, run:

```bash
python scripts/stats.py --runs runs --ref micropathmamba_kd-mambavision_t   # once the student exists
python scripts/fill_manuscript.py                                            # tables + abstract numbers
```

Then recompile `paper/manuscript.tex` on Overleaf. Anything not measured yet stays a red **XX**, so the paper is always honest about what is done.

Times are rough estimates for one RTX 4090 with data on an SSD.

**Which machine runs what**

| Machine | Device | Runs |
|---|---|---|
| Linux PC or cloud with an NVIDIA GPU (Kaggle/Colab are fine) | CUDA | everything, and the **only** option for MambaVision and VMamba, which need `mamba-ssm` CUDA kernels |
| Apple Silicon Mac | MPS, chosen automatically | the student, its no-SSM ablation and the timm baselines (MobileNet, ConvNeXt, Swin, EfficientViT, ResNet) |

On a Mac:

- Install with `pip install torch torchvision timm scikit-learn scipy pandas matplotlib pillow onnx onnxruntime`. Skip `mamba-ssm`, `causal-conv1d` and `mambavision`.
- Use `--workers 4`.
- A MacBook Air has no fan and slows down under long load, so plan on overnight runs for the student.

---

## Phase 0: setup and smoke test (≈ 1 hour)

```bash
conda create -n mambapath python=3.10 -y && conda activate mambapath
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_ssd_equivalence.py            # must print PASS
python -m models.micropathmamba                 # must print params: 2.35 M | output: (2, 9)
python scripts/make_folds.py --data /data/NCT-CRC-HE-100K --out folds.csv --seed 123

# 2-3 minutes each, on a tiny subset; outputs go to runs_smoke/ and never reach the paper
for M in micropathmamba mobilenetv3_s mambavision_t; do
  python train.py --model $M --fold 0 --seed 1 --epochs 1 --limit-train 900 --limit-val 450 \
      --data /data/NCT-CRC-HE-100K --ext /data/CRC-VAL-HE-7K --out runs
done
```

What to check before moving on:

- **Errors:** nothing crashes.
- **MambaVision:** `mambavision_t` loads. If not, `mamba-ssm` did not build; check that the CUDA toolkit matches your PyTorch.
- **Outputs:** each `runs_smoke/*/f0_s1/` folder contains `metrics.json`.
- **Export:**
  ```bash
  python scripts/export_onnx.py --model micropathmamba --ckpt runs_smoke/micropathmamba/f0_s1/best.pt --out deploy/smoke.onnx
  ```
  It must print `dynamic/control-flow ops: none`.

**Send me anything that fails.** The PyTorch code compiles, but it has never run on a GPU.

## Phase 1: first real numbers (≈ 10–14 GPU hours, fold 0, seed 1)

```bash
D="--data /data/NCT-CRC-HE-100K --ext /data/CRC-VAL-HE-7K --fold 0 --seed 1"
python train.py --model mambavision_t $D                    # teacher
python train.py --model mobilenetv3_s $D                    # MCU baseline
python train.py --model micropathmamba_cnn $D               # ablation without SSM
python train.py --model micropathmamba $D                   # student, no distillation
python train.py --model micropathmamba $D --teacher mambavision_t --teacher-ckpt runs/mambavision_t/f0_s1/best.pt
python scripts/complexity.py --models micropathmamba micropathmamba_cnn mambavision_t mobilenetv3_s
python scripts/fill_manuscript.py
```

**Decision point.** Compare the student with and without distillation, and the student against the no-SSM ablation, on the external 7K set:

- **If distillation and the SSM blocks each help**, continue to Phase 2.
- **If they don't help**, send me `paper/results_summary.md` before spending more GPU time. We may need a larger student (`micropathmamba_w15`) or a different distillation temperature.

## Phase 2: full benchmark (≈ 80–120 GPU hours, can run over several nights)

```bash
bash run_all.sh        # edit DATA/EXT paths at the top; comment out runs you already have
```

If time is short, run seeds 2–3 only for the four models that matter most:

- the best Mamba teacher;
- the distilled student;
- MobileNetV3-Small;
- ConvNeXt V2-T.

Everything else can stay at seed 1. The `runs` column in Table 1 shows readers how many runs each mean is based on.

## Phase 3: STM32N6 (≈ 1–2 days with the board)

1. Export and quantize the student, and evaluate the INT8 model on the host. The commands are in `run_all.sh` under "Export + INT8".
2. Repeat the export and quantization for the MobileNetV3-Small baseline:
   - export its fold-0 checkpoint with `export_onnx.py`;
   - quantize it with `quantize_onnx.py`;
   - evaluate it with `evaluate.py --onnx`, writing to `results/mobilenetv3_s_int8`.
3. Run the ST Edge AI analyse / generate / validate steps and the energy measurement in `stm32/README.md`.
4. Copy `results_template/stm32_measurements.json` to `results/stm32_measurements.json` and type in the measured values. Leave `null` for anything you have not measured.
5. Run `python scripts/fill_manuscript.py` again.

## What to send me after each phase

1. **`paper/results_summary.md`** (generated). This is enough for me to write the Results and Discussion text.
2. **`results/stats/`**: the summary, bootstrap and McNemar CSVs.
3. **Confusion matrix:** `results/student_int8/confusion_matrix.png`.
4. **Grad-CAM images:** any you produce.
5. **ST Edge AI reports:** for the deployability text, especially the error message if a Mamba teacher fails.

I will then:

- **Results:** write Sections 4–6 from the real numbers;
- **Figures:** produce Figure 1 (architecture) and the result figures;
- **Abstract:** adjust its claims to what the data actually shows.
