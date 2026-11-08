#!/bin/bash
# Mac (Apple Silicon) runner: checks + deployment side. Heavy training runs on Kaggle (kaggle/).
#   cd ~/PycharmProjects/MicroPathMamba-STM32N6 && caffeinate -i bash scripts/mac_run.sh
# 1-4: environment, data, checks, smoke test incl. ONNX + INT8 (about 10 minutes).
# 5:   if Kaggle results were copied into runs/, export + INT8-quantize the trained student,
#      measure INT8 accuracy on CRC-VAL-HE-7K, measure complexity and refresh the manuscript.
# Set MAC_TRAIN=1 to also train the small models here (slow: ~20+ min per epoch on a MacBook Air).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/mac_run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
DATA="${DATA:-$HOME/data}"
TRAIN="$DATA/NCT-CRC-HE-100K"; EXT="$DATA/CRC-VAL-HE-7K"
stamp() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

stamp "1. Python environment (.venv in the project folder)"
PY=python3; [ -x "$HOME/miniforge3/bin/python3" ] && PY="$HOME/miniforge3/bin/python3"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision timm scikit-learn scipy pandas matplotlib pillow onnx onnxruntime fvcore
python -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| Apple GPU:', torch.backends.mps.is_available())"

stamp "2. Data in $DATA"
mkdir -p "$DATA"
if [ ! -d "$EXT" ]; then
  [ -f "$DATA/CRC-VAL-HE-7K.zip" ] || curl -L -o "$DATA/CRC-VAL-HE-7K.zip" "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1"
  (cd "$DATA" && unzip -q -o CRC-VAL-HE-7K.zip)
fi
if [ ! -d "$TRAIN" ]; then
  [ -f "$DATA/NCT-CRC-HE-100K.zip" ] || curl -L -C - -o "$DATA/NCT-CRC-HE-100K.zip" "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1"
  (cd "$DATA" && unzip -q -o NCT-CRC-HE-100K.zip)
fi
for d in "$TRAIN" "$EXT"; do echo "$d: $(find "$d" -type f | wc -l | tr -d ' ') images"; done

stamp "3. Checks"
python tests/test_ssd_equivalence.py
python -m models.micropathmamba
[ -f folds.csv ] || python scripts/make_folds.py --data "$TRAIN" --out folds.csv --seed 123

stamp "4. Smoke test: train 1 epoch on 900 patches, export ONNX, quantize INT8"
python train.py --model micropathmamba --fold 0 --seed 1 --epochs 1 --limit-train 900 --limit-val 450 \
    --workers 4 --data "$TRAIN" --ext "$EXT"
python scripts/export_onnx.py --model micropathmamba --ckpt runs_smoke/micropathmamba/f0_s1/best.pt --out deploy/smoke_fp32.onnx
python scripts/quantize_onnx.py --fp32 deploy/smoke_fp32.onnx --out deploy/smoke_int8.onnx --fold 0 --n-calib 64

if [ "${MAC_TRAIN:-0}" = "1" ]; then
  stamp "optional: train small models on the Mac"
  for M in micropathmamba_cnn micropathmamba mobilenetv3_s; do
    [ -f "runs/$M/f0_s1/metrics.json" ] && { echo "skip $M"; continue; }
    python train.py --model "$M" --fold 0 --seed 1 --workers 4 --data "$TRAIN" --ext "$EXT"
  done
fi

stamp "5. Deployment side for trained models found in runs/"
STUDENT=$(ls -d runs/micropathmamba_kd-*/f0_s1 2>/dev/null | head -1 || true)
[ -z "$STUDENT" ] && [ -f runs/micropathmamba/f0_s1/best.pt ] && STUDENT=runs/micropathmamba/f0_s1
if [ -n "$STUDENT" ] && [ -f "$STUDENT/best.pt" ]; then
  echo "student checkpoint: $STUDENT"
  python scripts/export_onnx.py --model micropathmamba --ckpt "$STUDENT/best.pt" --out deploy/student_fp32.onnx
  python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8.onnx --fold 0
  python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8_decayfp.onnx --fold 0 --keep-decay-float
  python evaluate.py --onnx deploy/student_int8.onnx --data "$EXT" --out results/student_int8 --workers 4
  python evaluate.py --onnx deploy/student_int8_decayfp.onnx --data "$EXT" --out results/student_int8_decayfp --workers 4
  if [ -f runs/mobilenetv3_s/f0_s1/best.pt ]; then
    python scripts/export_onnx.py --model mobilenetv3_s --ckpt runs/mobilenetv3_s/f0_s1/best.pt --out deploy/mobilenetv3_s_fp32.onnx
    python scripts/quantize_onnx.py --fp32 deploy/mobilenetv3_s_fp32.onnx --out deploy/mobilenetv3_s_int8.onnx --fold 0
    python evaluate.py --onnx deploy/mobilenetv3_s_int8.onnx --data "$EXT" --out results/mobilenetv3_s_int8 --workers 4
  fi
else
  echo "No trained student yet. Copy the Kaggle 'runs' folder into this project, then run this script again."
fi
python scripts/complexity.py --models micropathmamba micropathmamba_cnn micropathmamba_tiny mobilenetv3_s \
    mobilenetv4_m convnextv2_t swin_t efficientvit_b1 resnet50 --out results/complexity.csv
python scripts/stats.py --runs runs --ref "$(basename "$(dirname "${STUDENT:-runs/none/f0_s1}")")" || echo "(stats skipped: not enough runs yet)"
python scripts/fill_manuscript.py
stamp "DONE. Log: $LOG"
