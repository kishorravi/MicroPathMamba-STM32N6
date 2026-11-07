#!/usr/bin/env bash
# Full experiment plan. Run on Linux with an NVIDIA GPU (MambaVision/VMamba need mamba-ssm CUDA kernels).
# Edit the two dataset paths, then run sections one at a time; each run writes runs/<model>/f<fold>_s<seed>/.
set -euo pipefail
DATA=/data/NCT-CRC-HE-100K
EXT=/data/CRC-VAL-HE-7K
SEEDS="1 2 3"; FOLDS="0 1 2"

python scripts/make_folds.py --data $DATA --out folds.csv --seed 123

# ---- Tier A: teachers and size-matched baselines (3 folds x 3 seeds) ----
for M in mambavision_t mambavision_t2 mambavision_s vmamba_t convnextv2_t swin_t efficientvit_b1 \
         mobilenetv4_m mobilenetv3_s resnet50; do
  for F in $FOLDS; do for S in $SEEDS; do
    python train.py --model $M --fold $F --seed $S --data $DATA --ext $EXT
  done; done
done

# ---- Tier C: student from scratch, student with distillation, and the no-SSM ablation ----
TEACHER=mambavision_t   # replace with the best Tier-A Mamba teacher on the external set
for F in $FOLDS; do for S in $SEEDS; do
  python train.py --model micropathmamba --fold $F --seed $S --data $DATA --ext $EXT
  python train.py --model micropathmamba_cnn --fold $F --seed $S --data $DATA --ext $EXT
  python train.py --model micropathmamba --fold $F --seed $S --data $DATA --ext $EXT \
      --teacher $TEACHER --teacher-ckpt runs/$TEACHER/f${F}_s${S}/best.pt
done; done

# ---- Export + INT8 (fold 0, seed 1 model is the deployed one) ----
CK=runs/micropathmamba_kd-$TEACHER/f0_s1/best.pt
python scripts/export_onnx.py --model micropathmamba --ckpt $CK --out deploy/student_fp32.onnx
python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8.onnx --fold 0
python scripts/quantize_onnx.py --fp32 deploy/student_fp32.onnx --out deploy/student_int8_decayfp.onnx --fold 0 --keep-decay-float
python evaluate.py --onnx deploy/student_int8.onnx --data $EXT --out results/student_int8
python evaluate.py --onnx deploy/student_int8_decayfp.onnx --data $EXT --out results/student_int8_decayfp

# ---- Tier B: teacher exports for the deployability table ----
python scripts/export_teacher_onnx.py --model mambavision_t --ckpt runs/mambavision_t/f0_s1/best.pt \
    --out deploy/teachers/mambavision_t_fp32.onnx || echo "export failed: record this in Table Deployability"

# ---- Complexity and statistics ----
python scripts/complexity.py --models micropathmamba micropathmamba_cnn mambavision_t mambavision_t2 \
    mambavision_s vmamba_t convnextv2_t swin_t efficientvit_b1 mobilenetv4_m mobilenetv3_s resnet50
python scripts/stats.py --runs runs --ref micropathmamba_kd-$TEACHER
# ---- Then follow stm32/README.md for on-board latency, energy and agreement ----
