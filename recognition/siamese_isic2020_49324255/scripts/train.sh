#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/content/PatternAnalysis-2025}
TRAIN_CSV=${TRAIN_CSV:-$REPO/data/train.csv}
VAL_CSV=${VAL_CSV:-$REPO/data/val.csv}
TEST_CSV=${TEST_CSV:-$REPO/data/test.csv}
IMAGES_ROOT=${IMAGES_ROOT:-/content/ISIC2020_224/train-image/image}
OUT_DIR=${OUT_DIR:-/content/drive/MyDrive/COMP3710/siamese_runs/exp_best_49324255}

python $REPO/recognition/siamese_isic2020_49324255/train.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --images_root "$IMAGES_ROOT" \
  --out_dir "$OUT_DIR" \
  --epochs 6 \
  --batch_size 48 \
  --img_size 224 \
  --num_workers 2 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --backbone efficientnet_b0 \
  --embed_dim 128 \
  --margin 1.0 \
  --seed 42 \
  --auto_threshold