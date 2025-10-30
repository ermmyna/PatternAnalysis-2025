# Siamese Network for ISIC 2020 Melanoma Classification

**Student:** Ermmyna Roselee Shah (49324255)  
**Repository:** PatternAnalysis-2025  
**Branch:** topic9-siamese  
**Project Folder:** `recognition/siamese_isic2020_49324255/`

---

## 1. Project Overview

This project implements a Siamese neural network for binary melanoma classification on dermoscopic images from the ISIC 2020 dataset. The task presents severe class imbalance (1.7% malignant) and is framed as a metric learning problem: learning embeddings where same-diagnosis lesions are similar and different-diagnosis lesions are dissimilar. The Siamese architecture with contrastive loss learns discriminative 128-dimensional embeddings, followed by a class-weighted linear probe for classification. This approach is well-suited for limited GPU/time constraints: it reuses pretrained EfficientNet-B0 features, trains only a projection head and probe (minimal parameters), and handles class imbalance through threshold optimization rather than complex sampling strategies.

---

## 2. Data

**Dataset:** [ISIC 2020 JPG 224×224 RESIZED](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized)  
- ~33,126 dermoscopic images (224×224 JPEGs)
- Pre-resized for computational efficiency

**Splits:** Patient-aware (no leakage) splits provided as CSVs in `data/`:
- `data/train.csv` (22,353 samples)
- `data/val.csv` (4,842 samples)
- `data/test.csv` (5,081 samples)

**CSV Columns:**
- `image_path`: relative path to image
- `target`: binary target (0=benign, 1=malignant)
- `patient_id`: patient identifier
- `lesion_id`: lesion identifier (optional, unused by pipeline)
- `image_name`: image filename

**Note:** Large image data is not committed to Git. Users must download the dataset and specify `--images_root` pointing to the image directory.

---

## 3. Method & Architecture

### Architecture
```
Image 1 (224×224×3) ──┐
                      ├──> EfficientNet-B0 (frozen) ──> Projection Head ──> L2 Norm ──> Embedding 1 (128D)
Image 2 (224×224×3) ──┘                                  (BN→ReLU→Linear)
                                                                │
                                                                ├──> Cosine Distance ──> Contrastive Loss
                                                                │
                                                         Embedding 2 (128D)
                                                                │
                                                                └──> Linear Probe (128→2) ──> Cross-Entropy + Class Weights
```

### Components

**Embedding Network:**
- Backbone: EfficientNet-B0 (ImageNet pretrained, frozen by default)
- Projection head: BatchNorm → ReLU → Linear(1280→128)
- L2 normalization of embeddings

**Siamese Head:**
- Cosine distance: `d = 1 - cos(z₁, z₂)`
- Contrastive loss: `L = y·d² + (1-y)·max(0, m-d)²` where `y=1` for same class, `m=1.0` margin

**Linear Probe:**
- Trained separately on frozen embeddings
- Class-weighted cross-entropy to handle imbalance
- Outputs P(malignant)

### Threshold Optimization

After training, sweep thresholds 0.01→0.99 on validation set to maximize F1 score. Apply optimal threshold (0.23) to test set. Report both ROC-AUC (rank-based, imbalance-robust) and PR-AUC (AUCPR, more informative under severe imbalance).

### Optional Features

- `--pos_ratio`: Control positive pair sampling ratio (default 0.5)
- `--unfreeze_after N`: Unfreeze last backbone block after N epochs with reduced LR (0.1×)

---

## 4. Environment & Requirements

**Hardware:** Google Colab with GPU (T4/A100)  
**Software:** Python 3.12, PyTorch 2.0+

**Dependencies:**
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
umap-learn>=0.5.3
tqdm
```

---

## 5. How to Reproduce (Colab)

### Setup
```python
# Clone repository
!git clone https://github.com/ermmyna/PatternAnalysis-2025.git
%cd PatternAnalysis-2025
!git checkout topic9-siamese

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
REPO = "/content/PatternAnalysis-2025"
TRAIN_CSV = f"{REPO}/data/train.csv"
VAL_CSV = f"{REPO}/data/val.csv"
TEST_CSV = f"{REPO}/data/test.csv"
IMAGES_ROOT = "/content/ISIC2020_224/train-image/image"
OUT_DIR = "/content/drive/MyDrive/COMP3710/siamese_runs/exp_best_49324255"
```

### Download Dataset (if not present)
```python
import os
import glob

if not os.path.exists(IMAGES_ROOT) or len(glob.glob(f"{IMAGES_ROOT}/*.jpg")) < 30000:
    print("Dataset not found. Downloading...")
    
    # Setup Kaggle API
    !pip install -q kaggle
    assert os.path.exists("/content/kaggle.json"), "Upload kaggle.json to /content/"
    !mkdir -p /root/.kaggle
    !cp /content/kaggle.json /root/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json
    
    # Download and extract
    !mkdir -p /content/ISIC2020_224
    !kaggle datasets download -d nischaydnk/isic-2020-jpg-224x224-resized -p /content/ISIC2020_224
    !unzip -q /content/ISIC2020_224/isic-2020-jpg-224x224-resized.zip -d /content/ISIC2020_224
    
    # Verify
    n_images = len(glob.glob(f"{IMAGES_ROOT}/*.jpg"))
    print(f"✓ Downloaded {n_images} images")
else:
    print(f"✓ Dataset already present at {IMAGES_ROOT}")
```

### Verify CSVs
```python
import os
for csv in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    assert os.path.exists(csv), f"Missing: {csv}"
print("✓ All split CSVs found")
```

### Dataloader Smoke Test
```python
import sys
sys.path.insert(0, f"{REPO}/recognition/siamese_isic2020_49324255")

from dataset import make_dataloader

dl = make_dataloader(TRAIN_CSV, batch_size=4, img_size=224, augment=True,
                     images_root=IMAGES_ROOT, num_workers=0)
img1, img2, same = next(iter(dl))
print(f"✓ Batch shapes: {img1.shape}, {img2.shape}, {same.shape}")
print(f"  Pair labels: {same.numpy()}")
```

### Test Driver Scripts (One Command)

For convenience, use the provided shell scripts:

**Eval-only (recompute threshold):**
```bash
bash recognition/siamese_isic2020_49324255/scripts/eval.sh
```

**Training (6 epochs):**
```bash
bash recognition/siamese_isic2020_49324255/scripts/train.sh
```

These scripts use environment variables (`REPO`, `TRAIN_CSV`, etc.) with sensible defaults. Override as needed.

### Eval-Only Run (Recompute Threshold)
```bash
python recognition/siamese_isic2020_49324255/train.py \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --test_csv $TEST_CSV \
  --images_root $IMAGES_ROOT \
  --out_dir $OUT_DIR \
  --epochs 0 \
  --batch_size 48 \
  --img_size 224 \
  --num_workers 2 \
  --auto_threshold
```

**Note:** If `$OUT_DIR/checkpoints/best.pt` does not exist, the script exits with an error; to reproduce the reported metrics, first run a short training (6–8 epochs) to produce `best.pt`, or place a compatible checkpoint at that path. Checkpoints are not committed to git.

**Outputs:**
- `$OUT_DIR/metrics.json` (test AUC, AUCPR, accuracy, F1, best_threshold)
- `$OUT_DIR/figures/` (ROC, PR val/test, confusion matrix, UMAP)

### Training Run
```bash
python recognition/siamese_isic2020_49324255/train.py \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --test_csv $TEST_CSV \
  --images_root $IMAGES_ROOT \
  --out_dir $OUT_DIR \
  --epochs 6 \
  --batch_size 48 \
  --img_size 224 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --backbone efficientnet_b0 \
  --embed_dim 128 \
  --margin 1.0 \
  --num_workers 2 \
  --seed 42
```

**Outputs:**
- `$OUT_DIR/checkpoints/best.pt` (best model by validation AUC)
- `$OUT_DIR/figures/` (ROC, PR test, confusion matrix, learning curves, UMAP)
- `$OUT_DIR/metrics.json`

### Prediction/Inference
```bash
python recognition/siamese_isic2020_49324255/predict.py \
  --csv $TEST_CSV \
  --images_root $IMAGES_ROOT \
  --checkpoint $OUT_DIR/checkpoints/best.pt \
  --out_dir $OUT_DIR \
  --img_size 224 \
  --max_samples 100
```

**Output:** `$OUT_DIR/predictions.csv` with columns: `image_name`, `true_label`, `predicted_label`, `prob_melanoma`

---

## 6. Results

### Test Set Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **AUC** | **0.837** | Strong discrimination, imbalance-robust |
| **AUCPR** | **0.139** | 6× better than random (baseline ≈0.023) |
| **Accuracy** | **0.952** | High but inflated by class imbalance |
| **F1 @ threshold=0.23** | **0.234** | Optimal threshold from validation |

**Interpretation:** The optimal threshold (0.23) is significantly lower than the default (0.5) due to severe class imbalance (1.7% malignant). This threshold balances precision and recall for the minority class, improving F1 from near-zero (at 0.5) to 0.234. ROC-AUC (0.837) demonstrates strong ranking ability, while AUCPR (0.139) reflects performance under extreme imbalance—approximately 6× the random-guess baseline AUCPR of ~0.023 (test prevalence).

### Visual Results

#### ROC Curve
![ROC Curve](results/figures/roc_curve.png)

The ROC curve (AUC=0.837) shows strong discrimination across all threshold settings, with the curve well above the random baseline (diagonal). The smooth curve indicates stable probability calibration.

#### Precision-Recall Curve
![PR Curve](results/figures/pr_curve_test.png)

The PR curve (AP=0.139) reflects the challenge of the 2.3% prevalence. The steep initial drop shows the difficulty maintaining precision as recall increases, typical of severe class imbalance scenarios.

#### Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

At threshold=0.23: 37/116 true positives (31.9% sensitivity), 4,802/4,965 true negatives (96.7% specificity). The 79 false negatives indicate the model misses ~68% of melanomas, highlighting the need for ensemble approaches in clinical deployment.

#### Learning Curves
![Learning Curves](results/figures/learning_curves.png)

Training converged smoothly with early stopping at epoch 6. Contrastive loss decreased steadily (0.23→0.19) while validation AUC peaked at epoch 2 (0.825). The gap between positive and negative pair distances increased from ~0.03 to ~0.19, indicating successful embedding space separation.

#### Embedding Space (UMAP)
![UMAP Embeddings](results/figures/embeddings_umap.png)

UMAP projection of 128D embeddings shows clear separation between benign (blue) and malignant (red) clusters, with some overlap in boundary regions where visual similarity is high. This visualization confirms the Siamese network learned diagnostically meaningful features.

### Outputs

**Figures:** `$OUT_DIR/figures/`
- `roc_curve.pdf` - ROC curve with AUC=0.837
- `pr_curve_val.pdf` - Validation PR curve (eval-only mode)
- `pr_curve_test.pdf` - Test PR curve with AP=0.139
- `confusion_matrix.pdf` - Test set confusion matrix at threshold=0.23
- `learning_curves.pdf` - Training loss and validation metrics
- `embeddings_umap.pdf` - UMAP projection of 128D embeddings

**Metrics:** `$OUT_DIR/metrics.json` - Complete test results and hyperparameters

**Predictions:** `$OUT_DIR/predictions.csv` - Per-image predictions with probabilities

---

## 7. Ablations / Notes

**Threshold Selection:** Validation-optimized threshold (0.23) vs. fixed 0.5 significantly improves F1 and recall on minority class. At 0.5, F1≈0 due to low positive predictions.

**Class Weighting:** Linear probe uses class-weighted cross-entropy (weight ratio ≈58:1) to handle imbalance during probe training.

**Optional Experiments:**
- `--pos_ratio` adjusts positive/negative pair sampling (default 0.5)
- `--unfreeze_after N` enables backbone fine-tuning after N epochs with 0.1× learning rate

---

## 8. Limitations & Future Work

### Limitations

1. **Class Imbalance Sensitivity:** Model performance heavily depends on threshold calibration; AUCPR (0.139) indicates room for improvement on minority class precision-recall trade-off
2. **Threshold Drift:** Optimal threshold may not generalize across different prevalence rates or acquisition sites
3. **Domain Shift:** Training on pooled multi-center data may not account for site-specific artifacts or imaging protocols

### Future Work

- **Focal Loss:** Address class imbalance during training rather than post-hoc thresholding
- **Calibration:** Apply Platt scaling or isotonic regression for better probability estimates
- **Test-Time Augmentation (TTA):** Average predictions across augmented views
- **Hard Negative Mining:** Focus training on difficult negative pairs near decision boundary
- **Stratified Evaluation:** Report per-site or per-anatomical-location performance to detect domain shift

---

## 9. How to Use the Model (Prediction/Retrieval)

### Generate Predictions
```bash
python recognition/siamese_isic2020_49324255/predict.py \
  --csv data/test.csv \
  --images_root /path/to/images \
  --checkpoint runs/exp1/checkpoints/best.pt \
  --out_dir runs/exp1 \
  --img_size 224 \
  --max_samples 1000
```

**Output:** `predictions.csv` with image names, true labels, predicted labels, and melanoma probabilities.

### Preprocessing

Input images are automatically:
1. Resized to 224×224
2. Normalized with ImageNet statistics: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

Apply the same preprocessing as training for consistent results.

---

## 10. Ethical & Data-Use Considerations

ISIC 2020 images are de-identified and released for research purposes under appropriate licenses. This model is intended for research only and not for clinical use. The dataset may contain site-specific or device-specific biases that could affect generalization. Prospective validation on diverse populations and clinical settings is required before any clinical deployment.

---

## 11. Academic Integrity / AI-Usage Statement

Generative AI (Claude 3.5 Sonnet) was used for:
- Coding assistance (PyTorch boilerplate, error handling, refactoring)
- Documentation (README structure, docstrings, comments)
- Debugging (PyTorch 2.6 compatibility fixes, optimizer suggestions)

All experimental design, hyperparameter selection, threshold optimization, data splits, training runs, and metric validation were performed by the student. Data handling followed course guidelines for patient-aware splitting. Results are reproducible via the provided commands and seed (42).

---

## 12. CLI Flags Reference (Appendix)

### train.py Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--train_csv` | `data/train.csv` | Path to training CSV |
| `--val_csv` | `data/val.csv` | Path to validation CSV |
| `--test_csv` | `data/test.csv` | Path to test CSV |
| `--images_root` | `ISIC2020/train` | Root directory containing images |
| `--out_dir` | `recognition/siamese_isic2020_49324255/runs/exp1` | Output directory for checkpoints and figures |
| `--epochs` | `15` | Number of training epochs (0 for eval-only) |
| `--batch_size` | `64` | Batch size for pair dataloader |
| `--img_size` | `224` | Image resize dimension (square) |
| `--lr` | `1e-3` | Learning rate for AdamW optimizer |
| `--weight_decay` | `1e-4` | L2 regularization weight |
| `--num_workers` | `2` | Number of dataloader workers |
| `--backbone` | `efficientnet_b0` | Timm backbone architecture |
| `--embed_dim` | `128` | Embedding dimension |
| `--margin` | `1.0` | Contrastive loss margin |
| `--unfreeze_after` | `0` | Unfreeze backbone after N epochs (0=never) |
| `--pos_ratio` | `0.5` | Probability of positive pairs |
| `--auto_threshold` | `False` | Auto-compute optimal threshold (flag, no value) |
| `--seed` | `42` | Random seed for reproducibility |

### predict.py Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | (required) | Path to CSV for inference |
| `--images_root` | (required) | Root directory containing images |
| `--checkpoint` | (required) | Path to trained checkpoint (.pt) |
| `--out_dir` | (required) | Output directory for predictions.csv |
| `--img_size` | `224` | Image resize dimension |
| `--max_samples` | `100` | Maximum samples to process |

### Project Structure
```
recognition/siamese_isic2020_49324255/
├── modules.py              # EmbeddingNet, SiameseHead, ContrastiveLoss
├── dataset.py              # ISICPairDataset, make_dataloader
├── utils.py                # Metrics, visualization, seed setting
├── train.py                # Training pipeline with dual optimization
├── predict.py              # Inference script
├── README.md               # This file
├── scripts/
│   ├── eval.sh             # Eval-only driver script
│   └── train.sh            # Training driver script
└── results/                # Not committed (user-generated)
    ├── checkpoints/
    │   └── best.pt
    ├── figures/
    │   ├── roc_curve.pdf
    │   ├── pr_curve_val.pdf
    │   ├── pr_curve_test.pdf
    │   ├── confusion_matrix.pdf
    │   ├── learning_curves.pdf
    │   └── embeddings_umap.pdf
    └── metrics.json
```
---

## 13. References

- **EfficientNet:** Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
- **Contrastive Loss / Siamese Networks:** Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality reduction by learning an invariant mapping. CVPR.
- **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.
- **ISIC Dataset:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. Scientific Data.
- **ISIC 2020 Challenge:** https://challenge2020.isic-archive.com/

---

## 14. Submission Notes (for Markers)

**Repository:** https://github.com/ermmyna/PatternAnalysis-2025  
**Branch:** `topic9-siamese`  
**PR Target:** `topic-recognition` (as per assignment instructions)

**Important:**
- No datasets or large model files are committed (`.gitignore` applied)
- Images must be downloaded separately via Kaggle link
- Eval-only mode (`--epochs 0 --auto_threshold`) reproduces reported metrics from checkpoint
- All code in `recognition/siamese_isic2020_49324255/` is executable as provided

**To Verify Results:**
1. Download dataset from Kaggle link (Section 2)
2. Run training command (Section 5) or use provided script: `bash scripts/train.sh`
3. Run eval-only command or use: `bash scripts/eval.sh`
4. Compare `metrics.json` with reported values (Section 6)

**Training Time:** ~27 minutes (6 epochs with early stopping) on Colab T4 GPU