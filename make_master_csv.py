# make_master_csv.py
import os
from pathlib import Path
import pandas as pd

# --------- EDIT THESE IF NEEDED ----------
IMAGES_DIR = Path("ISIC2020/train")  # folder containing ISIC_*.jpg
GROUNDTRUTH_CSV = Path("ISIC2020/ISIC_2020_Training_GroundTruth.csv")
METADATA_CSV   = Path("ISIC2020/ISIC_2020_Training_Metadata_v2.csv")
DUPLICATES_CSV = Path("ISIC2020/ISIC_2020_Training_Duplicates.csv")
OUT_CSV        = Path("data/isic2020_master.csv")
# -----------------------------------------

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# 1) Load CSVs
gt   = pd.read_csv(GROUNDTRUTH_CSV)
meta = pd.read_csv(METADATA_CSV)
dups = pd.read_csv(DUPLICATES_CSV)

print("GroundTruth columns:", list(gt.columns))
print("Metadata   columns:", list(meta.columns))
print("Duplicates columns:", list(dups.columns))

# 2) Resolve image-name column names
gt_name_col   = find_col(gt.columns,   ["image_name","image","file_name","name"])
meta_name_col = find_col(meta.columns, ["image_name","image","file_name","name"])

if gt_name_col is None or meta_name_col is None:
    raise KeyError("Could not find image name column in GroundTruth/Metadata CSVs.")

# 3) Build label=0/1 (melanoma) from GroundTruth
# Prefer 'target' if present, else derive from melanoma-like columns (e.g., 'MEL', 'melanoma', etc.)
if "target" in gt.columns:
    gt_small = gt[[gt_name_col, "target"]].rename(columns={gt_name_col: "image_name", "target": "label"})
else:
    # Look for a melanoma indicator column
    mela_candidates = [c for c in gt.columns if c.lower() in ("melanoma","mel","target_melanoma","is_melanoma")]
    if mela_candidates:
        mcol = mela_candidates[0]
        gt_small = gt[[gt_name_col, mcol]].rename(columns={gt_name_col: "image_name", mcol: "label"})
    else:
        # Fallback: if the sheet has one-hot diagnosis columns, treat any melanoma-like column as positive
        one_hot_mela = [c for c in gt.columns if "mel" in c.lower()]  # catches 'MEL' etc.
        if not one_hot_mela:
            raise KeyError("Could not find a melanoma/target column in GroundTruth CSV.")
        gt_small = gt[[gt_name_col] + one_hot_mela].rename(columns={gt_name_col: "image_name"})
        gt_small["label"] = (gt_small[one_hot_mela].sum(axis=1) > 0).astype(int)
        gt_small = gt_small[["image_name","label"]]

# 4) Keep only needed metadata columns
if not {"patient_id","lesion_id"}.issubset(set(meta.columns)):
    raise KeyError("Expected 'patient_id' and 'lesion_id' in Metadata v2 CSV.")
meta_small = meta[[meta_name_col, "patient_id", "lesion_id"]].rename(columns={meta_name_col: "image_name"})

# 5) Merge labels + metadata
df = gt_small.merge(meta_small, on="image_name", how="inner")

# 6) Remove duplicates: ISIC CSV lists pairs (image_name_1, image_name_2) — mark BOTH as duplicates
dup_cols = set(dups.columns)
if {"image_name_1","image_name_2"}.issubset(dup_cols):
    dup_names = pd.concat([
        dups["image_name_1"].astype(str),
        dups["image_name_2"].astype(str)
    ], ignore_index=True)
else:
    # fallback: a single column named something else
    single = find_col(dup_cols, ["image_name","image","file_name","name"])
    if single is None:
        raise KeyError("Could not resolve duplicate image name columns.")
    dup_names = dups[single].astype(str)

# strip '.jpg' if present so it matches no-extension 'image_name'
dup_names = dup_names.str.replace(".jpg", "", regex=False)
dup_set = set(dup_names.unique())

before = len(df)
df = df[~df["image_name"].astype(str).isin(dup_set)].copy()
print(f"Removed {before - len(df)} duplicate entries listed by ISIC.")

# 7) Create absolute image_path and drop missing files
df["image_path"] = df["image_name"].astype(str).apply(lambda s: str(IMAGES_DIR / f"{s}.jpg"))
missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
if len(missing) > 0:
    print("WARNING: some image files not found on disk (showing up to 5):")
    print(missing.head())
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()

# 8) Final tidy and save
df["label"] = df["label"].astype(int)
df = df[["image_path","label","patient_id","lesion_id","image_name"]]
df.to_csv(OUT_CSV, index=False)

print(f"Saved master CSV to: {OUT_CSV}  (rows: {len(df)})")
print(df.sample(min(5, len(df))))
