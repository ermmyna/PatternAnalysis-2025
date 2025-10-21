# make_splits.py
import pandas as pd
from pathlib import Path

MASTER_CSV = Path("data/isic2020_master.csv")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stratified_group_split(df, label_col, group_col,
                           train_size=0.70, val_size=0.15, test_size=0.15, seed=42):
    from sklearn.model_selection import GroupShuffleSplit

    def grouped_split(data, size, seed):
        gss = GroupShuffleSplit(n_splits=1, train_size=size, random_state=seed)
        groups = data[group_col].values
        train_idx, hold_idx = next(gss.split(data, groups=groups))
        return train_idx, hold_idx

    # train vs temp
    train_idx, temp_idx = grouped_split(df, train_size, seed)
    train_df = df.iloc[train_idx].copy()
    temp_df  = df.iloc[temp_idx].copy()

    # val vs test from temp
    rel = val_size / (val_size + test_size)
    val_idx, test_idx = grouped_split(temp_df, rel, seed + 1)
    val_df  = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # sanity print
    def stats(name, d):
        return f"{name}: n={len(d)}, melanoma%={100*d[label_col].mean():.2f}, unique_patients={d[group_col].nunique()}"
    print(stats("Train", train_df))
    print(stats("Val",   val_df))
    print(stats("Test",  test_df))

    # leakage check: no patient overlaps
    inter1 = set(train_df[group_col]) & set(val_df[group_col])
    inter2 = set(train_df[group_col]) & set(test_df[group_col])
    inter3 = set(val_df[group_col])   & set(test_df[group_col])
    assert not inter1 and not inter2 and not inter3, "Patient leakage across splits!"

    return train_df, val_df, test_df

def main():
    df = pd.read_csv(MASTER_CSV)
    needed = {"image_path","label","patient_id","lesion_id","image_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Master CSV missing columns: {missing}")

    df["label"] = df["label"].astype(int)

    # Group by patient to avoid leakage; switch to "lesion_id" if you prefer stricter grouping
    train_df, val_df, test_df = stratified_group_split(
        df, label_col="label", group_col="patient_id",
        train_size=0.70, val_size=0.15, test_size=0.15, seed=42
    )

    cols = ["image_path","label","patient_id","lesion_id","image_name"]
    train_df[cols].to_csv(OUT_DIR/"train.csv", index=False)
    val_df[cols].to_csv(OUT_DIR/"val.csv", index=False)
    test_df[cols].to_csv(OUT_DIR/"test.csv", index=False)
    print("Wrote:", OUT_DIR/"train.csv", OUT_DIR/"val.csv", OUT_DIR/"test.csv")

if __name__ == "__main__":
    main()
