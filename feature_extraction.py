"""
Feature Extraction and Normalization for CMU Keystroke Dynamics Dataset
Project: Privacy-Preserving Hybrid Authentication Framework
Phase: 2, Activity 2.1

Input:  DSL-StrongPasswordData.csv
Output: features_normalized.csv  (ready for One-Class SVM training)
        feature_stats.json       (transformation parameters for reproducibility)
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "datasets/DSL-StrongPasswordData.csv"
OUTPUT_DIR = "datasets/processed"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def load_data(path):
    """Load the CMU keystroke dataset and inspect its structure."""
    print("[1] Loading dataset...")
    df = pd.read_csv(path)
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {len(df.columns)}")
    print(f"    Unique subjects: {df['subject'].nunique()}")
    print(f"    Sessions per subject: {df['sessionIndex'].nunique()}")
    print(f"    Repetitions per session: {df['rep'].nunique()}")
    return df


# ============================================================
# STEP 2: IDENTIFY FEATURE COLUMNS
# ============================================================

def identify_features(df):
    """
    The CMU dataset has three types of timing features for each key:
      - H.key  : hold time (key press to release)
      - DD.key1.key2 : down-down interval (press key1 to press key2)
      - UD.key1.key2 : up-down interval (release key1 to press key2)

    For One-Class SVM we select the 9 hold-time features (one per character
    in the password '.tie5Roanl'), which are the most stable and discriminative.
    """
    print("[2] Identifying feature columns...")

    hold_cols = [c for c in df.columns if c.startswith("H.")]
    dd_cols   = [c for c in df.columns if c.startswith("DD.")]
    ud_cols   = [c for c in df.columns if c.startswith("UD.")]

    print(f"    Hold-time features: {len(hold_cols)}")
    print(f"    Down-down features: {len(dd_cols)}")
    print(f"    Up-down features:   {len(ud_cols)}")

    # Primary feature set: 9 hold times + 8 down-down + 8 up-down = 25 features
    # Start with all 25, then optionally prune by variance later.
    feature_cols = hold_cols + dd_cols + ud_cols
    print(f"    Selected feature set: {len(feature_cols)} columns")
    return feature_cols, hold_cols


# ============================================================
# STEP 3: CLEAN DATA
# ============================================================

def clean_data(df, feature_cols):
    """
    Clean the data by:
      1. Removing rows with NaN in feature columns (missing data)
      2. Removing rows where ANY HOLD-TIME feature is non-positive
         (a key press/release cannot take ≤ 0 seconds)

    Do NOT filter on DD/UD features: negative values are physically
    legitimate (key rollover: user presses key2 before releasing key1).
    """
    print("[3] Cleaning data...")
    before = len(df)

    # Step A: drop NaN (missing values) across ALL feature columns
    n_before_nan = len(df)
    df = df.dropna(subset=feature_cols)
    n_nan = n_before_nan - len(df)
    print(f"    Dropped {n_nan} rows with NaN values")

    # Step B: only HOLD-TIME features must be strictly positive.
    # Use the prefix "H." to filter hold-time-only columns.
    hold_cols = [c for c in feature_cols if c.startswith("H.")]
    mask = (df[hold_cols] > 0).all(axis=1)
    df = df[mask]

    after = len(df)
    print(f"    Removed {before - after} total rows")
    print(f"    Remaining rows: {after}")
    return df.reset_index(drop=True)


# ============================================================
# STEP 4: OUTLIER REMOVAL (per subject)
# ============================================================

def remove_outliers(df, feature_cols, z_threshold=3.5):
    print("[4] Removing outliers per subject (modified z-score)...")
    before = len(df)
    kept_groups = []
    for subject_id, group in df.groupby("subject", sort=False):
        if len(group) < 5:
            kept_groups.append(group)
            continue
        z = np.abs(stats.zscore(group[feature_cols], nan_policy="omit"))
        # Keep a row if ANY feature is within threshold (less aggressive
        # than requiring ALL features to be within threshold)
        keep_mask = (z < z_threshold).any(axis=1)
        cleaned = group[keep_mask].copy()
        cleaned["subject"] = subject_id
        kept_groups.append(cleaned)
    df = pd.concat(kept_groups, ignore_index=True)
    after = len(df)
    print(f"    Removed {before - after} outlier rows")
    print(f"    Remaining rows: {after}")
    return df.reset_index(drop=True)


# ============================================================
# STEP 5: NORMALIZATION (Multivariate Box-Cox)
# ============================================================

def normalize_features(df, feature_cols):
    """
    Apply a Yeo-Johnson power transformation (a Box-Cox variant that
    handles both positive and negative values) to normalize each feature.

    Why this matters:
      One-Class SVM with an RBF kernel assumes features are approximately
      Gaussian. Raw keystroke timings are right-skewed. The Yeo-Johnson
      transformation reduces skewness and stabilizes variance, which
      directly improves FAR/FRR.

    We fit the transformer on the FULL dataset (all subjects) so that
    the same transformation can be applied consistently to new users.
    """
    print("[5] Normalizing features (Yeo-Johnson power transform)...")

    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformed = transformer.fit_transform(df[feature_cols])

    df_norm = df.copy()
    df_norm[feature_cols] = transformed

    # Report skewness before and after for the first 5 features
    print("    Skewness comparison (first 5 features):")
    for col in feature_cols[:5]:
        before = stats.skew(df[col])
        after = stats.skew(df_norm[col])
        print(f"      {col:20s}  before={before:+.2f}  after={after:+.2f}")

    return df_norm, transformer


# ============================================================
# STEP 6: SAVE OUTPUTS
# ============================================================

def save_outputs(df_norm, transformer, feature_cols):
    """Save the normalized dataset and transformation parameters."""
    print("[6] Saving outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save normalized dataset
    out_csv = os.path.join(OUTPUT_DIR, "features_normalized.csv")
    df_norm.to_csv(out_csv, index=False)
    print(f"    Normalized dataset -> {out_csv}")

    # Save transformation parameters for reproducibility
    stats_dict = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "transformer_lambdas": transformer.lambdas_.tolist(),
        "transformer_mean": transformer._scaler.mean_.tolist(),
        "transformer_scale": transformer._scaler.scale_.tolist(),
        "random_seed": RANDOM_SEED,
    }
    out_json = os.path.join(OUTPUT_DIR, "feature_stats.json")
    with open(out_json, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"    Transformation parameters -> {out_json}")


# ============================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================

def summarize(df_norm, feature_cols):
    """Print a summary that you can copy into your paper's Methods section."""
    print("\n" + "=" * 60)
    print("SUMMARY FOR MANUSCRIPT")
    print("=" * 60)
    print(f"Total samples:        {len(df_norm)}")
    print(f"Subjects:             {df_norm['subject'].nunique()}")
    print(f"Samples per subject:  {len(df_norm) / df_norm['subject'].nunique():.1f}")
    print(f"Features extracted:   {len(feature_cols)}")
    print(f"Feature names:        {', '.join(feature_cols[:5])} ...")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data(DATASET_PATH)
    feature_cols, _ = identify_features(df)
    df = clean_data(df, feature_cols)
    df = remove_outliers(df, feature_cols)
    assert "subject" in df.columns, "subject column missing after outlier removal"
    df_norm, transformer = normalize_features(df, feature_cols)
    save_outputs(df_norm, transformer, feature_cols)
    summarize(df_norm, feature_cols)
    print("\n[DONE] Feature extraction complete. Ready for One-Class SVM training.")


if __name__ == "__main__":
    main()