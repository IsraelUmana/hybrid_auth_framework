"""
One-Class SVM Training and Evaluation
Project: Privacy-Preserving Hybrid Authentication Framework
Phase: 2, Activity 2.2 & 2.3

Input:  datasets/processed/features_normalized.csv
Output: models/ocsvm_subject_{id}.pkl
        results/classifier_metrics.csv
        results/global_metrics.json
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import warnings
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "datasets/processed/features_normalized.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Hyperparameters for One-Class SVM
# nu: Upper bound on the fraction of training errors (outliers)
# gamma: Kernel coefficient for 'rbf'
NU = 0.05          
GAMMA = 'scale'   
N_SPLITS = 5      

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    print("[1] Loading normalized dataset...")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    print(f"    Loaded {len(df)} samples, {df['subject'].nunique()} subjects, {len(feature_cols)} features.")
    return df, feature_cols

# ============================================================
# STEP 2: TRAIN AND EVALUATE PER SUBJECT
# ============================================================
def train_and_evaluate(df, feature_cols):
    print("[2] Training One-Class SVM per subject...")
    
    results = []
    subjects = sorted(df['subject'].unique())
    
    for subject in subjects:
        # Isolate genuine data for this subject
        subject_data = df[df['subject'] == subject]
        X = subject_data[feature_cols].values
        
        # 5-fold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        y_dummy = np.ones(len(X))  # Dummy labels for stratification
        
        far_list = []
        frr_list = []
        
        for train_idx, test_idx in skf.split(X, y_dummy):
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Train One-Class SVM on genuine data only
            clf = OneClassSVM(kernel='rbf', nu=NU, gamma=GAMMA)
            clf.fit(X_train)
            
            # --- FRR Calculation ---
            # Predict on test fold (genuine data)
            # In sklearn, -1 is an outlier (rejected), 1 is inlier (accepted)
            y_pred = clf.predict(X_test)
            frr = np.mean(y_pred == -1)  # Genuine rejected as anomaly
            frr_list.append(frr)
            
            # --- FAR Calculation ---
            # Simulate impostors by sampling 20 random rows from OTHER subjects
            impostor_data = df[df['subject'] != subject].sample(n=20, random_state=42)
            X_impostor = impostor_data[feature_cols].values
            
            y_imp_pred = clf.predict(X_impostor)
            far = np.mean(y_imp_pred == 1)  # Impostors accepted as genuine
            far_list.append(far)
            
        # Average metrics over the 5 folds
        mean_far = np.mean(far_list)
        mean_frr = np.mean(frr_list)
        
        # Train FINAL model on ALL genuine data for this subject (for deployment)
        final_clf = OneClassSVM(kernel='rbf', nu=NU, gamma=GAMMA)
        final_clf.fit(X)
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f"ocsvm_subject_{subject}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(final_clf, f)
            
        results.append({
            'subject': subject,
            'samples': len(X),
            'FAR': mean_far,
            'FRR': mean_frr
        })
        
    return pd.DataFrame(results)

# ============================================================
# STEP 3: SUMMARY AND SAVE
# ============================================================
def summarize_and_save(results_df):
    print("[3] Calculating aggregate metrics...")
    
    global_far = results_df['FAR'].mean()
    global_frr = results_df['FRR'].mean()
    
    # Save detailed results per subject
    out_csv = os.path.join(RESULTS_DIR, "classifier_metrics.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"    Detailed metrics saved to {out_csv}")
    
    # Save global summary
    summary = {
        "total_subjects": len(results_df),
        "global_far": round(global_far, 4),
        "global_frr": round(global_frr, 4),
        "nu": NU,
        "gamma": GAMMA,
        "n_splits": N_SPLITS
    }
    out_json = os.path.join(RESULTS_DIR, "global_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"    Global summary saved to {out_json}")
    
    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean False Acceptance Rate (FAR): {global_far * 100:.2f}%")
    print(f"Mean False Rejection Rate (FRR):  {global_frr * 100:.2f}%")
    print("=" * 60)
    print("[DONE] Models saved to 'models/' directory. Ready for ZKP integration.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, feature_cols = load_data()
    results_df = train_and_evaluate(df, feature_cols)
    summarize_and_save(results_df)