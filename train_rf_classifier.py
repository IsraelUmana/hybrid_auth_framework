"""
Random Forest Classifier for CMU Keystroke Dataset
Project: Privacy-Preserving Hybrid Authentication Framework
"""
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "datasets/processed/features_normalized.csv"
RESULTS_DIR = "results"
MODEL_DIR = "models/rf_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("[1] Loading normalized dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # We only use the 11 Hold-time features and 10 Down-Down features.
    # Literature shows UD (Up-Down) features add noise.
    feature_cols = [c for c in df.columns if c.startswith(("H.", "DD."))]
    print(f"    Selected {len(feature_cols)} features (Hold-time + Down-Down).")
    
    subjects = sorted(df['subject'].unique())
    
    print("[2] Training Random Forest Classifiers...")
    results = []
    
    for i, subject in enumerate(subjects):
        # 1. Create Binary Classification Problem
        # User target = 1, All other users = 0
        df_binary = df.copy()
        df_binary['target'] = np.where(df_binary['subject'] == subject, 1, 0)
        
        # 2. Extract Features and Target
        X = df_binary[feature_cols].values
        y = df_binary['target'].values
        
        # 3. Train/Test Split (80% train, 20% test)
        # Stratified split ensures the 1:50 ratio of Genuine to Impostor is maintained
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. Train Random Forest
        # n_estimators=100, max_depth=10 prevents extreme overfitting
        clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Save model for the ZKP phase
        with open(os.path.join(MODEL_DIR, f"rf_{subject}.pkl"), 'wb') as f:
            pickle.dump(clf, f)
            
        # 5. Evaluate
        y_pred = clf.predict(X_test)
        
        # Calculate FRR (Genuine classified as 0)
        genuine_mask = (y_test == 1)
        frr = np.mean(y_pred[genuine_mask] == 0)
        
        # Calculate FAR (Impostor classified as 1)
        impostor_mask = (y_test == 0)
        far = np.mean(y_pred[impostor_mask] == 1)
        
        results.append({
            'subject': subject,
            'FAR': far,
            'FRR': frr
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(subjects)} subjects...")

    # Calculate Global Averages
    results_df = pd.DataFrame(results)
    global_far = results_df['FAR'].mean()
    global_frr = results_df['FRR'].mean()
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean False Acceptance Rate (FAR): {global_far * 100:.4f}%")
    print(f"Mean False Rejection Rate (FRR):  {global_frr * 100:.4f}%")
    print("=" * 60)
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, "rf_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "global_metrics.json"), "w") as f:
        json.dump({"FAR": global_far, "FRR": global_frr, "features": len(feature_cols)}, f, indent=4)
        
    print(f"Models saved to {MODEL_DIR}. Ready for ZKP integration.")

if __name__ == "__main__":
    main()