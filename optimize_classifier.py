import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold
import itertools
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "datasets/processed/features_normalized.csv"

def evaluate_parameters(df, feature_cols, nu, gamma):
    subjects = sorted(df['subject'].unique())
    far_list = []
    frr_list = []
    
    # Use only 3-fold to speed up the grid search
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for subject in subjects:
        X = df[df['subject'] == subject][feature_cols].values
        y_dummy = np.ones(len(X))
        
        for train_idx, test_idx in skf.split(X, y_dummy):
            X_train, X_test = X[train_idx], X[test_idx]
            
            clf = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            clf.fit(X_train)
            
            # FRR
            y_pred = clf.predict(X_test)
            frr_list.append(np.mean(y_pred == -1))
            
            # FAR (Sample 20 impostors)
            X_imp = df[df['subject'] != subject].sample(n=20, random_state=42)[feature_cols].values
            y_imp_pred = clf.predict(X_imp)
            far_list.append(np.mean(y_imp_pred == 1))
            
    return np.mean(far_list), np.mean(frr_list)

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    hold_cols = [c for c in df.columns if c.startswith("H.")]
    all_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    
    # We will test two feature sets and three gamma values
    feature_sets = {
        "All 31 Features": all_cols,
        "Hold-Time Only (11)": hold_cols
    }
    
    gammas = [0.01, 0.05, 0.1]
    nus = [0.05, 0.1]
    
    print(f"{'Features':<20} | {'Gamma':<6} | {'Nu':<4} | {'FAR':<8} | {'FRR':<8}")
    print("-" * 60)
    
    for name, cols in feature_sets.items():
        for gamma in gammas:
            for nu in nus:
                far, frr = evaluate_parameters(df, cols, nu, gamma)
                print(f"{name:<20} | {gamma:<6} | {nu:<4} | {far*100:>6.2f}% | {frr*100:>6.2f}%")