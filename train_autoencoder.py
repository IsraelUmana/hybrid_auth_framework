import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "datasets/processed/features_normalized.csv"
RESULTS_DIR = "results"
MODEL_DIR = "models/autoencoder"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1. ARCHITECTURE (Tighter bottleneck)
# ============================================================
class BehavioralAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(BehavioralAutoencoder, self).__init__()
        # Compress 31 features to a very tight 4-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ============================================================
# 2. TRAINING
# ============================================================
def train_autoencoder(X_train, epochs=200):
    model = BehavioralAutoencoder(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()
    return model

def get_anomaly_scores(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        reconstructions = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
    return mse.numpy()

# ============================================================
# 3. TRUE IMPOSTOR EVALUATION
# ============================================================
def main():
    print("[1] Loading normalized dataset...")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith(("H.", "DD.", "UD."))]
    subjects = sorted(df['subject'].unique())
    
    print("[2] Training Autoencoders with Strict Impostor Testing...")
    results = []
    
    for i, subject in enumerate(subjects):
        # 1. Get user data
        X_user = df[df['subject'] == subject][feature_cols].values
        X_train, X_test = train_test_split(X_user, test_size=0.2, random_state=42)
        
        # 2. Train Model
        model = train_autoencoder(X_train, epochs=200)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"ae_{subject}.pt"))
        
        # 3. Calculate Genuine Scores (to set threshold)
        genuine_scores = get_anomaly_scores(model, X_test)
        
        # Set threshold to allow 5% FRR (95th percentile)
        threshold = np.percentile(genuine_scores, 95)
        frr = np.mean(genuine_scores > threshold)
        
        # 4. Strict FAR Calculation
        # Test against ALL data from one randomly selected other subject
        impostor_subject = np.random.choice([s for s in subjects if s != subject])
        X_impostor = df[df['subject'] == impostor_subject][feature_cols].values
        
        impostor_scores = get_anomaly_scores(model, X_impostor)
        far = np.mean(impostor_scores <= threshold)
        
        results.append({
            'subject': subject,
            'threshold': float(threshold),
            'FAR': far,
            'FRR': frr
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(subjects)} subjects...")
        
    # Calculate averages
    results_df = pd.DataFrame(results)
    global_far = results_df['FAR'].mean()
    global_frr = results_df['FRR'].mean()
    
    print("\n" + "=" * 60)
    print("AUTOENCODER EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean False Acceptance Rate (FAR): {global_far * 100:.2f}%")
    print(f"Mean False Rejection Rate (FRR):  {global_frr * 100:.2f}%")
    print("=" * 60)
    
    results_df.to_csv(os.path.join(RESULTS_DIR, "autoencoder_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "global_metrics.json"), "w") as f:
        json.dump({"FAR": global_far, "FRR": global_frr}, f, indent=4)
        
    print(f"Models and thresholds saved. Ready for ZKP integration.")

if __name__ == "__main__":
    main()