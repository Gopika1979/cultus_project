import time
import random
import math
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.datasets import airline

# =====================
# Global Configuration
# =====================
SEED = 42
SEQ_LEN = 12
EPOCHS = 20
NAS_TRIALS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================
# Dataset Utilities
# =====================
def load_dataset():
    data = airline.load_pandas().data["passengers"].values.astype("float32")
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data, scaler

def create_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================
# RNN Model
# =====================
class RNNModel(nn.Module):
    def __init__(self, rnn_type, hidden_size, num_layers, dropout):
        super().__init__()

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("Invalid RNN type")

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# =====================
# Training & Evaluation
# =====================
def train(model, loader, optimizer, criterion):
    model.train()
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            preds.extend(model(X).cpu().numpy())
            targets.extend(y.numpy())

    rmse = math.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae

# =====================
# NAS Search Space
# =====================
SEARCH_SPACE = {
    "rnn_type": ["LSTM", "GRU"],
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.2, 0.4],
    "learning_rate": [0.01, 0.001, 0.0005],
    "batch_size": [16, 32, 64]
}

def sample_config():
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

# =====================
# NAS Engine
# =====================
def run_nas(train_ds, val_ds):
    best_rmse = float("inf")
    best_config = None
    total_time = 0

    for i in range(NAS_TRIALS):
        config = sample_config()
        start = time.time()

        model = RNNModel(
            config["rnn_type"],
            config["hidden_size"],
            config["num_layers"],
            config["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"]
        )

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
        criterion = nn.MSELoss()

        for _ in range(EPOCHS):
            train(model, train_loader, optimizer, criterion)

        rmse, mae = evaluate(model, val_loader)
        elapsed = time.time() - start
        total_time += elapsed

        print(f"Trial {i+1:02d} | RMSE={rmse:.4f} | MAE={mae:.4f} | Time={elapsed:.2f}s")

        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config

    print(f"\nNAS completed in {total_time:.2f}s")
    return best_config

# =====================
# Main Pipeline
# =====================
def main():
    series, _ = load_dataset()
    X, y = create_sequences(series, SEQ_LEN)

    # 70 / 15 / 15 split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    test_ds = TimeSeriesDataset(X_test, y_test)

    # =====================
    # Baseline Model
    # =====================
    print("\nTraining baseline model...")
    baseline = RNNModel("LSTM", 32, 1, 0.0).to(DEVICE)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    for _ in range(EPOCHS):
        train(baseline, train_loader, optimizer, criterion)

    baseline_rmse, baseline_mae = evaluate(baseline, test_loader)

    # =====================
    # NAS Search
    # =====================
    print("\nStarting NAS search...")
    best_config = run_nas(train_ds, val_ds)

    # =====================
    # Final Training (Train + Val)
    # =====================
    print("\nRetraining best NAS model on full training set...")
    full_X = np.concatenate([X_train, X_val])
    full_y = np.concatenate([y_train, y_val])
    full_ds = TimeSeriesDataset(full_X, full_y)

    final_model = RNNModel(
        best_config["rnn_type"],
        best_config["hidden_size"],
        best_config["num_layers"],
        best_config["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_config["learning_rate"]
    )

    full_loader = DataLoader(full_ds, batch_size=best_config["batch_size"], shuffle=True)

    for _ in range(EPOCHS):
        train(final_model, full_loader, optimizer, criterion)

    nas_rmse, nas_mae = evaluate(final_model, test_loader)

    # =====================
    # Final Output
    # =====================
    print("\n========== FINAL RESULTS ==========")
    print("Baseline Model:")
    print(f"RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}\n")

    print("NAS Optimized Model:")
    print(f"RMSE: {nas_rmse:.4f}, MAE: {nas_mae:.4f}\n")

    print("Winning NAS Configuration:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
