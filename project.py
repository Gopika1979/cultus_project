import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.datasets import airline
import random
import math

# ======================
# Reproducibility
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Dataset Preparation
# ======================
def load_time_series():
    data = airline.load_pandas().data
    series = data["passengers"].values.astype("float32")
    return series

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# Model Definition
# ======================
class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("Unsupported RNN type")

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# ======================
# Training & Evaluation
# ======================
def train_model(model, loader, optimizer, criterion, epochs):
    model.train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

def evaluate_model(model, loader):
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

# ======================
# NAS Search Space
# ======================
SEARCH_SPACE = {
    "rnn_type": ["LSTM", "GRU"],
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.2, 0.4],
    "learning_rate": [1e-2, 1e-3, 5e-4],
    "batch_size": [16, 32, 64]
}

def sample_config():
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

# ======================
# NAS Engine (Random Search)
# ======================
def run_nas(train_data, val_data, input_size, trials=15):
    best_score = float("inf")
    best_config = None

    for trial in range(trials):
        config = sample_config()
        print(f"\n🔍 Trial {trial + 1} | Config: {config}")

        model = RNNModel(
            config["rnn_type"],
            input_size,
            config["hidden_size"],
            config["num_layers"],
            config["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config["batch_size"])

        train_model(model, train_loader, optimizer, criterion, epochs=20)
        rmse, mae = evaluate_model(model, val_loader)

        print(f"📊 RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if rmse < best_score:
            best_score = rmse
            best_config = config

    return best_config, best_score

# ======================
# Main Pipeline
# ======================
def main():
    series = load_time_series()

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    SEQ_LEN = 12
    X, y = create_sequences(series_scaled, SEQ_LEN)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    # Baseline
    print("\n📌 Training Baseline Model...")
    baseline_model = RNNModel("LSTM", 1, 32, 1, 0.0).to(DEVICE)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    train_model(baseline_model, train_loader, optimizer, criterion, epochs=20)
    baseline_rmse, baseline_mae = evaluate_model(baseline_model, val_loader)

    print(f"📉 Baseline RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")

    # NAS
    print("\n🚀 Starting Neural Architecture Search...")
    best_config, best_rmse = run_nas(train_dataset, val_dataset, 1)

    print("\n🏆 Best Configuration Found:")
    print(best_config)
    print(f"Best RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    main()
