# ============================================================
# BƯỚC 4: XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH
# Huấn luyện Linear / DLinear / NLinear cho từng ngân hàng
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle, os, sys

sys.path.append(os.path.dirname(__file__))
from models import MODEL_REGISTRY

# ---------- Cấu hình ----------
INPUT_LEN  = 60
PRED_LENS  = [1, 5, 21]
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
os.makedirs("data/models", exist_ok=True)


def make_sequences(series: np.ndarray, input_len: int, pred_len: int):
    """Tạo cặp (X, y) dạng sliding window."""
    X, y = [], []
    for i in range(len(series) - input_len - pred_len + 1):
        X.append(series[i : i + input_len])
        y.append(series[i + input_len : i + input_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model, X_train, y_train, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Huấn luyện một mô hình, trả về loss theo epoch."""
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(loader))

    return history


def train_all(data_path="data/processed/clean_data.csv"):
    print("=" * 50)
    print("HUẤN LUYỆN MÔ HÌNH")
    print("=" * 50)

    df      = pd.read_csv(data_path)
    tickers = df["ticker"].unique()

    for ticker in tickers:
        sub    = df[df["ticker"] == ticker].sort_values("time")
        prices = sub["close"].values

        price_split = int(len(prices) * 0.8)
        prices_train = prices[:price_split]
        prices_test  = prices[price_split:]

        scaler = MinMaxScaler()
        scaler.fit(prices_train.reshape(-1, 1))

        prices_train_scaled = scaler.transform(prices_train.reshape(-1, 1)).flatten()
        prices_test_scaled  = scaler.transform(prices_test.reshape(-1, 1)).flatten()

        # Lưu scaler
        with open(f"data/models/{ticker}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        for pred_len in PRED_LENS:
            train_series = prices_train_scaled
            test_series  = np.concatenate([prices_train_scaled[-INPUT_LEN:], prices_test_scaled])
            X_train, y_train = make_sequences(train_series, INPUT_LEN, pred_len)
            X_test,  y_test  = make_sequences(test_series,  INPUT_LEN, pred_len)

            # Lưu test sequences để evaluate_model dùng lại (không cần tính lại)
            np.save(f"data/models/{ticker}_Xtest_pred{pred_len}.npy", X_test)
            np.save(f"data/models/{ticker}_ytest_pred{pred_len}.npy", y_test)

            for model_name, ModelClass in MODEL_REGISTRY.items():
                model   = ModelClass(INPUT_LEN, pred_len)
                history = train_model(model, X_train, y_train)

                path = f"data/models/{ticker}_{model_name}_pred{pred_len}.pt"
                torch.save(model.state_dict(), path)

            print(f"  ✅ {ticker} | pred_len={pred_len} "
                  f"| train sequences={len(X_train)} | test sequences={len(X_test)}")

    print("\n✅ Hoàn thành huấn luyện tất cả mô hình!")


if __name__ == "__main__":
    train_all()