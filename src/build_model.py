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
INPUT_LEN  = 60          # nhìn lại 60 phiên giao dịch
PRED_LENS  = [1, 5, 21]  # dự đoán 1 ngày / 1 tuần / 1 tháng
EPOCHS     = 50 # epochs là số lần toàn bộ tập dữ liệu được đưa qua mô hình trong quá trình huấn luyện.
BATCH_SIZE = 32 # batch size là số lượng mẫu được đưa qua mô hình trước khi cập nhật trọng số một lần,
LR         = 1e-3 # learning rate là tốc độ học, xác định mức độ điều chỉnh trọng số của mô hình dựa trên lỗi của mỗi lần cập nhật.
os.makedirs("data/models", exist_ok=True)

# Biến dữ liệu chuỗi thành cặp (X, y) dạng sliding window để huấn luyện mô hình.
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

      # Chia train / test = 80 / 20 trước
        split = int(len(prices) * 0.8)

        # Chỉ fit scaler trên train set
        scaler = MinMaxScaler()
        scaler.fit(prices[:split].reshape(-1, 1))

        # Transform toàn bộ dữ liệu bằng scaler đã fit trên train
        prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()

        # Lưu scaler
        with open(f"data/models/{ticker}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        for pred_len in PRED_LENS:
            X, y = make_sequences(prices_scaled, INPUT_LEN, pred_len)
            X_train, y_train = X[:split], y[:split]

            for model_name, ModelClass in MODEL_REGISTRY.items():
                model = ModelClass(INPUT_LEN, pred_len)
                history = train_model(model, X_train, y_train)

                # Lưu mô hình
                path = f"data/models/{ticker}_{model_name}_pred{pred_len}.pt"
                torch.save(model.state_dict(), path)

            print(f"  ✅ {ticker} | pred_len={pred_len} → đã lưu 3 mô hình")

    print("\n✅ Hoàn thành huấn luyện tất cả mô hình!")


if __name__ == "__main__":
    train_all()