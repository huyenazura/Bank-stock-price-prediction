"""
train.py — Train LSTM / DLinear / NLinear cho 30 mã ngân hàng song song.

Cách dùng:
    python train.py

Mỗi process con xử lý 1 mã cổ phiếu độc lập.
Kết quả weight lưu vào:  models/<model>_<SYMBOL>_<window>d.pth
Loss curves lưu vào:     logs/loss_<SYMBOL>_<window>d.png
"""

import os
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — bắt buộc khi dùng multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vnstock import Vnstock

import config
from models_def import LSTMModel, DLinearModel, NLinearModel
from data_utils import StockDataset, preprocess_data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str):
    """Tải dữ liệu từ vnstock. Trả về DataFrame hoặc None nếu lỗi."""
    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df    = stock.quote.history(start=config.START_DATE, end=config.END_DATE, interval="1D")
        if df is None or len(df) < 200:
            print(f"[{symbol}] ⚠️  Không đủ dữ liệu (< 200 phiên), bỏ qua.")
            return None
        return df
    except Exception as e:
        print(f"[{symbol}] ❌ Lỗi tải dữ liệu: {e}")
        return None


def build_model(name: str, window: int):
    if name == "LSTM":
        return LSTMModel(1, config.HIDDEN_SIZE, 1)
    elif name == "DLinear":
        return DLinearModel(window, pred_len=1)
    else:
        return NLinearModel(window, pred_len=1)


def train_one(model, train_loader, val_loader, device):
    """Train 1 model, trả về (model, train_losses, val_losses)."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.to(device)
    train_losses, val_losses = [], []

    for _ in range(config.EPOCHS):
        # Train
        model.train()
        epoch_train = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx).view(-1), by.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train += loss.item()
        train_losses.append(epoch_train / max(1, len(train_loader)))

        # Validate
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                epoch_val += criterion(model(bx).view(-1), by.view(-1)).item()
        val_losses.append(epoch_val / max(1, len(val_loader)))
        scheduler.step()

    return model, train_losses, val_losses


def save_loss_plot(symbol, window, loss_dict):
    """Lưu biểu đồ loss cho 3 model trên cùng 1 hình."""
    plt.figure(figsize=(10, 5))
    for name, (tr, va) in loss_dict.items():
        plt.plot(tr, label=f"{name} Train")
        plt.plot(va, linestyle="--", label=f"{name} Val")
    plt.title(f"{symbol} — Window {window}d Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"logs/loss_{symbol}_{window}d.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Worker chạy trong process con
# ─────────────────────────────────────────────────────────────────────────────

def train_symbol(symbol: str):
    """
    Hàm được gọi trong process con.
    Train toàn bộ (3 model × 3 window) cho 1 mã cổ phiếu.
    """
    # Mỗi process dùng CPU để tránh tranh chấp GPU
    device = "cpu"

    print(f"\n[{symbol}] 🚀 Bắt đầu...")
    t0 = time.time()

    df = fetch_data(symbol)
    if df is None:
        return

    data = preprocess_data(df)   # [N, 1]

    for window in config.INPUT_WINDOWS:
        if len(data) < window + 50:
            print(f"[{symbol}] ⚠️  Bỏ qua window={window} (dữ liệu quá ngắn)")
            continue

        # Chronological split 80/20
        split   = int(0.8 * len(data))
        train_d = StockDataset(data[:split], window)
        val_d   = StockDataset(data[split:], window)

        if len(train_d) == 0 or len(val_d) == 0:
            print(f"[{symbol}] ⚠️  Dataset rỗng tại window={window}")
            continue

        train_loader = DataLoader(train_d, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_d,   batch_size=config.BATCH_SIZE, shuffle=False)

        loss_dict = {}
        for name in ["LSTM", "DLinear", "NLinear"]:
            model = build_model(name, window)
            trained, tr_loss, va_loss = train_one(model, train_loader, val_loader, device)

            # Lưu weight
            path = f"models/{name.lower()}_{symbol}_{window}d.pth"
            torch.save(trained.state_dict(), path)
            loss_dict[name] = (tr_loss, va_loss)
            print(f"  [{symbol}] ✅ {name} window={window}d  val_loss={va_loss[-1]:.5f}")

        save_loss_plot(symbol, window, loss_dict)

    print(f"[{symbol}] 🏁 Xong trong {(time.time()-t0)/60:.1f} phút")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for folder in ["models", "logs"]:
        os.makedirs(folder, exist_ok=True)

    symbols = list(dict.fromkeys(config.SYMBOLS))   # bỏ trùng nếu có
    print(f"=== Train {len(symbols)} mã ngân hàng | {config.NUM_WORKERS} workers ===")
    print(f"    Mỗi mã: {len(config.INPUT_WINDOWS)} windows × 3 models × {config.EPOCHS} epochs\n")

    # Pool giới hạn NUM_WORKERS process chạy cùng lúc
    with mp.Pool(processes=config.NUM_WORKERS) as pool:
        pool.map(train_symbol, symbols)

    print("\n=== ✅ HOÀN THÀNH TOÀN BỘ ===")


if __name__ == "__main__":
    # Bắt buộc trên Windows để multiprocessing hoạt động đúng
    mp.set_start_method("spawn", force=True)
    main()