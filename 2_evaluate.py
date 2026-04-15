"""
evaluate.py — Đánh giá LSTM / DLinear / NLinear trên 30 mã ngân hàng.

Cách dùng:
    python evaluate.py                        # đánh giá tất cả mã trong config.SYMBOLS
    python evaluate.py --symbols VCB MBB ACB  # chỉ đánh giá một số mã

Kết quả:
    logs/eval_<SYMBOL>.png   — biểu đồ so sánh từng mã
    logs/summary.csv         — bảng RMSE / MAE / R2 toàn bộ
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vnstock import Vnstock

import config
from models_def import LSTMModel, DLinearModel, NLinearModel
from data_utils import preprocess_data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str):
    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df    = stock.quote.history(
            start=config.START_DATE, end=config.END_DATE, interval="1D"
        )
        if df is None or len(df) < 200:
            print(f"[{symbol}] ⚠️  Không đủ dữ liệu, bỏ qua.")
            return None
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def get_test_predictions(df, symbol: str, model_name: str, window: int):
    """
    Chỉ evaluate trên 20% cuối (test set) — tránh data leakage.
    Trả về (preds_price, actuals_price) hoặc (None, None).
    """
    path = f"models/{model_name.lower()}_{symbol}_{window}d.pth"
    if not os.path.exists(path):
        return None, None

    # Dùng đúng preprocessing như lúc train
    data      = preprocess_data(df)           # log return, shape [N, 1]
    split_idx = int(0.8 * len(data))
    test_data = data[split_idx:]              # Fix: chỉ lấy 20% cuối

    if len(test_data) <= window:
        print(f"  [{symbol}] ⚠️  Test set quá ngắn cho window={window}d")
        return None, None

    # Tạo sliding windows trên test set
    inputs = [test_data[i : i + window] for i in range(len(test_data) - window)]
    X      = torch.FloatTensor(np.array(inputs))

    model = build_model(model_name, window)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        log_return_preds = model(X).numpy().flatten()

    # Inverse transform: log return → giá thực
    # Lấy giá close tương ứng với phần test
    close      = df["close"].values
    # Số điểm bị bỏ do log return (shift 1) = 1, nên test bắt đầu từ split_idx+1
    test_close = close[split_idx + 1:]       # giá thực tế trong test period
    base       = test_close[window : window + len(log_return_preds)]
    preds      = base * np.exp(log_return_preds)
    actuals    = test_close[window + 1 : window + 1 + len(log_return_preds)]

    min_len = min(len(preds), len(actuals))
    if min_len == 0:
        return None, None

    return preds[:min_len], actuals[:min_len]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate 1 mã
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_symbol(symbol: str) -> list[dict]:
    """
    Evaluate toàn bộ (3 model × 3 window) cho 1 mã.
    Lưu biểu đồ PNG. Trả về list dict kết quả để gộp vào summary.
    """
    print(f"\n[{symbol}] 📊 Đang đánh giá...")

    df = fetch_data(symbol)
    if df is None:
        return []

    summary_rows = []
    model_names  = ["LSTM", "DLinear", "NLinear"]
    colors       = {"LSTM": "red", "DLinear": "green", "NLinear": "blue"}

    n_windows = len(config.INPUT_WINDOWS)
    fig, axes = plt.subplots(1, n_windows, figsize=(7 * n_windows, 5))
    if n_windows == 1:
        axes = [axes]   # Fix: đảm bảo axes luôn là list

    fig.suptitle(f"{symbol} — Predictions vs Actual (test set 20%)",
                 fontsize=14, fontweight="bold")

    for col_idx, window in enumerate(config.INPUT_WINDOWS):
        ax = axes[col_idx]
        ref_plotted = False

        for name in model_names:
            preds, actuals = get_test_predictions(df, symbol, name, window)

            if preds is None:
                print(f"  [{symbol}] ⚠️  Thiếu weight: {name} window={window}d")
                continue

            rmse, mae, r2 = calculate_metrics(actuals, preds)
            summary_rows.append({
                "Symbol": symbol,
                "Window": f"{window}d",
                "Model":  name,
                "RMSE":   round(rmse, 4),
                "MAE":    round(mae,  4),
                "R2":     round(r2,   4),
            })
            print(f"  [{symbol}] {name:8s} w={window}d | "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

            # Vẽ actual một lần
            if not ref_plotted:
                ax.plot(actuals, color="black", linewidth=1.8,
                        label="Thực tế", alpha=0.9)
                ref_plotted = True

            ax.plot(preds, color=colors[name], linewidth=1.5,
                    linestyle="--", alpha=0.8,
                    label=f"{name} (RMSE {rmse:.4f})")

        ax.set_title(f"Window {window}d", fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Phiên giao dịch (test set)")
        ax.set_ylabel("Giá (VND)")

    plt.tight_layout()
    out_path = f"logs/eval_{symbol}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [{symbol}] 💾 Đã lưu biểu đồ → {out_path}")

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate stock forecast models")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Danh sách mã cần evaluate (mặc định: tất cả trong config.SYMBOLS)"
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    symbols = list(dict.fromkeys(args.symbols or config.SYMBOLS))
    print(f"=== Evaluate {len(symbols)} mã | "
          f"{len(config.INPUT_WINDOWS)} windows × 3 models ===\n")

    all_rows = []
    for symbol in symbols:
        rows = evaluate_symbol(symbol)
        all_rows.extend(rows)

    if not all_rows:
        print("\n⚠️  Không có kết quả nào — hãy chạy train.py trước.")
        return

    # ── Lưu summary CSV ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    csv_path   = "logs/summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # ── In bảng tổng hợp ─────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PERFORMANCE SUMMARY")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    # ── Top 5 model tốt nhất theo RMSE ───────────────────────────────────────
    print("\n" + "=" * 75)
    print("TOP 5 — RMSE thấp nhất")
    print("=" * 75)
    print(summary_df.nsmallest(5, "RMSE")[
        ["Symbol", "Window", "Model", "RMSE", "MAE", "R2"]
    ].to_string(index=False))

    print(f"\n✅ Summary đã lưu → {csv_path}")
    print(f"✅ Biểu đồ từng mã  → logs/eval_<SYMBOL>.png")
    print("=== HOÀN THÀNH ===")


if __name__ == "__main__":
    main()