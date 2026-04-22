"""
evaluate.py — Đánh giá hệ thống 2 pha (pretrain VCB + finetune 29 mã).

Cách dùng:
    python evaluate.py                        # đánh giá tất cả
    python evaluate.py --symbols VCB MBB ACB  # chỉ đánh giá một số mã

Kết quả:
    logs/eval_<SYMBOL>.png    — biểu đồ dự báo từng mã
    logs/summary.csv          — RMSE / MAE / R² toàn bộ
    logs/best_models.csv      — model tốt nhất theo window (từ pretrain VCB)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from models_def import build_model
from data_utils import fetch_data, preprocess_data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _load_selection() -> dict:
    """Load kết quả chọn model tốt nhất từ pha 1."""
    path = "logs/pretrain_selection.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _get_weight_path(symbol: str, window: int, selection: dict) -> tuple:
    """
    Trả về (weight_path, model_name, weight_type).
    Ưu tiên: finetune → pretrain → None.
    """
    info       = selection.get(window, {})
    best_model = info.get("best_model", "NLinear")

    # Thử finetune trước
    finetune_path = f"models/finetune_{best_model.lower()}_{symbol}_{window}d.pth"
    if os.path.exists(finetune_path):
        return finetune_path, best_model, "finetune"

    # Fallback sang pretrain (dùng cho mã đại diện VCB)
    pretrain_path = f"models/pretrain_{best_model.lower()}_{window}d.pth"
    if os.path.exists(pretrain_path):
        return pretrain_path, best_model, "pretrain"

    return None, best_model, "missing"


def get_predictions(df: pd.DataFrame, symbol: str, window: int,
                    selection: dict) -> tuple:
    """
    Dự báo trên 20% test set dùng model tốt nhất (finetune hoặc pretrain).
    Trả về (preds_logret, actuals_logret, model_name, weight_type).
    """
    weight_path, model_name, wtype = _get_weight_path(symbol, window, selection)
    if weight_path is None:
        return None, None, model_name, "missing"

    data      = preprocess_data(df, verbose=False)       # [N, F]
    n_features = data.shape[1]
    split_idx = int(0.8 * len(data))
    test_data = data[split_idx:]

    if len(test_data) <= window:
        return None, None, model_name, wtype

    inputs = np.array([test_data[i : i + window]
                       for i in range(len(test_data) - window)])
    X = torch.FloatTensor(inputs)

    model = build_model(model_name, window, config.HIDDEN_SIZE,
                        n_features=n_features)
    try:
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
    except RuntimeError as e:
        print(f"    ⚠️  Load weight thất bại [{symbol} w={window}d]: {e}")
        return None, None, model_name, "error"

    model.eval()
    with torch.no_grad():
        log_return_preds = model(X).numpy().flatten()

    # Cột 0 = log_return (target)
    actuals = test_data[window:, 0]
    min_len = min(len(log_return_preds), len(actuals))
    if min_len == 0:
        return None, None, model_name, wtype

    return log_return_preds[:min_len], actuals[:min_len], model_name, wtype


# ─────────────────────────────────────────────────────────────────────────────
# Đánh giá 1 mã
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_symbol(symbol: str, selection: dict) -> list:
    """
    Đánh giá toàn bộ 3 window cho 1 mã dùng model tốt nhất từ selection.
    Lưu biểu đồ PNG. Trả về list dict kết quả.
    """
    print(f"\n[{symbol}] 📊 Đánh giá...")

    df = fetch_data(symbol, config.START_DATE, config.END_DATE, verbose=False)
    if df is None:
        return []

    summary_rows = []
    n_windows    = len(config.INPUT_WINDOWS)
    fig, axes    = plt.subplots(1, n_windows, figsize=(7 * n_windows, 5), squeeze=False)

    for col_idx, window in enumerate(config.INPUT_WINDOWS):
        ax = axes[0][col_idx]

        preds, actuals, model_name, wtype = get_predictions(
            df, symbol, window, selection
        )

        horizon = config.PRED_HORIZONS.get(window, f"{window}d")

        if preds is None:
            print(f"  [{symbol}] ⚠️  Không có weight — w={window}d model={model_name}")
            ax.text(0.5, 0.5, f"Thiếu weight\n{model_name} w={window}d",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=9)
            ax.set_title(f"Dự báo {horizon}")
            continue

        rmse, mae, r2 = calc_metrics(actuals, preds)
        summary_rows.append({
            "symbol"    : symbol,
            "window"    : window,
            "model"     : model_name,
            "weight"    : wtype,
            "rmse"      : round(rmse, 6),
            "mae"       : round(mae,  6),
            "r2"        : round(r2,   4),
        })
        print(f"  [{symbol}] w={window:2d}d | {model_name:8s} [{wtype:8s}] | "
              f"RMSE={rmse:.6f}  MAE={mae:.6f}  R²={r2:.4f}")

        # Vẽ
        x = np.arange(len(actuals))
        ax.plot(x, actuals, color="black",   linewidth=1.8, label="Thực tế",    zorder=5)
        ax.plot(x, preds,   color="#f0883e", linewidth=1.5, linestyle="--",
                alpha=0.9, label=f"{model_name} [{wtype}]\nRMSE={rmse:.4f}", zorder=4)
        ax.fill_between(x, actuals, preds, alpha=0.08, color="#f0883e")
        ax.set_title(f"Dự báo {horizon}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Phiên giao dịch (test set 20%)")
        ax.set_ylabel("Log Return")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{symbol} — Model tốt nhất (pretrain VCB → finetune) | Test set 20%",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"logs/eval_{symbol}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [{symbol}] 💾 → {out}")

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    # Load kết quả chọn model từ pha 1
    selection = _load_selection()
    if not selection:
        print("⚠️  Không tìm thấy logs/pretrain_selection.json.")
        print("   Hãy chạy: python train.py --phase 1 trước.")
        return

    print("\n" + "═"*65)
    print("  ĐÁNH GIÁ MÔ HÌNH — Model tốt nhất từ Pretrain VCB")
    print("═"*65)
    print("  Selection từ Pretrain:")
    for w, info in selection.items():
        print(f"    Window {w:2d}d → {info['best_model']:8s} "
              f"(val_loss={info['val_loss']:.6f})")
    print(f"\n  Metrics: RMSE, MAE, R²  |  Test set: 20% cuối")
    print("═"*65)

    symbols  = list(dict.fromkeys(args.symbols or config.SYMBOLS))
    all_rows = []

    for symbol in symbols:
        rows = evaluate_symbol(symbol, selection)
        all_rows.extend(rows)

    if not all_rows:
        print("\n⚠️  Không có kết quả. Chạy train.py trước.")
        return

    summary = pd.DataFrame(all_rows)
    summary.to_csv("logs/summary.csv", index=False)

    # ── In bảng tổng hợp ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  PERFORMANCE SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    # ── Thống kê theo model & weight type ────────────────────────────────────
    print("\n" + "="*80)
    print("  THỐNG KÊ TRUNG BÌNH THEO WINDOW")
    print("="*80)
    agg = (summary.groupby("window")[["rmse","mae","r2"]]
                  .agg(["mean","std"]).round(6))
    print(agg.to_string())

    # ── Top 10 RMSE thấp nhất ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  TOP 10 — RMSE thấp nhất")
    print("="*80)
    print(summary.nsmallest(10, "rmse")
                 [["symbol","window","model","weight","rmse","mae","r2"]]
                 .to_string(index=False))

    # ── Lưu best_models.csv (model tốt nhất × window từ pretrain) ────────────
    best_rows = []
    for w, info in selection.items():
        best_rows.append({
            "window"    : w,
            "best_model": info["best_model"],
            "val_loss"  : info["val_loss"],
            "all_results": str(info.get("all_results", {})),
        })
    pd.DataFrame(best_rows).to_csv("logs/best_models.csv", index=False)

    print(f"\n✅ summary.csv      → logs/summary.csv")
    print(f"✅ best_models.csv  → logs/best_models.csv")
    print(f"✅ Biểu đồ từng mã  → logs/eval_<SYMBOL>.png")
    print("\n" + "═"*65)
    print("  HOÀN THÀNH ĐÁNH GIÁ")
    print("═"*65)


if __name__ == "__main__":
    main()