# ============================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# Tính RMSE (sai số bình phương trung bình có căn), MAE (sai số tuyệt đối trung bình), R² (hệ số xác định) trên tập test độc lập
# Chọn mô hình tốt nhất cho mỗi ngân hàng + horizon
# ============================================================

import pandas as pd
import numpy as np
import torch
import pickle, os, sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(__file__))
from models import MODEL_REGISTRY
from build_model import make_sequences, INPUT_LEN, PRED_LENS

# Hàm tải mô hình đã lưu
def load_model(ticker, model_name, pred_len):
    ModelClass = MODEL_REGISTRY[model_name]
    model      = ModelClass(INPUT_LEN, pred_len)
    path       = f"data/models/{ticker}_{model_name}_pred{pred_len}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Hàm đánh giá 1 mô hình trên tập test, trả về dict chỉ số RMSE, MAE, R²
def evaluate_one(ticker, model_name, pred_len, prices_scaled, scaler, split):
    X, y = make_sequences(prices_scaled, INPUT_LEN, pred_len)
    X_test, y_test = X[split:], y[split:]

    if len(X_test) == 0:
        return None

    model = load_model(ticker, model_name, pred_len)
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    y_true = inv(y_test[:, -1])
    y_pred = inv(preds[:, -1])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    return {"ticker": ticker, "model": model_name,
            "pred_len": pred_len, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_all(data_path="data/processed/clean_data.csv"):
    print("=" * 50)
    print("ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 50)

    df      = pd.read_csv(data_path)
    results = []

    for ticker in df["ticker"].unique():
        sub    = df[df["ticker"] == ticker].sort_values("time")
        prices = sub["close"].values

        with open(f"data/models/{ticker}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        prices_scaled = scaler.transform(prices.reshape(-1, 1)).flatten()

        split = int(len(prices_scaled) * 0.8)

        for pred_len in PRED_LENS:
            for model_name in MODEL_REGISTRY:
                res = evaluate_one(ticker, model_name, pred_len,
                                   prices_scaled, scaler, split)
                if res:
                    results.append(res)

    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)

    # Lưu kết quả
    results_df.to_csv("data/model_evaluation.csv", index=False)

    # --- In tổng quan ---
    print("\n📌 Kết quả đánh giá (trung bình toàn bộ ngân hàng):")
    summary = (results_df
               .groupby(["model", "pred_len"])[["RMSE", "MAE", "R2"]]
               .mean()
               .round(4))
    print(summary.to_string())

    # --- Mô hình tốt nhất (RMSE thấp nhất) ---
    print("\n🏆 Mô hình tốt nhất theo từng horizon:")
    best = (results_df
            .sort_values("RMSE")
            .groupby("pred_len")
            .first()[["model", "RMSE", "MAE", "R2"]])
    print(best.to_string())

    print(f"\n💾 Đã lưu chi tiết: data/model_evaluation.csv")
    return results_df


if __name__ == "__main__":
    evaluate_all()