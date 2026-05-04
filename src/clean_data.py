# ============================================================
# BƯỚC 3: XỬ LÝ & LÀM SẠCH DỮ LIỆU
# Chuẩn hóa, tạo features, lưu dữ liệu sạch
# ============================================================

import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch và chuẩn hóa dữ liệu."""
    df = df.copy()
    
    # 1. Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    # 2. Chuẩn hóa thời gian
    df["time"] = pd.to_datetime(df["time"])

    # 3. Sắp xếp theo thời gian
    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)

    # 4. Xóa duplicate theo (ticker, time)
    df = df.drop_duplicates(subset=["ticker", "time"])

    # 5. xử lí missing values
    df = df.dropna()

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các features bổ sung cho mô hình."""

    # Tỷ suất sinh lời hàng ngày
    df["daily_return"] = df.groupby("ticker")["close"].pct_change().fillna(0)

    # Log giá => biến động giá sẽ trở nên ổn định hơn, giảm ảnh hưởng của outliers
    df["close_log"] = np.log(df["close"])

    # Biến động giá (High - Low) / Close
    df["price_range"] = (df["high"] - df["low"]) / df["close"]

    # Moving average 5 và 21 ngày để nắm bắt xu hướng ngắn hạn và dài hạn
    df["ma5"]  = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["ma21"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(21, min_periods=1).mean())

    return df


def process_pipeline(input_path="data/raw_data.csv",output_path="data/processed/clean_data.csv"):

    print("=" * 50)
    print("XỬ LÝ DỮ LIỆU")
    print("=" * 50)

    # Load
    df_raw = pd.read_csv(input_path)
    print(f"\n📥 Dữ liệu gốc: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")

    # Làm sạch
    df_clean = clean_data(df_raw)
    print(f"🧹 Sau khi làm sạch: {df_clean.shape[0]:,} dòng")

    # Tạo features
    df_final = create_features(df_clean)
    print(f"⚙️  Sau khi tạo features: {df_final.shape[1]} cột")

    # Lưu
    df_final.to_csv(output_path, index=False)
    print(f"\n💾 Đã lưu: {output_path}")

    # In mẫu kết quả
    print("\n📌 Dữ liệu sau xử lý (5 dòng đầu):")
    print(df_final.head().to_string())

    print("\n📌 Thống kê tổng quan:")
    print(df_final[["close", "daily_return", "ma5", "ma21"]].describe().round(4).to_string())

    return df_final


if __name__ == "__main__":
    df = process_pipeline()