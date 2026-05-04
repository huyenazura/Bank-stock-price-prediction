# ============================================================
# BƯỚC 2: KHAI PHÁ DỮ LIỆU (EDA)
# Thống kê mô tả, kiểm tra null, phân phối, tương quan
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Thư viện vẽ biểu đồ cơ bản
import seaborn as sns #Thư viện vẽ biểu đồ đẹp hơn
import os

os.makedirs("data/eda_plots", exist_ok=True)

def explore_data(df: pd.DataFrame):
    print("=" * 50)
    print("KHAI PHÁ DỮ LIỆU (EDA)")
    print("=" * 50)

    # --- 1. Thông tin cơ bản ---
    print("\n📌 Shape:", df.shape)
    print("\n📌 Kiểu dữ liệu:\n", df.dtypes)
    print("\n📌 5 dòng đầu:\n", df.head())

    # --- 2. Kiểm tra missing values ---
    missing = df.isnull().sum()
    print("\n📌 Missing values:\n", missing[missing > 0] if missing.any() else "  → Không có giá trị thiếu")

    # --- 3. Thống kê mô tả ---
    print("\n📌 Thống kê mô tả (giá đóng cửa):")
    print(df.groupby("ticker")["close"].describe().round(2))

    # --- 4. Số phiên giao dịch mỗi ngân hàng ---
    print("\n📌 Số phiên giao dịch mỗi ngân hàng:")
    print(df.groupby("ticker").size().sort_values(ascending=False))

    # --- 5. Ma trận tương quan giá đóng cửa ---
    pivot = df.pivot_table(index="time", columns="ticker", values="close")
    corr  = pivot.corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, annot=False, cmap="RdYlGn", linewidths=0.3, vmin=-1, vmax=1)
    plt.title("Ma trận tương quan giá đóng cửa - 30 Ngân hàng")
    plt.tight_layout()
    plt.savefig("data/eda_plots/correlation_matrix.png")
    plt.close()
    print("✅ Đã lưu biểu đồ: data/eda_plots/correlation_matrix.png")


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv")
    explore_data(df)