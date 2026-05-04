import pandas as pd
from datetime import datetime, timedelta
from vnstock import Vnstock
import os

# ------ Danh sách 20 ngân hàng ------
BANK_TICKERS = [
    "VCB", "BID", "CTG", "TCB", "MBB",
    "ACB", "VPB", "STB", "HDB", "TPB",
    "VIB", "MSB", "OCB", "SHB", "LPB",
    "EIB", "NAB", "PGB", "ABB", "BVB"
]

# ------ Khoảng thời gian 3 năm ------
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

os.makedirs("data", exist_ok=True) #Tạo thư mục data nếu chưa tồn tại

def collect_one_stock(ticker: str) -> pd.DataFrame:
    """Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume) của một mã cổ phiếu."""
    stock = Vnstock().stock(symbol=ticker)
    df = stock.quote.history(start=START_DATE, end=END_DATE, interval="1D")
    df["ticker"] = ticker
    return df


def collect_all_banks() -> pd.DataFrame:
    """Lấy dữ liệu toàn bộ 20 ngân hàng, gộp thành 1 DataFrame."""
    all_data = []

    for ticker in BANK_TICKERS:
        try:
            df = collect_one_stock(ticker)
            all_data.append(df)
            print(f"  ✅ {ticker}: {len(df)} phiên")
        except Exception as e:
            print(f"  ❌ {ticker}: lỗi - {e}")

    combined = pd.concat(all_data, ignore_index=True) #Gộp tất cả dữ liệu lại với nhau
    combined.to_csv("data/raw_data.csv", index=False) #Lưu dữ liệu thô vào file CSV
    print(f"\n💾 Đã lưu: data/raw_data.csv  |  Tổng {len(combined)} dòng")
    return combined


if __name__ == "__main__":
    print("=" * 50)
    print("THU THẬP DỮ LIỆU 20 NGÂN HÀNG")
    print(f"Từ {START_DATE} đến {END_DATE}")
    print("=" * 50)
    df = collect_all_banks()
    print(df.head())