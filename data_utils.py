"""
data_utils.py — Thu thập, khai phá & xử lý dữ liệu toàn diện cho PyTorch.

═══════════════════════════════════════════════════════════════════
  GIAI ĐOẠN 1 — LÀM SẠCH & CHUẨN HÓA FORMAT (Data Cleaning)
    1.1  Convert time → datetime64  → chuẩn hóa format thời gian, xóa NaT
    1.2  Sort theo thời gian        → tăng dần, đảm bảo chuỗi đúng thứ tự
    1.3  Kiểm tra missing           → báo cáo NaN / Inf / cột thiếu / trùng ngày
    1.4  Chuẩn hóa tên cột         → lowercase, strip khoảng trắng
    1.5  Ép kiểu số                 → OHLCV → float64
    1.6  Xử lý Inf → NaN            → thay Inf trước khi fill
    1.7  Xử lý giá trị thiếu (NaN)  → ffill → bfill → drop
    1.8  Xử lý dữ liệu nhiễu        → IQR fence (1%–99%)
    1.9  Xử lý lỗi nhập liệu        → close ≤ 0, high < low, volume < 0
    1.10 Loại bỏ ngày trùng lặp     → giữ dòng cuối cùng theo time

  GIAI ĐOẠN 2 — BIẾN ĐỔI DỮ LIỆU (Feature Engineering)
    2.1  Log return                  → log(close_t / close_{t-1})
    2.2  Clip ngoại lệ ±5σ          → loại bỏ spike bất thường
    2.3  MA (5, 10, 20, 60)          → Moving Average
    2.4  EMA (12, 26)                → Exponential Moving Average
    2.5  RSI (14)                    → Relative Strength Index
    2.6  MACD & Signal               → MACD = EMA12 - EMA26
    2.7  Bollinger Bands             → BB20 ±2σ (width, %B)
    2.8  Rate of Change              → ROC5, ROC10
    2.9  Chuẩn hóa MinMaxScaler      → tất cả feature về [0, 1]

  GIAI ĐOẠN 3 — TÍCH HỢP DỮ LIỆU (Data Integration)
    3.1  Định dạng thống nhất        → chuẩn hóa tên cột, kiểu dữ liệu
    3.2  Clean + Transform mỗi mã   → dict {symbol: df_transformed}
    3.3  Báo cáo tính toàn vẹn       → shape, missing, status từng mã
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# StockDataset
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """Chuyển mảng features [N, F] thành cặp (X_window, y_next) để train."""

    def __init__(self, data: np.ndarray, window_size: int, target_col: int = 0):
        self.data       = torch.FloatTensor(data)
        self.window     = window_size
        self.target_col = target_col

    def __len__(self):
        return max(0, len(self.data) - self.window)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window, self.target_col]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Thu thập dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str, start: str, end: str, verbose: bool = True):
    try:
        from vnstock import Quote
        quote = Quote(symbol=symbol, source="VCI")
        df    = quote.history(start=start, end=end, interval="1D")
        if df is None or len(df) < 50:
            if verbose:
                print(f"  [{symbol}] ⚠️  Không đủ dữ liệu ({0 if df is None else len(df)} dòng).")
            return None
        if verbose:
            print(f"  [{symbol}] ✅ Tải thành công — {len(df)} phiên "
                  f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})")
        return df
    except Exception as e:
        if verbose:
            print(f"  [{symbol}] ❌ Lỗi: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Khai phá dữ liệu (EDA)
# ─────────────────────────────────────────────────────────────────────────────

def explore_data(df: pd.DataFrame, symbol: str = "") -> None:
    tag = f"[{symbol}] " if symbol else ""
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  {tag}KHAI PHÁ DỮ LIỆU THÔ (EDA)")
    print(sep)
    print(f"  Shape          : {df.shape}")
    print(f"  Cột            : {df.columns.tolist()}")
    print(f"\n  Kiểu dữ liệu:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<18}: {dtype}")
    print(f"\n  Thống kê mô tả (OHLCV):")
    ohlcv = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    print(df[ohlcv].describe().round(2).to_string() if ohlcv else df.describe().round(2).to_string())
    print(f"\n  Giá trị NaN:")
    for col, cnt in df.isnull().sum().items():
        print(f"    {col:<18}: {cnt}" + ("  ⚠️" if cnt > 0 else ""))
    inf_found = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        n = int(np.isinf(df[col].values).sum())
        if n > 0:
            inf_found[col] = n
    if inf_found:
        print(f"\n  Giá trị Inf: {inf_found}  ⚠️")
    dup = df.duplicated().sum()
    print(f"\n  Dòng trùng lặp : {dup}" + (" ⚠️" if dup > 0 else ""))
    print(f"\n  5 dòng đầu:")
    print(df.head(5).to_string())
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, body: str, verbose: bool) -> None:
    if verbose:
        print(f"\n{'─'*65}")
        print(f"  {title}")
        print(f"{'─'*65}")
        print(body)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ═════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 1 — LÀM SẠCH DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    # chuẩn hóa thời gian
    df['time'] = pd.to_datetime(df['time'])
    
    #sort theo thời gian
    df = df.sort_values('time')
    
    # Xử lý missing value
    df = df.dropna()
    
    # xử lí duplicate
    df = df.drop_duplicates()
    
    # Tạo  essential features  
    df['daily_return'] = df['close'].pct_change()
    df['close_log'] = np.log(df['close'])
    
    # Xử lý giá trị thiếu (NaN) trong daily_return sau khi tính pct_change
    if df['daily_return'].isnull().sum() > 0:
        df['daily_return'].fillna(0, inplace=True)
    
    return df



# ═════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 2 — BIẾN ĐỔI DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def transform_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"\n{'═'*65}")
        print("  GIAI ĐOẠN 2 — BIẾN ĐỔI DỮ LIỆU")
        print(f"{'═'*65}")

    df    = df.copy()
    close = df["close"]

    # 2.1 Log return
    df["log_return"] = np.log(close / close.shift(1))
    _section("Bước 2.1 — Log return [log(Pt/Pt-1)]",
             f"  Min {df['log_return'].min():.6f}  Max {df['log_return'].max():.6f}  "
             f"Mean {df['log_return'].mean():.6f}  Std {df['log_return'].std():.6f}", verbose)

    # 2.2 Clip ±5σ
    mu, sigma = df["log_return"].mean(), df["log_return"].std()
    if sigma > 0:
        lo, hi  = mu - 5*sigma, mu + 5*sigma
        n_clip  = int(((df["log_return"] < lo) | (df["log_return"] > hi)).sum())
        df["log_return"] = df["log_return"].clip(lo, hi)
        _section("Bước 2.2 — Clip ngoại lệ ±5σ",
                 f"  Ngưỡng [{lo:.6f}, {hi:.6f}]  |  Điểm bị clip: {n_clip}", verbose)

    # 2.3 MA
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = close.rolling(w).mean()
    _section("Bước 2.3 — Moving Average (MA5, MA10, MA20, MA60)",
             f"  MA20 (5 cuối):\n{df['ma20'].tail(5).round(2).to_string()}", verbose)

    # 2.4 EMA
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    _section("Bước 2.4 — EMA (EMA12, EMA26)",
             f"  EMA12 (5 cuối):\n{df['ema12'].tail(5).round(2).to_string()}", verbose)

    # 2.5 RSI
    df["rsi14"] = _rsi(close, 14)
    _section("Bước 2.5 — RSI (14)",
             f"  Min {df['rsi14'].min():.2f}  Max {df['rsi14'].max():.2f}\n"
             f"  RSI14 (5 cuối):\n{df['rsi14'].tail(5).round(2).to_string()}", verbose)

    # 2.6 MACD
    df["macd"]        = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    _section("Bước 2.6 — MACD & Signal",
             f"  MACD (5 cuối):\n"
             f"{df[['macd','macd_signal','macd_hist']].tail(5).round(4).to_string()}", verbose)

    # 2.7 Bollinger Bands
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2*bb_std
    df["bb_lower"] = bb_mid - 2*bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    _section("Bước 2.7 — Bollinger Bands (BB20 ±2σ)",
             f"  BB (5 cuối):\n"
             f"{df[['bb_upper','bb_lower','bb_width','bb_pct']].tail(5).round(4).to_string()}", verbose)

    # 2.8 ROC
    df["roc5"]  = close.pct_change(5)
    df["roc10"] = close.pct_change(10)
    _section("Bước 2.8 — Rate of Change (ROC5, ROC10)",
             f"  ROC5 (5 cuối):\n{df['roc5'].tail(5).round(4).to_string()}", verbose)

    # Drop NaN từ rolling
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if verbose:
        print(f"\n  ⚙️  Drop NaN (rolling/shift): {rows_before - len(df)} dòng → còn {len(df)}")

    # 2.9 MinMax scale tất cả feature (trừ log_return và time)
    exclude        = {"time", "log_return"}
    feature_cols   = [c for c in df.columns
                      if c not in exclude and df[c].dtype in [np.float32, np.float64, float]]
    scaler_params  = {}
    for col in feature_cols:
        cmin, cmax = df[col].min(), df[col].max()
        denom      = cmax - cmin
        df[col]    = (df[col] - cmin) / denom if denom > 0 else 0.0
        scaler_params[col] = (cmin, cmax)

    _section("Bước 2.9 — Chuẩn hóa MinMax [0, 1]",
             f"  Số cột scale : {len(feature_cols)}\n"
             f"  Cột scale    : {feature_cols}\n"
             f"  Cột KHÔNG    : {list(exclude)}\n\n"
             f"  Kiểm tra sau scale (5 cuối):\n"
             f"{df[feature_cols[:5]].tail(5).round(4).to_string()}", verbose)

    df.attrs["scaler_params"] = scaler_params
    df.attrs["feature_cols"]  = feature_cols

    if verbose:
        print(f"\n  ✅ Sau biến đổi: {df.shape} — {len(feature_cols)} features + log_return")
        print(f"  Tất cả cột: {df.columns.tolist()}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# GIAI ĐOẠN 3 — TÍCH HỢP DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def integrate_data(data_dict: dict, verbose: bool = False) -> tuple:
    if verbose:
        print(f"\n{'═'*65}")
        print(f"  GIAI ĐOẠN 3 — TÍCH HỢP DỮ LIỆU ({len(data_dict)} mã)")
        print(f"{'═'*65}")

    processed, report = {}, []

    for symbol, df_raw in data_dict.items():
        row = {"symbol": symbol, "status": "OK",
               "raw_rows": 0, "clean_rows": 0, "final_rows": 0,
               "features": 0, "note": ""}
        if df_raw is None:
            row.update({"status": "SKIP", "note": "Không có dữ liệu"})
            report.append(row)
            continue

        row["raw_rows"] = len(df_raw)
        try:
            df = df_raw.copy()
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            df_clean             = clean_data(df,       verbose=False)
            df_final             = transform_data(df_clean, verbose=False)
            row["clean_rows"]    = len(df_clean)
            row["final_rows"]    = len(df_final)
            row["features"]      = len(df_final.attrs.get("feature_cols", []))
            processed[symbol]    = df_final
        except Exception as e:
            row.update({"status": "ERROR", "note": str(e)})

        report.append(row)

    report_df = pd.DataFrame(report)

    if verbose:
        print(f"\n  ─── Báo cáo tích hợp ───")
        print(report_df.to_string(index=False))
        ok   = (report_df["status"] == "OK").sum()
        skip = (report_df["status"] == "SKIP").sum()
        err  = (report_df["status"] == "ERROR").sum()
        print(f"\n  ✅ OK: {ok}   ⚠️  SKIP: {skip}   ❌ ERROR: {err}")
        if processed:
            common = None
            for df in processed.values():
                dates  = set(df["time"].dt.date) if "time" in df.columns else set()
                common = dates if common is None else common & dates
            print(f"  Phiên chung (giao nhau): {len(common) if common else 'N/A'}")

    return processed, report_df


# ═════════════════════════════════════════════════════════════════════════════
# preprocess_data — wrapper tương thích với train.py / evaluate.py / app.py
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame, verbose: bool = False,
                    return_col_names: bool = False):
    """
    Pipeline đầy đủ: clean → transform → ndarray [N, F].

    Cột 0  = log_return (target)
    Cột 1+ = features kỹ thuật đã MinMax scale

    Args:
        return_col_names : Nếu True, trả về (arr, col_names) thay vì chỉ arr.
                           Dùng trong main() để in bảng với tên cột thật.
    Returns:
        arr              : np.ndarray [N, F] float32
        col_names (opt.) : list[str] tên từng cột theo thứ tự
    """
    df_clean = clean_data(df,          verbose=verbose)
    df_final = transform_data(df_clean, verbose=verbose)

    feature_cols = df_final.attrs.get("feature_cols", [])
    ordered      = ["log_return"] + [c for c in feature_cols if c in df_final.columns]
    arr          = df_final[ordered].values.astype(np.float32)

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  OUTPUT preprocess_data()")
        print(f"{'═'*65}")
        print(f"  Shape        : {arr.shape}  [N phiên × {arr.shape[1]} features]")
        print(f"  Thứ tự cột   : {ordered}")
        print(f"  dtype        : {arr.dtype}")
        print(f"  log_return (5 đầu): {arr[:5, 0]}")

    if return_col_names:
        return arr, ordered
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Main — toàn bộ pipeline: thu thập → xử lý → train model
# ─────────────────────────────────────────────────────────────────────────────

def _print_data_table(arr: np.ndarray, sym: str,
                      col_names: list = None) -> None:
    """
    In TOÀN BỘ dữ liệu sau xử lý — mọi hàng, mọi cột, không bỏ sót.

    Args:
        arr       : ndarray [N, F] đã xử lý.
        sym       : Tên mã cổ phiếu.
        col_names : Tên cột thực tế từ DataFrame (truyền vào từ transform_data).
    """
    N, F = arr.shape

    # ── Tên cột ───────────────────────────────────────────────────────────────
    default_names = ["log_return",  "close",       "open",        "high",
                     "low",         "volume",      "ma5",         "ma10",
                     "ma20",        "ma60",        "ema12",       "ema26",
                     "rsi14",       "macd",        "macd_signal", "macd_hist",
                     "bb_upper",    "bb_lower",    "bb_width",    "bb_pct",
                     "roc5",        "roc10"]
    labels = (col_names if col_names and len(col_names) == F
              else default_names[:F])
    W = F * 13 + 10

    # ── Tiêu đề ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  [{sym}] TOÀN BỘ DỮ LIỆU ĐẦU VÀO SAU XỬ LÝ")
    print(f"  Số phiên  : {N}")
    print(f"  Số feature: {F}")
    print(f"  Thứ tự cột: {labels}")
    print(f"{'═' * W}")

    # ── Header cột ────────────────────────────────────────────────────────────
    print(f"  {'idx':>5}  " + "  ".join(f"{c:>12}" for c in labels))
    print(f"  {'─' * (W - 2)}")

    # ── In TỪNG HÀNG — không bỏ sót hàng nào ────────────────────────────────
    for i in range(N):
        row_vals = "  ".join(f"{arr[i, j]:+12.6f}" for j in range(F))
        print(f"  {i:5d}  {row_vals}")

    # ── Thống kê tổng hợp từng cột ────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  THỐNG KÊ TỪNG FEATURE  (N = {N} phiên)")
    print(f"{'─' * W}")
    hdr = f"  {{:<16}} {{:>12}} {{:>12}} {{:>14}} {{:>14}} {{:>12}} {{:>6}}"
    print(hdr.format("Feature", "Min", "Max", "Mean", "Std", "Median", "NaN"))
    print(f"  {'─' * (W - 2)}")
    for j, name in enumerate(labels):
        col = arr[:, j]
        print(hdr.format(
            name,
            f"{col.min():+.6f}",
            f"{col.max():+.6f}",
            f"{col.mean():+.8f}",
            f"{col.std():.8f}",
            f"{float(np.median(col)):+.6f}",
            str(int(np.isnan(col).sum())),
        ))
    print(f"{'═' * W}\n")


def _quick_train(arr: np.ndarray, sym: str, window: int,
                 device: str = "cpu") -> None:
    """
    Train nhanh 3 model (LSTM, DLinear, NLinear) trên arr đã xử lý.
    Dùng để xác nhận pipeline data → model chạy thông suốt.
    Không lưu weight — chỉ in val_loss cuối.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader

    try:
        import config as _cfg
        epochs    = min(20, _cfg.EPOCHS)      # chạy nhanh trong demo
        batch     = _cfg.BATCH_SIZE
        lr        = _cfg.LEARNING_RATE
        hidden    = _cfg.HIDDEN_SIZE
    except ImportError:
        epochs, batch, lr, hidden = 20, 32, 0.001, 64

    from models_def import build_model

    n_features   = arr.shape[1]
    split        = int(0.8 * len(arr))
    train_ds     = StockDataset(arr[:split], window, target_col=0)
    val_ds       = StockDataset(arr[split:], window, target_col=0)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"    ⚠️  Dataset rỗng tại window={window}d, bỏ qua train demo.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    criterion    = nn.MSELoss()

    print(f"\n  Train demo — window={window}d | "
          f"train={len(train_ds)} mẫu | val={len(val_ds)} mẫu | "
          f"X:[{window}×{n_features}]")

    for name in ["LSTM", "DLinear", "NLinear"]:
        model     = build_model(name, window, hidden, n_features=n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.to(device)

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx).view(-1), by.view(-1))
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx).view(-1), by.view(-1)).item()
        val_loss /= max(1, len(val_loader))

        print(f"    ✅ {name:8s} | {epochs} epochs | val_loss = {val_loss:.6f}")


def main():
    import sys

    # ── Đọc config ────────────────────────────────────────────────────────────
    try:
        import config
        START_DATE      = config.START_DATE
        END_DATE        = config.END_DATE
        PRETRAIN_SYMBOL = config.PRETRAIN_SYMBOL       # "VCB"
        ALL_SYMBOLS     = config.SYMBOLS               # toàn bộ 30 mã
        WINDOWS         = config.INPUT_WINDOWS
        DEVICE          = config.DEVICE
    except ImportError:
        from datetime import datetime, timedelta
        END_DATE        = datetime.now().strftime("%Y-%m-%d")
        START_DATE      = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
        PRETRAIN_SYMBOL = "VCB"
        ALL_SYMBOLS     = ["VCB", "MBB", "TCB"]
        WINDOWS         = [1, 5, 21]
        DEVICE          = "cpu"

    # Cho phép override mã từ CLI: python data_utils.py VCB MBB
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ALL_SYMBOLS

    print("═"*65)
    print("  PIPELINE HOÀN CHỈNH: VNSTOCK → XỬ LÝ → TRAIN MODEL")
    print(f"  Mã xử lý   : {symbols}")
    print(f"  Khoảng     : {START_DATE} → {END_DATE}")
    print(f"  Windows    : {WINDOWS}")
    print(f"  Device     : {DEVICE}")
    print("═"*65)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 1 — THU THẬP DỮ LIỆU THẬT TỪ VNSTOCK
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("  BƯỚC 1 — THU THẬP DỮ LIỆU TỪ VNSTOCK")
    print(f"{'═'*65}")

    raw_dict = {}
    for sym in symbols:
        raw_dict[sym] = fetch_data(sym, START_DATE, END_DATE, verbose=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 2 — KHAI PHÁ DỮ LIỆU THÔ (EDA)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("  BƯỚC 2 — KHAI PHÁ DỮ LIỆU THÔ (EDA)")
    print(f"{'═'*65}")

    for sym, df in raw_dict.items():
        if df is not None:
            explore_data(df, symbol=sym)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 3 — XỬ LÝ DỮ LIỆU (3 giai đoạn) + IN KẾT QUẢ
    # ══════════════════════════════════════════════════════════════════════════
    processed_arrays = {}   # {symbol: np.ndarray [N, F]}

    for sym, df_raw in raw_dict.items():
        if df_raw is None:
            print(f"\n  [{sym}] ❌ Bỏ qua — không có dữ liệu.")
            continue

        print(f"\n{'#'*65}")
        print(f"  BƯỚC 3 — XỬ LÝ DỮ LIỆU: {sym}")
        print(f"{'#'*65}")

        try:
            # Gọi pipeline đầy đủ với verbose=True → in chi tiết từng bước
            arr, col_names = preprocess_data(df_raw, verbose=True,
                                             return_col_names=True)
        except Exception as e:
            print(f"  ❌ Lỗi xử lý [{sym}]: {e}")
            continue

        # In TOÀN BỘ dữ liệu đầu vào sau xử lý (tất cả hàng, tất cả cột)
        _print_data_table(arr, sym, col_names=col_names)

        processed_arrays[sym] = arr

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 4 — TÍCH HỢP DỮ LIỆU (báo cáo toàn bộ mã)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*65}")
    print("  BƯỚC 4 — TÍCH HỢP DỮ LIỆU (báo cáo tổng hợp)")
    print(f"{'#'*65}")
    _, report = integrate_data(raw_dict, verbose=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 5 — ĐƯA DỮ LIỆU ĐÃ XỬ LÝ VÀO TRAIN MODEL (demo nhanh)
    # Dữ liệu thật từ vnstock → preprocess_data() → StockDataset → train loop
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*65}")
    print("  BƯỚC 5 — TRAIN MODEL TRÊN DỮ LIỆU THẬT")
    print(f"  (Demo 20 epochs — dùng data đã xử lý ở bước 3)")
    print(f"{'#'*65}")

    for sym, arr in processed_arrays.items():
        print(f"\n{'─'*65}")
        print(f"  [{sym}] — data shape: {arr.shape}")
        print(f"{'─'*65}")

        for window in WINDOWS:
            if len(arr) < window + 30:
                print(f"  ⚠️  window={window}d: không đủ dữ liệu, bỏ qua.")
                continue
            _quick_train(arr, sym, window, device=DEVICE)

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("  ✅ HOÀN THÀNH TOÀN BỘ PIPELINE")
    print(f"  Mã xử lý thành công : {list(processed_arrays.keys())}")
    print(f"  Để train đầy đủ     : python train.py")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()