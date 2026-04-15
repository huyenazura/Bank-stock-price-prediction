"""
app.py — Streamlit dashboard dự báo giá cổ phiếu ngân hàng.
Chạy: streamlit run app.py
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import streamlit as st
from vnstock import Vnstock
from datetime import datetime, timedelta

import config
from models_def import LSTMModel, DLinearModel, NLinearModel
from data_utils import preprocess_data


# ─────────────────────────────────────────────────────────────────────────────
# 1. CẤU HÌNH TRANG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VN Bank Stock Forecast",
    page_icon="🏦",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TẢI DỮ LIỆU (cache 1 giờ)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str):
    try:
        vn    = Vnstock()
        stock = vn.stock(symbol=symbol, source="VCI")
        df    = stock.quote.history(
            start=config.START_DATE,
            end=config.END_DATE,
            interval="1D",
        )
        if df is None or len(df) == 0:
            return None
        df["time"]         = pd.to_datetime(df["time"])
        df                 = df.sort_values("time").reset_index(drop=True)
        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["close_log"]    = np.log(df["close"])
        df["ma20"]         = df["close"].rolling(20).mean().bfill()
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. DỰ BÁO — inverse transform log return → giá thực
# ─────────────────────────────────────────────────────────────────────────────
def build_model(name: str, window: int):
    if name == "LSTM":
        return LSTMModel(1, config.HIDDEN_SIZE, 1)
    elif name == "DLinear":
        return DLinearModel(window, pred_len=1)
    else:
        return NLinearModel(window, pred_len=1)


def get_prediction(df, symbol: str, model_name: str, window: int):
    """
    Trả về (preds_price, actuals_price) — đơn vị VND thực tế.
    Hoặc (None, None) nếu thiếu file weight.
    """
    path = f"models/{model_name.lower()}_{symbol}_{window}d.pth"
    if not os.path.exists(path):
        return None, None

    data = preprocess_data(df)       # log return, shape [N, 1]
    model = build_model(model_name, window)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    inputs = [data[i : i + window] for i in range(len(data) - window)]
    X      = torch.FloatTensor(np.array(inputs))

    with torch.no_grad():
        log_return_preds = model(X).numpy().flatten()

    # Inverse transform: giá_dự_báo(t+1) = giá_thực(t) × exp(log_return_pred)
    close      = df["close"].values
    base       = close[window : window + len(log_return_preds)]
    preds      = base * np.exp(log_return_preds)
    actuals    = close[window + 1 : window + 1 + len(log_return_preds)]

    min_len = min(len(preds), len(actuals))
    return preds[:min_len], actuals[:min_len]


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER UI
# ─────────────────────────────────────────────────────────────────────────────
def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", labelsize=8)


def check_weights_exist(symbol: str) -> dict:
    """Trả về dict {window: {model: bool}} cho biết file weight nào đã có."""
    status = {}
    for w in config.INPUT_WINDOWS:
        status[w] = {}
        for m in ["LSTM", "DLinear", "NLinear"]:
            status[w][m] = os.path.exists(f"models/{m.lower()}_{symbol}_{w}d.pth")
    return status


# ─────────────────────────────────────────────────────────────────────────────
# 5. GIAO DIỆN
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 VN Bank Stock Forecast")
st.caption(f"Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ── Sidebar: chọn mã & tham số ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    selected_symbol = st.selectbox(
        "Chọn mã ngân hàng",
        options=list(dict.fromkeys(config.SYMBOLS)),
    )
    selected_model  = st.selectbox(
        "Mô hình dự báo chính",
        options=["NLinear", "DLinear", "LSTM"],
    )
    selected_window = st.selectbox(
        "Lookback window (ngày)",
        options=config.INPUT_WINDOWS,
    )
    n_recent = st.slider(
        "Số phiên hiển thị trên biểu đồ",
        min_value=30, max_value=250, value=60, step=10,
    )
    st.divider()

    # Trạng thái file weight
    st.subheader("📁 Trạng thái weight")
    ws = check_weights_exist(selected_symbol)
    rows = []
    for w, models in ws.items():
        for m, ok in models.items():
            rows.append({"Window": f"{w}d", "Model": m, "Có weight": "✅" if ok else "❌"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── Tải dữ liệu ──────────────────────────────────────────────────────────────
df = fetch_stock_data(selected_symbol)

if df is None:
    st.error("❌ Không thể tải dữ liệu. Kiểm tra kết nối hoặc mã cổ phiếu.")
    st.stop()

# ── Metrics nhanh ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Giá hiện tại",          f"{df['close'].iloc[-1]:,.0f} VND")
c2.metric("Biến động ngày",        f"{df['daily_return'].iloc[-1]*100:.2f}%")
c3.metric("MA20",                  f"{df['ma20'].iloc[-1]:,.0f} VND")
c4.metric("Khối lượng (phiên cuối)", f"{df['volume'].iloc[-1]:,.0f}")

st.divider()

# ── Tab chính ────────────────────────────────────────────────────────────────
tab_chart, tab_forecast = st.tabs(["📊 Phân tích dữ liệu", "🔮 Dự báo ML"])

# ===== TAB 1: Phân tích =====
with tab_chart:
    st.subheader(f"Phân tích đặc trưng — {selected_symbol}")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.write("### 💰 Giá & MA20")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["time"], df["close"], color="#1f77b4", linewidth=1.8, label="Close")
        ax.plot(df["time"], df["ma20"],  color="#ff7f0e", linewidth=1.2,
                linestyle="--", label="MA20")
        ax.set_ylabel("Giá (VND)")
        ax.legend(fontsize=8)
        format_date_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    with r1c2:
        st.write("### 🌀 Log Price")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["time"], df["close_log"], color="#2ca02c", linewidth=1.8)
        ax.set_ylabel("Log(Giá)")
        format_date_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.write("### 📊 Daily Return (%)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["time"], df["daily_return"],
                alpha=0.4, color="gray", label="Return")
        ax.plot(df["time"], df["daily_return"].rolling(20).mean(),
                color="red", label="MA20")
        ax.axhline(0, color="black", linewidth=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(fontsize=8)
        format_date_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    with r2c2:
        st.write("### 📦 Trading Volume")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(df["time"], df["volume"], color="#d62728", alpha=0.5)
        ax.set_ylabel("Số lượng CP")
        format_date_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

# ===== TAB 2: Dự báo =====
with tab_forecast:
    st.subheader(f"Dự báo ML — {selected_symbol} | Window {selected_window}d")

    col_single, col_compare = st.columns(2)

    # ── Biểu đồ 1 model được chọn ──────────────────────────────────────────
    with col_single:
        st.write(f"### {selected_model}")
        preds, actuals = get_prediction(df, selected_symbol, selected_model, selected_window)

        if preds is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(actuals[-n_recent:], color="black",  linewidth=1.5, label="Thực tế")
            ax.plot(preds[-n_recent:],   color="red",    linewidth=1.5,
                    linestyle="--", label=f"Dự báo {selected_model}")
            ax.set_title(f"{selected_symbol} — {selected_model} (window {selected_window}d)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            next_price = preds[-1]
            last_price = df["close"].iloc[-1]
            delta_pct  = (next_price - last_price) / last_price * 100
            st.success(
                f"💡 **Giá dự báo phiên tới:** {next_price:,.0f} VND "
                f"({'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.2f}%)"
            )
        else:
            st.warning(
                f"Chưa có file weight cho **{selected_model}** "
                f"— {selected_symbol} — window {selected_window}d.\n\n"
                "Hãy chạy `python train.py` trước."
            )

    # ── Biểu đồ so sánh 3 model ────────────────────────────────────────────
    with col_compare:
        st.write("### So sánh 3 mô hình")
        colors  = {"NLinear": "blue", "DLinear": "green", "LSTM": "orange"}
        fig, ax = plt.subplots(figsize=(8, 4))

        ref_plotted = False
        any_model   = False

        for m in ["NLinear", "DLinear", "LSTM"]:
            p, a = get_prediction(df, selected_symbol, m, selected_window)
            if p is not None:
                any_model = True
                if not ref_plotted:
                    ax.plot(a[-n_recent:], color="black", linewidth=2, label="Thực tế")
                    ref_plotted = True
                ax.plot(p[-n_recent:], color=colors[m],
                        linewidth=1.5, linestyle="--", alpha=0.85, label=m)

        if any_model:
            ax.set_title(f"{selected_symbol} — window {selected_window}d")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            st.info("So sánh để xác định model nào bám sát thực tế nhất.")
        else:
            st.warning("Chưa có file weight nào. Chạy `python train.py` trước.")
            plt.close(fig)