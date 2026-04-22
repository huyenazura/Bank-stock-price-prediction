"""
app.py — Streamlit dashboard dự báo giá cổ phiếu ngân hàng Việt Nam.

Chạy:  streamlit run app.py

Trang 1 — Tổng quan     : Giá cao nhất / thấp nhất, tương quan 30 NH
Trang 2 — Chi tiết NH   : Biểu đồ phân tích + dự báo ML (model tốt nhất)
Trang 3 — Đánh giá ML   : RMSE / MAE / R² — bảng so sánh + biểu đồ
Trang 4 — Tin tức       : NewsAPI theo từng mã ngân hàng
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from datetime import datetime, timedelta
import streamlit as st

import config
from models_def import build_model
from data_utils import fetch_data, preprocess_data


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình trang
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VN Bank Forecast",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tuỳ chỉnh — màu chủ đạo xanh navy / vàng tài chính
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0d1117; }
    .block-container { padding: 1.5rem 2rem; }
    h1, h2, h3 { color: #e6edf3; font-weight: 700; letter-spacing: -0.5px; }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #f0f6fc; font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono'; }
    .metric-delta-pos { color: #3fb950; font-size: 0.9rem; }
    .metric-delta-neg { color: #f85149; font-size: 0.9rem; }
    .news-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.8rem;
    }
    .news-title { color: #58a6ff; font-weight: 600; font-size: 0.95rem; }
    .news-meta  { color: #8b949e; font-size: 0.78rem; margin-top: 4px; }
    .news-desc  { color: #c9d1d9; font-size: 0.85rem; margin-top: 6px; line-height: 1.5; }
    .stSelectbox label, .stSlider label { color: #8b949e !important; }
    div[data-testid="metric-container"] { background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#0d1117",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#8b949e",
    "text.color"       : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "grid.color"       : "#21262d",
    "grid.alpha"       : 0.8,
    "legend.facecolor" : "#161b22",
    "legend.edgecolor" : "#30363d",
})


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock(symbol: str):
    df = fetch_data(symbol, config.START_DATE, config.END_DATE, verbose=False)
    if df is None:
        return None
    df["time"]         = pd.to_datetime(df["time"])
    df                 = df.sort_values("time").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["close_log"]    = np.log(df["close"])
    df["ma20"]         = df["close"].rolling(20).mean().bfill()
    df["ma60"]         = df["close"].rolling(60).mean().bfill()
    return df


@st.cache_data(ttl=7200, show_spinner=False)
def load_all_stocks():
    results = {}
    prog    = st.progress(0, text="Đang tải dữ liệu 30 ngân hàng...")
    for i, sym in enumerate(config.SYMBOLS):
        results[sym] = load_stock(sym)
        prog.progress((i + 1) / len(config.SYMBOLS), text=f"Đang tải {sym}...")
    prog.empty()
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def load_summary_csv():
    path = "logs/summary.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=1800, show_spinner=False)
def load_best_models_csv():
    path = "logs/best_models.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def get_best_model_name(symbol: str, window: int) -> str:
    """Trả về tên model tốt nhất cho (symbol, window) từ best_models.csv."""
    best = load_best_models_csv()
    if best is None:
        return "NLinear"   # fallback
    row = best[(best["symbol"] == symbol) & (best["window"] == window)]
    if row.empty:
        return "NLinear"
    return row.iloc[0]["model"]


# ─────────────────────────────────────────────────────────────────────────────
# Dự báo
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(symbol: str, model_name: str, window: int):
    """
    Dự báo toàn bộ tập dữ liệu, inverse transform → VND.
    Trả về (preds, actuals) hoặc (None, None).
    """
    path = f"models/{model_name.lower()}_{symbol}_{window}d.pth"
    if not os.path.exists(path):
        return None, None

    df = load_stock(symbol)
    if df is None:
        return None, None

    data = preprocess_data(df, verbose=False)

    inputs = np.array([data[i : i + window] for i in range(len(data) - window)])
    X      = torch.FloatTensor(inputs)

    model = build_model(model_name, window, config.HIDDEN_SIZE)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:
        return None, None
    model.eval()

    with torch.no_grad():
        lr_preds = model(X).numpy().flatten()

    close   = df["close"].values
    base    = close[window : window + len(lr_preds)]
    preds   = base * np.exp(lr_preds)
    actuals = close[window + 1 : window + 1 + len(lr_preds)]
    min_len = min(len(preds), len(actuals))
    if min_len == 0:
        return None, None
    return preds[:min_len], actuals[:min_len]


def next_day_forecast(symbol: str, window: int):
    """Dự báo giá phiên tiếp theo (VND) dùng model tốt nhất."""
    best_name = get_best_model_name(symbol, window)
    path      = f"models/{best_name.lower()}_{symbol}_{window}d.pth"
    if not os.path.exists(path):
        return None, best_name

    df = load_stock(symbol)
    if df is None:
        return None, best_name

    data = preprocess_data(df, verbose=False)
    if len(data) < window:
        return None, best_name

    x = torch.FloatTensor(data[-window:]).unsqueeze(0)   # [1, window, 1]
    model = build_model(best_name, window, config.HIDDEN_SIZE)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:
        return None, best_name
    model.eval()

    with torch.no_grad():
        lr_pred = model(x).item()

    last_close = df["close"].iloc[-1]
    return last_close * math.exp(lr_pred), best_name


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.tick_params(labelsize=8)


def fig_price(df):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df["time"], df["close"], color="#58a6ff", linewidth=1.5, label="Close")
    ax.plot(df["time"], df["ma20"],  color="#f0883e", linewidth=1,
            linestyle="--", label="MA20", alpha=0.85)
    ax.plot(df["time"], df["ma60"],  color="#bc8cff", linewidth=1,
            linestyle=":",  label="MA60", alpha=0.85)
    ax.set_ylabel("Giá (nghìn đồng)")
    ax.legend(fontsize=8)
    _date_axis(ax)
    fig.tight_layout()
    return fig


def fig_log_price(df):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df["time"], df["close_log"], color="#3fb950", linewidth=1.5)
    ax.set_ylabel("Ln(Giá)")
    _date_axis(ax)
    fig.tight_layout()
    return fig


def fig_daily_return(df):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors  = ["#3fb950" if r >= 0 else "#f85149" for r in df["daily_return"]]
    ax.bar(df["time"], df["daily_return"], color=colors, width=1.5, alpha=0.7)
    ax.plot(df["time"], df["daily_return"].rolling(20).mean(),
            color="#f0883e", linewidth=1.2, label="MA20")
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Biến động ngày")
    ax.legend(fontsize=8)
    _date_axis(ax)
    fig.tight_layout()
    return fig


def fig_volume(df):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(df["time"], df["volume"], color="#58a6ff", alpha=0.4)
    ax.plot(df["time"], df["volume"], color="#58a6ff", linewidth=0.8)
    ax.set_ylabel("Khối lượng")
    _date_axis(ax)
    fig.tight_layout()
    return fig


def fig_forecast(symbol: str, window: int, n_recent: int = 60):
    best_name           = get_best_model_name(symbol, window)
    preds, actuals      = get_prediction(symbol, best_name, window)
    horizon_label       = config.PRED_HORIZONS.get(window, f"{window}d")

    fig, ax = plt.subplots(figsize=(9, 3.8))

    if preds is None:
        ax.text(0.5, 0.5, f"Chưa có weight: {best_name}\nChạy train.py trước",
                ha="center", va="center", transform=ax.transAxes,
                color="#f85149", fontsize=11)
        fig.tight_layout()
        return fig, best_name, None, None

    show_p = preds[-n_recent:]
    show_a = actuals[-n_recent:]
    x      = np.arange(len(show_a))

    ax.plot(x, show_a, color="#c9d1d9", linewidth=1.8, label="Thực tế", zorder=5)
    ax.plot(x, show_p, color="#f0883e", linewidth=1.5, linestyle="--",
            alpha=0.9, label=f"Dự báo ({best_name})", zorder=4)
    ax.fill_between(x, show_a, show_p, alpha=0.08, color="#f0883e")

    ax.set_title(f"Dự báo {horizon_label}", fontsize=10, fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Giá (nghìn đồng)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig, best_name, preds, actuals


# ─────────────────────────────────────────────────────────────────────────────
# News
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(symbol: str, bank_name: str) -> list:
    """Lấy tin tức từ NewsAPI về ngân hàng."""
    api_key = config.NEWS_API_KEY
    if api_key == "YOUR_NEWSAPI_KEY_HERE" or not api_key:
        return []

    query = f"{bank_name} ngân hàng {symbol}"
    url   = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&language={config.NEWS_LANGUAGE}"
        f"&pageSize={config.NEWS_PAGE_SIZE}"
        f"&sortBy=publishedAt"
        f"&apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 VN Bank Forecast")
    st.caption(f"Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.divider()

    page = st.radio(
        "Điều hướng",
        ["🏠 Tổng quan", "📈 Chi tiết ngân hàng", "🤖 Đánh giá mô hình", "📰 Tin tức"],
        label_visibility="collapsed",
    )
    st.divider()

    if page in ["📈 Chi tiết ngân hàng", "📰 Tin tức"]:
        selected_symbol = st.selectbox(
            "Chọn mã ngân hàng",
            options=config.SYMBOLS,
            format_func=lambda s: f"{s} — {config.BANK_NAMES.get(s, s)}",
        )
        n_recent = st.slider("Số phiên hiển thị (dự báo)", 30, 250, 60, step=10)
    else:
        selected_symbol = config.SYMBOLS[0]
        n_recent = 60

    st.divider()
    st.caption(f"Dữ liệu: {config.START_DATE} → {config.END_DATE}")
    st.caption(f"Device: `{config.DEVICE}`")


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 1 — TỔNG QUAN
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Tổng quan":
    st.title("🏠 Tổng quan thị trường ngân hàng")

    with st.spinner("Đang tải dữ liệu toàn bộ ngân hàng..."):
        all_stocks = load_all_stocks()

    # Tổng hợp snapshot cuối ngày
    rows = []
    for sym, df in all_stocks.items():
        if df is None or len(df) == 0:
            continue
        rows.append({
            "symbol"    : sym,
            "name"      : config.BANK_NAMES.get(sym, sym),
            "close"     : df["close"].iloc[-1],
            "change_pct": df["daily_return"].iloc[-1] * 100,
            "volume"    : df["volume"].iloc[-1],
            "high_3y"   : df["close"].max(),
            "low_3y"    : df["close"].min(),
        })
    snap = pd.DataFrame(rows).dropna()

    if snap.empty:
        st.warning("Không thể tải dữ liệu. Kiểm tra kết nối mạng.")
        st.stop()

    # ── Metrics nổi bật ───────────────────────────────────────────────────────
    top_high = snap.loc[snap["close"].idxmax()]
    top_low  = snap.loc[snap["close"].idxmin()]
    top_gain = snap.loc[snap["change_pct"].idxmax()]
    top_loss = snap.loc[snap["change_pct"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Giá cao nhất hôm nay",
              f"{top_high['close']:,.1f}k",
              f"{top_high['symbol']} — {top_high['name']}")
    c2.metric("📉 Giá thấp nhất hôm nay",
              f"{top_low['close']:,.1f}k",
              f"{top_low['symbol']} — {top_low['name']}")
    c3.metric("🚀 Tăng mạnh nhất",
              f"+{top_gain['change_pct']:.2f}%",
              top_gain["symbol"])
    c4.metric("📉 Giảm mạnh nhất",
              f"{top_loss['change_pct']:.2f}%",
              top_loss["symbol"])

    st.divider()

    # ── Bảng giá tổng hợp ────────────────────────────────────────────────────
    st.subheader("📋 Bảng giá cuối phiên — 30 ngân hàng")
    display = snap[["symbol", "name", "close", "change_pct", "volume",
                    "high_3y", "low_3y"]].copy()
    display.columns = ["Mã", "Tên", "Giá (k)", "% Ngày", "KL",
                       "Đỉnh 3Y", "Đáy 3Y"]
    display["% Ngày"] = display["% Ngày"].round(2)

    def color_pct(val):
        color = "#3fb950" if val > 0 else ("#f85149" if val < 0 else "#8b949e")
        return f"color: {color}; font-weight: 600"

    st.dataframe(
        display.style.applymap(color_pct, subset=["% Ngày"]),
        use_container_width=True, height=450, hide_index=True,
    )

    st.divider()

    # ── Ma trận tương quan ────────────────────────────────────────────────────
    st.subheader("🔗 Ma trận tương quan log return — 30 ngân hàng")

    returns_dict = {}
    for sym, df in all_stocks.items():
        if df is not None and len(df) > 50:
            r = np.log(df["close"] / df["close"].shift(1)).dropna()
            returns_dict[sym] = r.values

    if len(returns_dict) >= 2:
        min_len = min(len(v) for v in returns_dict.values())
        mat     = np.column_stack([v[-min_len:] for v in returns_dict.values()])
        corr    = np.corrcoef(mat.T)
        syms    = list(returns_dict.keys())

        fig, ax = plt.subplots(figsize=(12, 10))
        im      = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(syms)))
        ax.set_yticks(range(len(syms)))
        ax.set_xticklabels(syms, rotation=90, fontsize=7)
        ax.set_yticklabels(syms, fontsize=7)
        ax.set_title("Correlation Matrix — Log Return", fontsize=12, fontweight="bold",
                     color="#e6edf3", pad=15)

        # Giá trị trong ô
        for i in range(len(syms)):
            for j in range(len(syms)):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=5.5, color="#0d1117" if abs(corr[i,j]) > 0.5 else "#c9d1d9")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Bar chart giá cuối phiên ──────────────────────────────────────────────
    st.subheader("📊 So sánh giá đóng cửa — Toàn bộ ngân hàng")
    snap_sorted = snap.sort_values("close", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars    = ax.barh(snap_sorted["symbol"], snap_sorted["close"],
                      color=["#3fb950" if c >= 0 else "#f85149"
                             for c in snap_sorted["change_pct"]],
                      alpha=0.8, edgecolor="#0d1117", linewidth=0.3)
    ax.set_xlabel("Giá đóng cửa (nghìn đồng)")
    ax.set_title("Giá đóng cửa phiên gần nhất", fontsize=11, fontweight="bold",
                 color="#e6edf3")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 2 — CHI TIẾT NGÂN HÀNG
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Chi tiết ngân hàng":
    sym       = selected_symbol
    bank_name = config.BANK_NAMES.get(sym, sym)

    st.title(f"📈 {sym} — {bank_name}")

    df = load_stock(sym)
    if df is None:
        st.error(f"Không thể tải dữ liệu cho {sym}.")
        st.stop()

    # Metrics nhanh
    last   = df.iloc[-1]
    prev   = df.iloc[-2]
    delta  = (last["close"] - prev["close"]) / prev["close"] * 100
    arrow  = "▲" if delta >= 0 else "▼"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá đóng cửa",    f"{last['close']:,.1f}k",       f"{arrow} {abs(delta):.2f}%")
    col2.metric("MA20",             f"{last['ma20']:,.1f}k")
    col3.metric("Khối lượng",       f"{last['volume']:,.0f}")
    col4.metric("Số phiên dữ liệu", f"{len(df)}")

    st.divider()

    # ── Tab 1: Phân tích dữ liệu ──────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Phân tích dữ liệu", "🔮 Dự báo ML"])

    with tab1:
        st.subheader("Stock Price Over Time")
        st.pyplot(fig_price(df))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Log-transformed Price")
            st.pyplot(fig_log_price(df))
        with c2:
            st.subheader("Daily Returns Over Time")
            st.pyplot(fig_daily_return(df))

        st.subheader("Volume Over Time")
        st.pyplot(fig_volume(df))

    # ── Tab 2: Dự báo ML ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Dự báo giá — Model tốt nhất (theo RMSE thấp nhất)")

        best_df = load_best_models_csv()
        if best_df is None:
            st.info("💡 Chưa có kết quả đánh giá. Hãy chạy `evaluate.py` để tự động chọn model tốt nhất.")

        for window in config.INPUT_WINDOWS:
            horizon_label = config.PRED_HORIZONS.get(window, f"{window}d")
            st.markdown(f"### 🗓️ Dự báo {horizon_label}")

            fig, best_name, preds, actuals = fig_forecast(sym, window, n_recent)

            # Dự báo phiên tiếp theo
            forecast_price, _ = next_day_forecast(sym, window)
            if forecast_price is not None:
                last_close = df["close"].iloc[-1]
                diff_pct   = (forecast_price - last_close) / last_close * 100
                sign       = "▲" if diff_pct >= 0 else "▼"
                color      = "green" if diff_pct >= 0 else "red"
                st.markdown(
                    f"**Model sử dụng:** `{best_name}`  "
                    f"| **Giá dự báo phiên tới:** `{forecast_price:,.1f}k VND` "
                    f"<span style='color:{color};font-weight:700'>{sign} {abs(diff_pct):.2f}%</span>",
                    unsafe_allow_html=True,
                )

            st.pyplot(fig)
            plt.close(fig)
            st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 3 — ĐÁNH GIÁ MÔ HÌNH
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 Đánh giá mô hình":
    st.title("🤖 Đánh giá hiệu suất mô hình")
    st.caption("RMSE · MAE · R²  —  tập kiểm tra độc lập 20% cuối")

    summary = load_summary_csv()
    if summary is None:
        st.warning("Chưa có file `logs/summary.csv`. Hãy chạy `python evaluate.py` trước.")
        st.code("python evaluate.py")
        st.stop()

    # Chuẩn hoá tên cột
    summary.columns = [c.lower() for c in summary.columns]

    # ── Bộ lọc ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    sel_sym    = f1.multiselect("Lọc mã",    options=sorted(summary["symbol"].unique()),
                                 default=sorted(summary["symbol"].unique())[:5])
    sel_win    = f2.multiselect("Lọc window", options=sorted(summary["window"].unique()),
                                 default=sorted(summary["window"].unique()))
    sel_mod    = f3.multiselect("Lọc model",  options=sorted(summary["model"].unique()),
                                 default=sorted(summary["model"].unique()))

    mask = (
        summary["symbol"].isin(sel_sym if sel_sym else summary["symbol"].unique()) &
        summary["window"].isin(sel_win if sel_win else summary["window"].unique()) &
        summary["model"].isin(sel_mod if sel_mod else summary["model"].unique())
    )
    filtered = summary[mask].copy()

    # ── Metrics tổng hợp ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng cấu hình", f"{len(filtered)}")
    m2.metric("RMSE trung bình", f"{filtered['rmse'].mean():.4f}")
    m3.metric("MAE trung bình",  f"{filtered['mae'].mean():.4f}")
    m4.metric("R² trung bình",   f"{filtered['r2'].mean():.4f}")

    st.divider()

    # ── Bảng chi tiết ────────────────────────────────────────────────────────
    st.subheader("📋 Bảng chi tiết RMSE / MAE / R²")

    def highlight_best(s):
        """Tô màu RMSE thấp nhất và R2 cao nhất trong nhóm."""
        if s.name in ["rmse", "mae"]:
            is_best = s == s.min()
        elif s.name == "r2":
            is_best = s == s.max()
        else:
            return [""] * len(s)
        return ["background-color: #1a3a2a; color: #3fb950; font-weight: 700"
                if v else "" for v in is_best]

    st.dataframe(
        filtered.sort_values(["symbol", "window", "rmse"])
                .style.apply(highlight_best, subset=["rmse", "mae", "r2"]),
        use_container_width=True, height=420, hide_index=True,
    )

    st.divider()

    # ── Bar chart RMSE theo model ─────────────────────────────────────────────
    st.subheader("📊 So sánh RMSE trung bình theo mô hình × window")
    grouped = (filtered.groupby(["model", "window"])["rmse"]
                       .mean().reset_index())

    windows = sorted(grouped["window"].unique())
    models  = ["LSTM", "DLinear", "NLinear"]
    x       = np.arange(len(windows))
    width   = 0.25
    colors_map = {"LSTM": "#e74c3c", "DLinear": "#2ecc71", "NLinear": "#3498db"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(models):
        sub  = grouped[grouped["model"] == m].sort_values("window")
        vals = sub["rmse"].values if len(sub) == len(windows) else [0] * len(windows)
        ax.bar(x + i * width, vals, width,
               label=m, color=colors_map[m], alpha=0.85, edgecolor="#0d1117")

    ax.set_xticks(x + width)
    ax.set_xticklabels([config.PRED_HORIZONS.get(w, str(w)) for w in windows], fontsize=9)
    ax.set_ylabel("RMSE trung bình")
    ax.set_title("RMSE theo Model × Horizon", fontsize=11, fontweight="bold", color="#e6edf3")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Scatter RMSE vs R2 ────────────────────────────────────────────────────
    st.subheader("🔍 Scatter: RMSE vs R² (toàn bộ cấu hình)")
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in models:
        sub = filtered[filtered["model"] == m]
        ax.scatter(sub["rmse"], sub["r2"], label=m,
                   color=colors_map[m], alpha=0.7, s=40, edgecolors="#0d1117", linewidth=0.3)
    ax.set_xlabel("RMSE (thấp hơn = tốt hơn)")
    ax.set_ylabel("R² (cao hơn = tốt hơn)")
    ax.set_title("RMSE vs R²", fontsize=11, fontweight="bold", color="#e6edf3")
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Model tốt nhất ────────────────────────────────────────────────────────
    st.subheader("🏆 Model tốt nhất cho từng mã × window (RMSE thấp nhất)")
    best_df = load_best_models_csv()
    if best_df is not None:
        sel_best = best_df[best_df["symbol"].isin(
            sel_sym if sel_sym else best_df["symbol"].unique()
        )]
        # Pivot table
        pivot = sel_best.pivot_table(
            index="symbol", columns="window", values="model", aggfunc="first"
        )
        pivot.columns = [config.PRED_HORIZONS.get(c, str(c)) for c in pivot.columns]

        def color_model(val):
            m = {"LSTM": "#3a1f1f", "DLinear": "#1a3a22", "NLinear": "#1a263a"}
            return f"background-color: {m.get(val, '')}; color: #e6edf3; font-weight: 600"

        st.dataframe(
            pivot.style.applymap(color_model),
            use_container_width=True,
        )
    else:
        st.info("Chưa có file `logs/best_models.csv`.")


# ─────────────────────────────────────────────────────────────────────────────
# TRANG 4 — TIN TỨC
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📰 Tin tức":
    sym       = selected_symbol
    bank_name = config.BANK_NAMES.get(sym, sym)

    st.title(f"📰 Tin tức — {sym} ({bank_name})")

    if config.NEWS_API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        st.warning(
            "⚠️  Chưa cấu hình API key NewsAPI.\n\n"
            "1. Đăng ký miễn phí tại [newsapi.org](https://newsapi.org)\n"
            "2. Mở file `config.py` và thay `YOUR_NEWSAPI_KEY_HERE` bằng key thật.\n"
            "3. Reload trang."
        )
    else:
        with st.spinner(f"Đang tải tin tức về {sym}..."):
            articles = fetch_news(sym, bank_name)

        if not articles:
            st.info("Không tìm thấy tin tức gần đây. Thử đổi từ khoá hoặc kiểm tra API key.")
        else:
            st.caption(f"Tìm thấy {len(articles)} bài viết gần đây")
            for art in articles:
                title       = art.get("title",       "Không có tiêu đề")
                description = art.get("description", "")
                url         = art.get("url",         "#")
                source      = art.get("source", {}).get("name", "")
                published   = art.get("publishedAt", "")[:10]

                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title"><a href="{url}" target="_blank"
                        style="text-decoration:none; color:#58a6ff;">{title}</a></div>
                    <div class="news-meta">📰 {source} &nbsp;|&nbsp; 🗓️ {published}</div>
                    <div class="news-desc">{description}</div>
                </div>
                """, unsafe_allow_html=True)

    # Biến động giá gần đây ngay dưới tin tức
    st.divider()
    st.subheader(f"📈 Giá {sym} — 90 phiên gần nhất")
    df = load_stock(sym)
    if df is not None:
        df_recent = df.tail(90)
        fig, ax   = plt.subplots(figsize=(11, 3.5))
        ax.plot(df_recent["time"], df_recent["close"],
                color="#58a6ff", linewidth=1.8)
        ax.fill_between(df_recent["time"], df_recent["close"],
                        df_recent["close"].min() * 0.995,
                        color="#58a6ff", alpha=0.12)
        ax.set_ylabel("Giá (nghìn đồng)")
        _date_axis(ax)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)