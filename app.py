# ============================================================
# ỨNG DỤNG STREAMLIT - Hỗ trợ quyết định đầu tư cổ phiếu ngân hàng
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch, pickle, os, sys
import feedparser
from datetime import datetime

sys.path.append("src")
from models import MODEL_REGISTRY

# ── Cấu hình trang ──────────────────────────────────────────
st.set_page_config(
    page_title="VN Bank Stock Forecast",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #e2e8f0; }
  [data-testid="stSidebar"]          { background: #0f1629; }
  .news-card {
      background: #111827;
      border-left: 3px solid #3b82f6;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 12px;
  }
  .news-card a { color: #60a5fa; text-decoration: none; font-weight: 600; }
  .news-card .news-meta { color: #6b7280; font-size: 12px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

INPUT_LEN = 60
PRED_LENS = {"1 ngày (1 phiên)": 1, "1 tuần (5 phiên)": 5, "1 tháng (21 phiên)": 21}

BANK_TICKERS = [
    "VCB","BID","CTG","TCB","MBB","ACB","VPB","STB","HDB","TPB",
    "VIB","MSB","OCB","SHB","LPB","EIB","NAB","PGB","ABB","BVB"
]
BANK_NAMES = {
    "VCB":"Vietcombank","BID":"BIDV","CTG":"VietinBank","TCB":"Techcombank",
    "MBB":"MB Bank","ACB":"ACB","VPB":"VPBank","STB":"Sacombank",
    "HDB":"HDBank","TPB":"TPBank","VIB":"VIB","MSB":"MSB","OCB":"OCB",
    "SHB":"SHB","LPB":"LPBank","EIB":"Eximbank","NAB":"Nam A Bank",
    "PGB":"PG Bank","ABB":"ABBank","BVB":"Bao Viet Bank"
}

RSS_FEEDS = [
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://cafef.vn/ngan-hang.rss",
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss",
    "https://vietstock.vn/820/chung-khoan.rss",
]
ALIAS_MAP = {
    "VCB":["vietcombank"],"BID":["bidv"],"CTG":["vietinbank"],"TCB":["techcombank"],
    "MBB":["mb bank","mbbank"],"VPB":["vpbank"],"STB":["sacombank"],"HDB":["hdbank"],
    "TPB":["tpbank"],"SHB":["shb"],"LPB":["lpbank","lienvietpostbank"],
    "EIB":["eximbank"],"ACB":["acb"],"VIB":["vib"],"MSB":["msb"],"OCB":["ocb"],
    "NAB":["nam a bank"],"PGB":["pg bank"],"ABB":["abbank"],"BVB":["bao viet bank"],
}

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_data():
    path = "data/processed/clean_data.csv"
    if not os.path.exists(path):
        st.error("⚠️ Chưa có dữ liệu. Hãy chạy pipeline từ src/ trước.")
        st.stop()
    return pd.read_csv(path, parse_dates=["time"])


def load_scaler(ticker):
    path = f"data/models/{ticker}_scaler.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_cached(ticker, model_name, pred_len):
    ModelClass = MODEL_REGISTRY[model_name]
    model      = ModelClass(INPUT_LEN, pred_len)
    path       = f"data/models/{ticker}_{model_name}_pred{pred_len}.pt"
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(ticker, pred_len, prices, scaler):
    """Dự báo tương lai — dùng scaler đã fit trên train."""
    if len(prices) < INPUT_LEN:
        return {}
    scaled  = scaler.transform(prices.reshape(-1, 1)).flatten()
    x_input = torch.tensor(scaled[-INPUT_LEN:], dtype=torch.float32).unsqueeze(0)
    results = {}
    for name in MODEL_REGISTRY:
        m = load_model_cached(ticker, name, pred_len)
        if m is None:
            continue
        with torch.no_grad():
            out = m(x_input).numpy().flatten()
        results[name] = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
    return results


def backtest(ticker, pred_len, prices, scaler):
    """
    Trượt cửa sổ step=1 trên tập test (20% cuối).
    Lấy bước đầu tiên của mỗi dự đoán → đường dự báo liên tục.
    Scaler đã fit chỉ trên train nên không bị leakage.
    """
    # Fit scaler chỉ trên 80% đầu (khớp với build_model)
    price_split   = int(len(prices) * 0.8)
    prices_train  = prices[:price_split]
    prices_test   = prices[price_split:]

    scaled_train = scaler.transform(prices_train.reshape(-1, 1)).flatten()
    scaled_test  = scaler.transform(prices_test.reshape(-1, 1)).flatten()

    # Ghép 60 ngày cuối train làm context cho test sequences
    test_series = np.concatenate([scaled_train[-INPUT_LEN:], scaled_test])

    results = {name: [] for name in MODEL_REGISTRY}
    for i in range(len(scaled_test)):
        if i + pred_len > len(scaled_test):
            break
        x_in = torch.tensor(test_series[i: i + INPUT_LEN], dtype=torch.float32).unsqueeze(0)
        for name in MODEL_REGISTRY:
            m = load_model_cached(ticker, name, pred_len)
            if m is None:
                continue
            with torch.no_grad():
                out = m(x_in).numpy().flatten()
            inv = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
            results[name].append(float(inv[0]))

    return {k: np.array(v) for k, v in results.items() if v}


def _parse_date(entry):
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.today().strftime("%Y-%m-%d")


@st.cache_data(ttl=1800)
def fetch_news(ticker, max_results=10):
    import re
    bank_name = BANK_NAMES.get(ticker, ticker)
    keywords  = [ticker.lower(), bank_name.lower()] + ALIAS_MAP.get(ticker, [])
    collected, seen_urls = [], set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                if not any(kw in (title + summary).lower() for kw in keywords):
                    continue
                url = entry.get("link", "#")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                image_url = ""
                if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                    image_url = entry.media_thumbnail[0].get("url", "")
                if not image_url and summary:
                    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary)
                    if m:
                        image_url = m.group(1)

                clean_summary = re.sub(r"<[^>]+>", "", summary).strip()
                clean_summary = clean_summary[:200] + "..." if len(clean_summary) > 200 else clean_summary

                collected.append({
                    "title": title, "url": url,
                    "source": feed.feed.get("title", ""),
                    "published": _parse_date(entry),
                    "summary": clean_summary, "image": image_url,
                })
                if len(collected) >= max_results:
                    break
        except Exception:
            continue
        if len(collected) >= max_results:
            break

    collected.sort(key=lambda x: x["published"], reverse=True)

    if not collected:
        return [
            {"title": f"[Mẫu] Cập nhật kết quả kinh doanh {bank_name} quý gần nhất",
             "url": "#", "source": "CafeF",
             "published": datetime.today().strftime("%Y-%m-%d"),
             "summary": "Thông tin kết quả kinh doanh mới nhất.", "image": ""},
        ]
    return collected


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 VN Bank Forecast")
    st.markdown("---")
    page = st.radio(
        "Chọn trang",
        ["📊 Tổng quan", "🔍 Ngân hàng", "📰 Tin tức"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dữ liệu: vnstock | Mô hình: LTSF-Linear")
    st.caption(f"Cập nhật: {datetime.today().strftime('%d/%m/%Y')}")

df = load_data()

# ════════════════════════════════════════════════════════════
# TRANG 1: TỔNG QUAN
# ════════════════════════════════════════════════════════════
if page == "📊 Tổng quan":
    st.title("📊 Tổng quan thị trường cổ phiếu ngân hàng")

    latest = (df.sort_values("time").groupby("ticker").last()
                .reset_index()[["ticker", "close"]])
    latest["bank_name"] = latest["ticker"].map(BANK_NAMES)

    st.subheader("💹 Giá đóng cửa hiện tại — 20 Ngân hàng")
    latest_sorted = latest.sort_values("close", ascending=False)
    fig_bar = go.Figure(go.Bar(
        x=latest_sorted["ticker"], y=latest_sorted["close"],
        marker=dict(color=latest_sorted["close"], colorscale="Blues", showscale=False),
        text=latest_sorted["close"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside"
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0", height=400,
        margin=dict(t=20, b=20), xaxis_tickangle=-45
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Biến động giá hôm nay
    latest_2days = df.sort_values("time").groupby("ticker").tail(2).copy()
    latest_2days["daily_return"] = latest_2days.groupby("ticker")["close"].pct_change()
    latest_change = latest_2days.groupby("ticker").last().reset_index()
    latest_sorted2 = latest_change.sort_values("daily_return", ascending=False)

    st.subheader("📅 Biến động giá hôm nay (%)")
    colors_ret = ["#34d399" if v >= 0 else "#f87171" for v in latest_sorted2["daily_return"]]
    fig_ret = go.Figure(go.Bar(
        x=latest_sorted2["ticker"],
        y=latest_sorted2["daily_return"] * 100,
        marker_color=colors_ret,
        text=(latest_sorted2["daily_return"] * 100).apply(lambda x: f"{x:+.2f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Thay đổi: %{y:+.2f}%<extra></extra>"
    ))
    fig_ret.add_hline(y=0, line_color="#475569", line_width=1)
    fig_ret.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0", height=320,
        margin=dict(t=20, b=20), xaxis_tickangle=-20
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # Mạng lưới tương quan
    st.subheader("🔗 Mạng lưới tương quan giá đóng cửa")
    import networkx as nx

    pivot = df.pivot_table(index="time", columns="ticker", values="close")
    corr  = pivot.corr()
    tickers_corr = list(corr.columns)
    THRESHOLD = 0.5

    G = nx.Graph()
    G.add_nodes_from(tickers_corr)
    for i, t1 in enumerate(tickers_corr):
        for j, t2 in enumerate(tickers_corr):
            if i < j and abs(corr.loc[t1, t2]) >= THRESHOLD:
                G.add_edge(t1, t2, weight=float(corr.loc[t1, t2]))

    pos      = nx.spring_layout(G, seed=42, k=5.0, iterations=120)
    avg_corr = corr.abs().mean()
    min_s, max_s = avg_corr.min(), avg_corr.max()
    node_sizes = {
        t: 20 + ((avg_corr[t] - min_s) / (max_s - min_s + 1e-9)) * 50
        for t in tickers_corr
    }

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        val    = data["weight"]
        alpha  = min(abs(val) * 0.8, 0.6)
        color  = f"rgba(30,136,229,{alpha})" if val > 0 else f"rgba(216,90,48,{alpha})"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=0.5 + abs(val) * 2.5, color=color),
            hoverinfo="none", showlegend=False,
        ))

    node_trace = go.Scatter(
        x=[pos[t][0] for t in tickers_corr],
        y=[pos[t][1] for t in tickers_corr],
        mode="markers+text",
        marker=dict(
            size=[node_sizes[t] for t in tickers_corr],
            color=[avg_corr[t] for t in tickers_corr],
            colorscale=[[0,"#c8dfe0"],[0.4,"#5DCAA5"],[0.75,"#1D9E75"],[1,"#085041"]],
            cmin=float(min_s), cmax=float(max_s), showscale=True,
            colorbar=dict(title=dict(text="Tương quan TB", font=dict(size=11, color="#94a3b8")),
                          thickness=12, tickfont=dict(size=10, color="#94a3b8"),
                          outlinewidth=0, x=1.02),
            line=dict(width=2, color="rgba(255,255,255,0.25)"), sizemode="diameter",
        ),
        text=tickers_corr, textposition="middle center",
        textfont=dict(size=9, color="#ffffff", family="Arial Black"),
        hovertext=[
            f"<b>{t} — {BANK_NAMES.get(t,t)}</b><br>"
            f"Tương quan TB: {avg_corr[t]:.2f}<br>Số kết nối: {G.degree(t)}"
            for t in tickers_corr
        ],
        hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
    )

    fig_corr = go.Figure(data=edge_traces + [node_trace])
    fig_corr.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=650, margin=dict(t=30, b=30, l=30, r=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,22,41,0.8)",
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                    font=dict(size=11, color="#94a3b8"))
    )
    fig_corr.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(30,136,229,0.7)"),
        name="Tương quan dương", showlegend=True))
    fig_corr.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(216,90,48,0.7)"),
        name="Tương quan âm", showlegend=True))

    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        f"🔵 Kích thước node = mức độ liên kết TB  |  "
        f"🟢 Xanh = tương quan dương  |  🔴 Đỏ = tương quan âm  |  "
        f"Ngưỡng: |corr| ≥ {THRESHOLD}"
    )


# ════════════════════════════════════════════════════════════
# TRANG 2: NGÂN HÀNG
# ════════════════════════════════════════════════════════════
elif page == "🔍 Ngân hàng":
    ticker = st.selectbox(
        "Chọn ngân hàng", BANK_TICKERS,
        format_func=lambda t: f"{t} — {BANK_NAMES.get(t, t)}"
    )
    sub = df[df["ticker"] == ticker].sort_values("time").copy()
    st.markdown(f"## 🏦 {BANK_NAMES.get(ticker, ticker)} ({ticker})")

    tab1, tab2, tab3 = st.tabs(["📈 Phân tích dữ liệu", "🔮 Dự báo", "📊 Đánh giá mô hình"])

    # ── Tab 1: Phân tích dữ liệu ────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Stock Price Over Time")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=sub["time"], y=sub["close"],
                mode="lines", name="Giá đóng cửa",
                line=dict(color="#38bdf8", width=1.5)))
            fig1.add_trace(go.Scatter(x=sub["time"], y=sub["ma21"],
                mode="lines", name="MA 21",
                line=dict(color="#f59e0b", width=1, dash="dash")))
            fig1.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320,
                margin=dict(t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.subheader("Log-transformed Price")
            fig2 = go.Figure(go.Scatter(x=sub["time"], y=sub["close_log"],
                mode="lines", line=dict(color="#a78bfa", width=1.5)))
            fig2.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Daily Returns Over Time")
            colors_bar = np.where(sub["daily_return"] >= 0, "#34d399", "#f87171")
            fig3 = go.Figure(go.Bar(x=sub["time"], y=sub["daily_return"],
                marker_color=colors_bar))
            fig3.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.subheader("Volume Over Time")
            fig4 = go.Figure(go.Bar(x=sub["time"], y=sub["volume"],
                marker_color="#60a5fa"))
            fig4.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2: Dự báo ────────────────────────────────────────
    with tab2:
        scaler = load_scaler(ticker)
        if scaler is None:
            st.warning("⚠️ Chưa có mô hình đã huấn luyện cho ngân hàng này.")
            st.stop()

        prices       = sub["close"].values
        prices_dates = sub["time"].values
        colors_model = {"Linear": "#38bdf8", "DLinear": "#f59e0b", "NLinear": "#34d399"}

        for label, pred_len in PRED_LENS.items():
            st.subheader(f"📅 Dự báo {label}")
            preds = predict(ticker, pred_len, prices, scaler)
            if not preds:
                st.info("Chưa có mô hình, hãy chạy src/build_model.py trước.")
                continue

            bt_preds  = backtest(ticker, pred_len, prices, scaler)
            price_split = int(len(prices) * 0.8)
            bt_dates    = prices_dates[price_split: price_split + len(next(iter(bt_preds.values())))]

            last_date   = sub["time"].max()
            future_days = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)

            fig = go.Figure()

            # Giá thực tế
            fig.add_trace(go.Scatter(
                x=sub["time"], y=sub["close"],
                mode="lines", name="Giá thực tế",
                line=dict(color="#94a3b8", width=1.5)
            ))

            # Đường backtest từng mô hình
            for name, vals in bt_preds.items():
                n = min(len(vals), len(bt_dates))
                if n == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=bt_dates[:n], y=vals[:n],
                    mode="lines", name=name,
                    line=dict(color=colors_model.get(name, "#fff"), width=1.5, dash="dot"),
                ))

            # Đường kẻ dọc hôm nay
            fig.add_vline(
                x=pd.Timestamp(last_date).timestamp() * 1000,
                line_dash="dash", line_color="#64748b", line_width=1.5,
                annotation_text="Hôm nay", annotation_position="top right",
                annotation_font_color="#94a3b8"
            )

            # Dự báo tương lai
            for name, vals in preds.items():
                connect_x = [last_date] + list(future_days)
                connect_y = [prices[-1]] + list(vals)
                fig.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    mode="lines+markers", name=f"{name} →",
                    line=dict(color=colors_model.get(name, "#fff"), width=2.5, dash="dash"),
                    marker=dict(size=6), showlegend=False
                ))

            fig.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=450,
                margin=dict(t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            table_data = {"Ngày": future_days.strftime("%d/%m/%Y")}
            for name, vals in preds.items():
                table_data[name] = [f"{v:,.0f}" for v in vals]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Tab 3: Đánh giá mô hình ─────────────────────────────
    with tab3:
        EVAL_PATH   = "data/model_evaluation.csv"
        DM_PATH     = "data/dm_test_results.csv"
        REGIME_PATH = "data/regime_mae.csv"
        WF_PATH     = "data/walkforward_cv.csv"

        if not os.path.exists(EVAL_PATH):
            st.warning("⚠️ Chưa có file đánh giá. Hãy chạy `src/evaluate_model.py` trước.")
            st.code("python src/evaluate_model.py", language="bash")
        else:
            eval_df   = pd.read_csv(EVAL_PATH)
            df_ticker = eval_df[eval_df["ticker"] == ticker].copy()

            if df_ticker.empty:
                st.info(f"Không có dữ liệu đánh giá cho {ticker}.")
            else:
                st.markdown("### 📊 Kết quả đánh giá")

                colors_model = {"Linear": "#38bdf8", "DLinear": "#f59e0b",
                                "NLinear": "#34d399", "Naive": "#f87171"}
                HORIZON_LABELS = {
                    1:  "1 ngày (1 phiên)",
                    5:  "1 tuần (5 phiên)",
                    21: "1 tháng (21 phiên)"
                }

                for pred_len, label in HORIZON_LABELS.items():
                    df_h = df_ticker[df_ticker["pred_len"] == pred_len].copy()
                    if df_h.empty:
                        continue

                    st.markdown(f"#### 📅 {label}")

                    # ── Bảng metrics đầy đủ ───────────────────────────
                    metric_cols = ["model", "RMSE", "MAE", "R2", "VarianceRatio"]
                    available   = [c for c in metric_cols if c in df_h.columns]
                    df_display  = df_h[available].copy()
                    df_display.columns = [c.replace("model", "Mô hình")
                                            .replace("VarianceRatio", "VR") for c in available]
                    best_name   = df_h.loc[df_h["RMSE"].idxmin(), "model"]

                    def highlight_best(row):
                        style = "background-color:#1e3a2f;color:#34d399;font-weight:bold"
                        return [style if row["Mô hình"] == best_name else "" for _ in row]

                    fmt = {}
                    for c in df_display.columns:
                        if c in ("RMSE", "MAE"):
                            fmt[c] = "{:,.0f}"
                        elif c in ("R2", "VR"):
                            fmt[c] = "{:.4f}"

                    styled = df_display.style.apply(highlight_best, axis=1).format(fmt)
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Cảnh báo over-smoothing
                    vr_col = "VarianceRatio"
                    if vr_col in df_h.columns:
                        low_vr = df_h[df_h[vr_col] < 0.4]
                        if not low_vr.empty:
                            names = ", ".join(low_vr["model"].tolist())
                            st.warning(
                                f"⚠️ **Over-smoothing detected** — {names}: "
                                f"Variance Ratio < 0.4. Mô hình dự báo quá phẳng, "
                                f"đặc biệt trong giai đoạn biến động cao (crisis)."
                            )

                    # ── MAE-by-step plot ───────────────────────────────
                    mae_img = f"data/eval_plots/{ticker}_mae_step_pred{pred_len}.png"
                    if os.path.exists(mae_img):
                        st.image(mae_img,
                                 caption=f"MAE theo từng bước — horizon {pred_len} ngày")

                    # ── Walk-forward CV ────────────────────────────────
                    wf_img = f"data/eval_plots/{ticker}_wf_pred{pred_len}.png"
                    if os.path.exists(wf_img):
                        st.image(wf_img,
                                 caption=f"Walk-forward CV (4 folds) — horizon {pred_len} ngày")
                    elif os.path.exists(WF_PATH):
                        wf_df = pd.read_csv(WF_PATH)
                        wf_t  = wf_df[(wf_df["ticker"] == ticker) &
                                      (wf_df["pred_len"] == pred_len)]
                        if not wf_t.empty:
                            st.markdown("**Walk-forward CV — MAE theo fold**")
                            pivot = wf_t.pivot(index="fold", columns="model", values="MAE")
                            fig_wf = go.Figure()
                            for col in pivot.columns:
                                fig_wf.add_trace(go.Scatter(
                                    x=pivot.index, y=pivot[col],
                                    mode="lines+markers", name=col,
                                    line=dict(color=colors_model.get(col, "#94a3b8"))
                                ))
                            fig_wf.update_layout(
                                paper_bgcolor="#111827", plot_bgcolor="#111827",
                                font_color="#e2e8f0", height=280,
                                margin=dict(t=10, b=10),
                                xaxis_title="Fold", yaxis_title="MAE",
                                legend=dict(orientation="h")
                            )
                            st.plotly_chart(fig_wf, use_container_width=True)

                    # ── Conditional MAE per Regime ─────────────────────
                    if os.path.exists(REGIME_PATH):
                        regime_df = pd.read_csv(REGIME_PATH)
                        reg_t = regime_df[(regime_df["ticker"] == ticker) &
                                          (regime_df["pred_len"] == pred_len)]
                        if not reg_t.empty:
                            st.markdown("**MAE theo Regime thị trường**")
                            st.caption(
                                "Regime phân loại theo volatility 21 ngày: "
                                "🟢 Low (< p33) | 🟡 Normal (p33–p67) | 🔴 Crisis (> p67)"
                            )
                            reg_display = reg_t[["model", "low", "normal", "crisis"]].copy()
                            reg_display.columns = ["Mô hình", "Low vol", "Normal", "Crisis"]

                            def style_regime(val):
                                if pd.isna(val) or not isinstance(val, (int, float)):
                                    return ""
                                col_max = reg_t[["low","normal","crisis"]].max().max()
                                intensity = min(int((val / (col_max + 1e-8)) * 180), 180)
                                return f"background-color: rgba(248,113,113,{intensity/255:.2f})"

                            reg_styled = (reg_display.style
                                .applymap(style_regime, subset=["Low vol", "Normal", "Crisis"])
                                .format({"Low vol": "{:,.0f}", "Normal": "{:,.0f}", "Crisis": "{:,.0f}"}))
                            st.dataframe(reg_styled, use_container_width=True, hide_index=True)

                            regime_img = f"data/eval_plots/{ticker}_regime_pred{pred_len}.png"
                            if os.path.exists(regime_img):
                                st.image(regime_img,
                                         caption=f"MAE theo Regime — horizon {pred_len} ngày")

                    # ── Diebold-Mariano Test ───────────────────────────
                    if os.path.exists(DM_PATH):
                        dm_df = pd.read_csv(DM_PATH)
                        dm_t  = dm_df[(dm_df["ticker"] == ticker) &
                                      (dm_df["pred_len"] == pred_len)]
                        if not dm_t.empty:
                            st.markdown("**Diebold-Mariano Test — Best model vs Naive**")
                            st.caption(
                                "H₀: best model và Naive có độ chính xác như nhau. "
                                "p < 0.05 → best model tốt hơn Naive có ý nghĩa thống kê. ✅"
                            )
                            dm_display = dm_t[["best_model","DM_stat","p_value",
                                               "significant","Naive_MAE","Best_MAE"]].copy()
                            dm_display.columns = ["Best model","DM stat","p-value",
                                                  "Significant","MAE Naive","MAE Best"]
                            st.dataframe(dm_display, use_container_width=True, hide_index=True)

                    st.markdown("---")

                # ── Limitations (honest) ───────────────────────────────
                with st.expander("⚠️ Limitations — Over-smoothing & Model Constraints"):
                    st.markdown("""
**1. Over-smoothing trong giai đoạn biến động cao (Crisis)**
Các mô hình LTSF-Linear có xu hướng dự báo "phẳng" (flat prediction) trong giai đoạn
khủng hoảng — khi giá biến động mạnh và đột ngột. Variance Ratio thường < 0.4 trong
regime *crisis*, cho thấy mô hình không tái tạo được biên độ dao động thực tế.
Điều này là hạn chế cố hữu của kiến trúc tuyến tính, không phải lỗi triển khai.

**2. Ý nghĩa thống kê (DM test)**
Một số horizon (đặc biệt pred_len=21) có thể không vượt qua DM test (p ≥ 0.05),
tức là mô hình không tốt hơn Naive một cách có ý nghĩa thống kê ở tầm xa.
Người dùng nên ưu tiên dùng mô hình ở horizon ngắn (pred_len=1 hoặc 5).

**3. Walk-forward CV vs Holdout**
Kết quả holdout 80/20 thường lạc quan hơn walk-forward CV do thị trường
có regime shift theo thời gian. Nếu MAE walk-forward tăng dần qua các fold,
mô hình có dấu hiệu kém ổn định theo thời gian.

**4. Dữ liệu đơn chiều**
Mô hình chỉ dùng chuỗi giá đóng cửa. Các yếu tố vĩ mô, tin tức, khối lượng
giao dịch chưa được đưa vào, làm giảm khả năng dự báo trong các sự kiện
bất thường (earnings surprise, thay đổi chính sách tiền tệ đột ngột).
                    """)


# ════════════════════════════════════════════════════════════
# TRANG 3: TIN TỨC
# ════════════════════════════════════════════════════════════
elif page == "📰 Tin tức":
    st.title("📰 Tin tức ngân hàng")
    st.caption("Nguồn: CafeF · VnExpress · Vietstock (RSS) — cập nhật mỗi 30 phút")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        ticker_news = st.selectbox(
            "Chọn ngân hàng", BANK_TICKERS,
            format_func=lambda t: f"{t} — {BANK_NAMES.get(t, t)}",
            key="news_ticker"
        )
    with col_right:
        if st.button("🔄 Làm mới"):
            st.cache_data.clear()
            st.rerun()

    st.subheader(f"🏦 {BANK_NAMES.get(ticker_news, ticker_news)} ({ticker_news})")

    with st.spinner("Đang tải tin tức..."):
        news_list = fetch_news(ticker_news)

    st.caption(f"Tìm thấy {len(news_list)} bài viết")
    for item in news_list:
        summary_html = (
            f"<div style='color:#94a3b8;font-size:13px;margin-top:6px'>{item['summary']}</div>"
            if item.get("summary") else ""
        )
        st.markdown(f"""
        <div class="news-card">
            <a href="{item['url']}" target="_blank">{item['title']}</a>
            {summary_html}
            <div class="news-meta">📅 {item['published']} &nbsp;|&nbsp; 📰 {item['source']}</div>
        </div>
        """, unsafe_allow_html=True)