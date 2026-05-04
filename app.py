# ============================================================
# ỨNG DỤNG STREAMLIT - Hỗ trợ quyết định đầu tư cổ phiếu ngân hàng
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch, pickle, os, sys
import feedparser
from datetime import datetime, timedelta

sys.path.append("src")
from models import MODEL_REGISTRY, LinearModel, DLinearModel, NLinearModel

# ── Cấu hình trang ──────────────────────────────────────────
st.set_page_config(
    page_title="VN Bank Stock Forecast",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS tùy chỉnh ───────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #e2e8f0; }
  [data-testid="stSidebar"]          { background: #0f1629; }
  .metric-card {
      background: linear-gradient(135deg, #1a2035, #1e2a45);
      border: 1px solid #2d3f6e;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
  }
  .metric-card .label { color: #94a3b8; font-size: 13px; margin-bottom: 6px; }
  .metric-card .value { color: #38bdf8; font-size: 26px; font-weight: 700; }
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

# ── Hàm tiện ích ────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    path = "data/processed/clean_data.csv"
    if not os.path.exists(path):
        st.error("⚠️ Chưa có dữ liệu. Hãy chạy pipeline từ src/ trước.")
        st.stop()
    df = pd.read_csv(path, parse_dates=["time"])
    return df


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
    """Chạy cả 3 mô hình, trả về dict {model_name: array giá dự đoán}."""
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


def load_scaler(ticker):
    path = f"data/models/{ticker}_scaler.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── RSS Feeds tài chính Việt Nam (miễn phí, không cần API key) ──
RSS_FEEDS = [
    "https://cafef.vn/thi-truong-chung-khoan.rss",   # CafeF – chứng khoán
    "https://cafef.vn/ngan-hang.rss",                 # CafeF – ngân hàng
    "https://vnexpress.net/rss/kinh-doanh.rss",       # VnExpress – kinh doanh
    "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss",  # VnExpress – chứng khoán
    "https://vietstock.vn/820/chung-khoan.rss",       # Vietstock
]

ALIAS_MAP = {
    "VCB": ["vietcombank"],
    "BID": ["bidv"],
    "CTG": ["vietinbank"],
    "TCB": ["techcombank"],
    "MBB": ["mb bank", "mbbank"],
    "VPB": ["vpbank"],
    "STB": ["sacombank"],
    "HDB": ["hdbank"],
    "TPB": ["tpbank"],
    "SHB": ["shb"],
    "LPB": ["lpbank", "lienvietpostbank"],
    "EIB": ["eximbank"],
    "ACB": ["acb"],
    "VIB": ["vib"],
    "MSB": ["msb"],
    "OCB": ["ocb"],
    "NAB": ["nam a bank"],
    "PGB": ["pg bank"],
    "ABB": ["abbank"],
    "BVB": ["bao viet bank"],
}

def _parse_date(entry) -> str:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.today().strftime("%Y-%m-%d")

@st.cache_data(ttl=1800)
def fetch_news(ticker, max_results: int = 10):
    bank_name = BANK_NAMES.get(ticker, ticker)
    keywords  = [ticker.lower(), bank_name.lower()] + ALIAS_MAP.get(ticker, [])

    collected = []
    seen_urls = set()

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                text    = (title + " " + summary).lower()

                if not any(kw in text for kw in keywords):
                    continue

                url = entry.get("link", "#")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # ── Lấy ảnh thumbnail ──────────────────────────
                image_url = ""

                # Cách 1: media_thumbnail (phổ biến nhất)
                if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                    image_url = entry.media_thumbnail[0].get("url", "")

                # Cách 2: media_content
                if not image_url and hasattr(entry, "media_content") and entry.media_content:
                    image_url = entry.media_content[0].get("url", "")

                # Cách 3: enclosures (podcast/image RSS)
                if not image_url and hasattr(entry, "enclosures") and entry.enclosures:
                    for enc in entry.enclosures:
                        if "image" in enc.get("type", ""):
                            image_url = enc.get("url", "")
                            break

                # Cách 4: parse thẻ <img> trong summary HTML
                if not image_url and summary:
                    import re
                    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary)
                    if match:
                        image_url = match.group(1)

                # ── Clean summary (bỏ HTML tags) ───────────────
                import re
                clean_summary = re.sub(r"<[^>]+>", "", summary).strip()
                clean_summary = clean_summary[:200] + "..." if len(clean_summary) > 200 else clean_summary

                collected.append({
                    "title":     title,
                    "url":       url,
                    "source":    feed.feed.get("title", ""),
                    "published": _parse_date(entry),
                    "summary":   clean_summary,
                    "image":     image_url,
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
            {
                "title":     f"[Mẫu] Cập nhật kết quả kinh doanh {bank_name} Quý gần nhất",
                "url":       "#", "source": "CafeF",
                "published": datetime.today().strftime("%Y-%m-%d"),
                "summary":   "Thông tin kết quả kinh doanh mới nhất của ngân hàng.",
                "image":     "",
            },
            {
                "title":     f"[Mẫu] {bank_name} công bố kế hoạch tăng vốn điều lệ",
                "url":       "#", "source": "VnExpress",
                "published": datetime.today().strftime("%Y-%m-%d"),
                "summary":   "Kế hoạch tăng vốn điều lệ trong năm tới của ngân hàng.",
                "image":     "",
            },
        ]

    return collected
# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 VN Bank Forecast")
    st.markdown("---")
    page = st.radio(
        "Chọn trang",
        ["📊 Tổng quan", "🔍 Ngân hàng", "📰 Tin tức"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dữ liệu: vnstock | Mô hình: LTSF")
    st.caption(f"Cập nhật: {datetime.today().strftime('%d/%m/%Y')}")

df = load_data()

# ════════════════════════════════════════════════════════════
# TRANG 1: TỔNG QUAN
# ════════════════════════════════════════════════════════════
if page == "📊 Tổng quan":
    st.title("📊 Tổng quan thị trường cổ phiếu ngân hàng")

    latest = (df.sort_values("time")
                .groupby("ticker")
                .last()
                .reset_index()[["ticker", "close"]])
    latest["bank_name"] = latest["ticker"].map(BANK_NAMES)

    highest = latest.loc[latest["close"].idxmax()]
    lowest  = latest.loc[latest["close"].idxmin()]

    st.markdown("<br>", unsafe_allow_html=True)

    # Biểu đồ bar giá hiện tại
    st.subheader("💹 Giá đóng cửa hiện tại - 20 Ngân hàng")
    latest_sorted = latest.sort_values("close", ascending=False)
    fig_bar = go.Figure(go.Bar(
        x=latest_sorted["ticker"],
        y=latest_sorted["close"],
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
    latest_2days = df.sort_values("time").groupby("ticker").tail(2)
    latest_2days = latest_2days.copy()
    latest_2days["daily_return"] = latest_2days.groupby("ticker")["close"].pct_change()
    latest_change = latest_2days.groupby("ticker").last().reset_index()
    latest_change["bank_name"] = latest_change["ticker"].map(BANK_NAMES)
    latest_sorted2 = latest_change.sort_values("daily_return", ascending=False)

    st.markdown('<div class="section-header">📅 Biến động giá hôm nay (%)</div>', unsafe_allow_html=True)
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

    # Ma trận tương quan - Citation Network style
    st.subheader("🔗 Mạng lưới tương quan giá đóng cửa")

    import networkx as nx

    pivot = df.pivot_table(index="time", columns="ticker", values="close")
    corr  = pivot.corr()
    tickers_corr = list(corr.columns)

    # ── Tạo graph ──────────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(tickers_corr)

    THRESHOLD = 0.5  # Chỉ vẽ cạnh khi tương quan >= 0.5
    for i, t1 in enumerate(tickers_corr):
        for j, t2 in enumerate(tickers_corr):
            if i < j:
                val = corr.loc[t1, t2]
                if abs(val) >= THRESHOLD:
                    G.add_edge(t1, t2, weight=float(val))

    # ── Force-directed layout ───────────────────────────────────
    pos = nx.spring_layout(G, seed=42, k=5.0, iterations=120)

    # ── Kích thước node = trung bình tương quan tuyệt đối ──────
    avg_corr = corr.abs().mean()

    # Scale node size: min 20px, max 70px
    min_s, max_s = avg_corr.min(), avg_corr.max()
    node_sizes = {
        t: 20 + ((avg_corr[t] - min_s) / (max_s - min_s + 1e-9)) * 50
        for t in tickers_corr
    }

    # ── Vẽ edges theo nhóm màu ─────────────────────────────────
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        val     = data["weight"]
        alpha   = min(abs(val) * 0.8, 0.6)
        color   = f"rgba(30,136,229,{alpha})" if val > 0 else f"rgba(216,90,48,{alpha})"
        width   = 0.5 + abs(val) * 2.5

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Vẽ nodes ───────────────────────────────────────────────
    node_x      = [pos[t][0] for t in tickers_corr]
    node_y      = [pos[t][1] for t in tickers_corr]
    node_size   = [node_sizes[t] for t in tickers_corr]
    node_color  = [avg_corr[t] for t in tickers_corr]
    node_hover  = [
        f"<b>{t} — {BANK_NAMES.get(t, t)}</b><br>"
        f"Tương quan TB: {avg_corr[t]:.2f}<br>"
        f"Số kết nối: {G.degree(t)}"
        for t in tickers_corr
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=[
                [0.0, "#c8dfe0"],
                [0.4, "#5DCAA5"],
                [0.75, "#1D9E75"],
                [1.0, "#085041"],
            ],
            cmin=float(min_s),
            cmax=float(max_s),
            showscale=True,
            colorbar=dict(
                title=dict(text="Tương quan TB", font=dict(size=11, color="#94a3b8")),
                thickness=12,
                tickfont=dict(size=10, color="#94a3b8"),
                outlinewidth=0,
                x=1.02,
            ),
            line=dict(width=2, color="rgba(255,255,255,0.25)"),
            sizemode="diameter",
        ),
        text=tickers_corr,
        textposition="middle center",
        textfont=dict(size=9, color="#ffffff", family="Arial Black"),
        hovertext=node_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    )

    # ── Highlight top-3 node tương quan cao nhất ───────────────
    top3 = avg_corr.nlargest(3).index.tolist()
    highlight_trace = go.Scatter(
        x=[pos[t][0] for t in top3],
        y=[pos[t][1] for t in top3],
        mode="markers",
        marker=dict(
            size=[node_sizes[t] + 10 for t in top3],
            color="rgba(0,0,0,0)",
            line=dict(width=2.5, color="rgba(255,255,255,0.7)"),
            sizemode="diameter",
        ),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Annotation: tên node lớn nhất ──────────────────────────
    top1 = avg_corr.idxmax()
    annotations = [dict(
        x=pos[top1][0], y=pos[top1][1] + 0.12,
        text=f"<b>{top1}</b>",
        showarrow=False,
        font=dict(size=11, color="#38bdf8"),
        bgcolor="rgba(10,14,26,0.6)",
        borderpad=3,
    )]

    fig_corr = go.Figure(data=edge_traces + [node_trace, highlight_trace])
    fig_corr.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0",
        height=650,
        margin=dict(t=30, b=30, l=30, r=60),
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        hovermode="closest",
    )

    # Legend thủ công
    fig_corr.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10,  color="rgba(30,136,229,0.7)"),
        name="Tương quan dương", showlegend=True
    ))
    fig_corr.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(216,90,48,0.7)"),
        name="Tương quan âm", showlegend=True
    ))
    fig_corr.update_layout(
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(15,22,41,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11, color="#94a3b8"),
        )
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Caption giải thích
    st.caption(
        "🔵 Kích thước node = mức độ liên kết trung bình  |  "
        "🟢 Cạnh xanh = tương quan dương  |  🔴 Cạnh đỏ = tương quan âm  |  "
        f"Ngưỡng hiển thị: |corr| ≥ {THRESHOLD}"
    )


# ════════════════════════════════════════════════════════════
# TRANG 2: Ngân hàng
# ════════════════════════════════════════════════════════════
elif page == "🔍 Ngân hàng":
    ticker = st.selectbox(
        "Chọn ngân hàng",
        BANK_TICKERS,
        format_func=lambda t: f"{t} — {BANK_NAMES.get(t, t)}"
    )

    sub = df[df["ticker"] == ticker].sort_values("time").copy()
    st.markdown(f"## 🏦 {BANK_NAMES.get(ticker, ticker)} ({ticker})")

    tab1, tab2, tab3 = st.tabs(["📈 Phân tích dữ liệu", "🔮 Dự báo", "📊 Đánh giá mô hình"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Stock Price Over Time")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=sub["time"], y=sub["close"],
                mode="lines", name="Giá đóng cửa",
                line=dict(color="#38bdf8", width=1.5)
            ))
            fig1.add_trace(go.Scatter(
                x=sub["time"], y=sub["ma21"],
                mode="lines", name="MA 21",
                line=dict(color="#f59e0b", width=1, dash="dash")
            ))
            fig1.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320,
                margin=dict(t=10, b=10), legend=dict(orientation="h")
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.subheader("Log-transformed Price")
            fig2 = go.Figure(go.Scatter(
                x=sub["time"], y=sub["close_log"],
                mode="lines", line=dict(color="#a78bfa", width=1.5)
            ))
            fig2.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Daily Returns Over Time")
            colors = np.where(sub["daily_return"] >= 0, "#34d399", "#f87171")
            fig3 = go.Figure(go.Bar(
                x=sub["time"], y=sub["daily_return"],
                marker_color=colors
            ))
            fig3.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            st.subheader("Volume Over Time")
            fig4 = go.Figure(go.Bar(
                x=sub["time"], y=sub["volume"],
                marker_color="#60a5fa"
            ))
            fig4.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", height=320, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig4, use_container_width=True)

        with tab2:
            scaler = load_scaler(ticker)
            if scaler is None:
                st.warning("⚠️ Chưa có mô hình đã huấn luyện cho ngân hàng này.")
                st.stop()

            prices       = sub["close"].values
            prices_dates = sub["time"].values
            colors_model = {"Linear": "#38bdf8", "DLinear": "#f59e0b", "NLinear": "#34d399"}

            # ── Hàm backtest: trượt từng bước 1, lấy giá trị đầu tiên của dự đoán ──
            def backtest(ticker, pred_len, prices, scaler):
                scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
                split  = int(len(scaled) * 0.8)

                results = {name: [] for name in MODEL_REGISTRY}
                for i in range(split, len(scaled) - pred_len + 1):
                    if i < INPUT_LEN:
                        continue
                    x_in = torch.tensor(scaled[i - INPUT_LEN: i], dtype=torch.float32).unsqueeze(0)
                    for name in MODEL_REGISTRY:
                        m = load_model_cached(ticker, name, pred_len)
                        if m is None:
                            continue
                        with torch.no_grad():
                            out = m(x_in).numpy().flatten()
                        inv = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
                        results[name].append(float(inv[0]))  # chỉ lấy bước đầu tiên

                return {k: np.array(v) for k, v in results.items() if v}

            for label, pred_len in PRED_LENS.items():
                st.subheader(f"📅 Dự báo {label}")
                preds = predict(ticker, pred_len, prices, scaler)
                if not preds:
                    st.info("Chưa có mô hình, hãy chạy build_model.py trước.")
                    continue

                # ── Tính backtest ──────────────────────────────────────────
                bt_preds  = backtest(ticker, pred_len, prices, scaler)
                split_idx = int(len(prices) * 0.8)
                # step=1: mỗi ngày trong tập test là 1 điểm dự báo
                bt_indices = list(range(split_idx, len(prices) - pred_len + 1))
                bt_dates   = [prices_dates[i] for i in bt_indices]
                bt_actuals = [prices[i]       for i in bt_indices]

                # ── Dự báo tương lai ───────────────────────────────────────
                last_date   = sub["time"].max()
                future_days = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)

                fig = go.Figure()

                # Giá thực tế
                fig.add_trace(go.Scatter(
                    x=sub["time"], y=sub["close"],
                    mode="lines", name="Giá thực tế",
                    line=dict(color="#94a3b8", width=1.5)
                ))

                # Đường dự báo backtest — liên tục từ đầu tập test đến hôm nay
                for name, vals in bt_preds.items():
                    n = min(len(vals), len(bt_dates))
                    if n == 0:
                        continue
                    fig.add_trace(go.Scatter(
                        x=bt_dates[:n], y=vals[:n],
                        mode="lines", name=name,
                        line=dict(color=colors_model.get(name, "#fff"), width=1, dash="4px,4px")
                    ))

                # Đường kẻ dọc phân tách lịch sử / tương lai
                fig.add_vline(
                    x=pd.Timestamp(last_date).timestamp() * 1000,
                    line_dash="dash", line_color="#64748b", line_width=1.5,
                    annotation_text="Hôm nay", annotation_position="top right",
                    annotation_font_color="#94a3b8"
                )

                # Dự báo tương lai — nối liền từ điểm cuối
                for name, vals in preds.items():
                    connect_x = [last_date] + list(future_days)
                    connect_y = [prices[-1]] + list(vals)
                    fig.add_trace(go.Scatter(
                        x=connect_x, y=connect_y,
                        mode="lines", name=f"{name} →",
                        line=dict(color=colors_model.get(name, "#fff"), width=1.5, dash="dash"),
                        marker=dict(size=7),
                        showlegend=False
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

    # ── Tab 3: Đánh giá mô hình (đọc từ model_evaluation.csv) ──
    with tab3:
        EVAL_PATH = "data/model_evaluation.csv"
        if not os.path.exists(EVAL_PATH):
            st.warning("⚠️ Chưa có file đánh giá. Hãy chạy `src/evaluate.py` trước.")
            st.code("python src/evaluate.py", language="bash")
        else:
            eval_df = pd.read_csv(EVAL_PATH)
            # Lọc theo ticker đang chọn
            df_ticker = eval_df[eval_df["ticker"] == ticker].copy()

            if df_ticker.empty:
                st.info(f"Không có dữ liệu đánh giá cho {ticker}.")
            else:
                st.markdown("### 📊 Kết quả đánh giá trên tập kiểm tra (20% dữ liệu cuối)")
                st.caption("Nguồn: data/model_evaluation.csv — sinh bởi evaluate.py")

                colors_model = {"Linear": "#38bdf8", "DLinear": "#f59e0b", "NLinear": "#34d399"}

                HORIZON_LABELS = {1: "1 ngày (1 phiên)", 5: "1 tuần (5 phiên)", 21: "1 tháng (21 phiên)"}

                for pred_len, label in HORIZON_LABELS.items():
                    df_h = df_ticker[df_ticker["pred_len"] == pred_len].copy()
                    if df_h.empty:
                        continue

                    st.markdown(f"#### 📅 {label}")

                    # ── Bảng metrics ──────────────────────────────────
                    df_display = df_h[["model", "RMSE", "MAE", "R2"]].copy()
                    df_display.columns = ["Mô hình", "RMSE ", "MAE ", "R²"]

                    # Highlight mô hình tốt nhất (RMSE thấp nhất)
                    best_idx  = df_h["RMSE"].idxmin()
                    best_name = df_h.loc[best_idx, "model"]

                    def highlight_best(row):
                        return ["background-color: #1e3a2f; color: #34d399; font-weight:bold"
                                if row["Mô hình"] == best_name else "" for _ in row]

                    styled = (df_display.style
                              .apply(highlight_best, axis=1)
                              .format({"RMSE ": "{:,.0f}", "MAE ": "{:,.0f}", "R²": "{:.4f}"}))
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                
                    # ── Biểu đồ RMSE so sánh các mô hình ─────────────
                    model_names = df_h["model"].tolist()
                    rmse_vals   = df_h["RMSE"].tolist()
                    bar_colors  = [
                        "#34d399" if n == best_name else colors_model.get(n, "#94a3b8")
                        for n in model_names
                    ]

                    col_left, col_right = st.columns(2)



# ════════════════════════════════════════════════════════════
# TRANG 3: TIN TỨC  (RSS — CafeF / VnExpress / Vietstock)
# ════════════════════════════════════════════════════════════
elif page == "📰 Tin tức":
    st.title("📰 Tin tức biến động ngân hàng")
    st.caption("Nguồn: CafeF · VnExpress · Vietstock (RSS) — cập nhật mỗi 20 phút")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        ticker_news = st.selectbox(
            "Chọn ngân hàng",
            BANK_TICKERS,
            format_func=lambda t: f"{t} — {BANK_NAMES.get(t, t)}",
            key="news_ticker"
        )

    with col_right:
        if st.button("🔄 Làm mới tin tức"):
            st.cache_data.clear()
            st.rerun()

    bank_name = BANK_NAMES.get(ticker_news, ticker_news)
    st.subheader(f"🏦 Tin tức: {bank_name} ({ticker_news})")

    with st.spinner("Đang tải tin tức từ RSS..."):
        news_list = fetch_news(ticker_news)

    if not news_list:
        st.info("Không tìm thấy tin tức gần đây.")
    else:
        st.caption(f"Tìm thấy {len(news_list)} bài viết")
        for item in news_list:
            summary_html = (
                f"<div style='color:#94a3b8; font-size:13px; margin-top:6px;'>{item['summary']}</div>"
                if item.get("summary") else ""
            )
            st.markdown(f"""
            <div class="news-card">
                <a href="{item['url']}" target="_blank">{item['title']}</a>
                {summary_html}
                <div class="news-meta">📅 {item['published']} &nbsp;|&nbsp; 📰 {item['source']}</div>
            </div>
            """, unsafe_allow_html=True)