# ============================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH (nâng cấp)
# - Walk-forward CV (4 folds)
# - Naive baseline 
# - Diebold-Mariano test (so sánh top model vs Naive)
# - Conditional MAE theo regime (low / normal / crisis)
# - Variance Ratio
# ============================================================

import pandas as pd
import numpy as np
import torch
import pickle, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(__file__))
from models import MODEL_REGISTRY
from build_model import INPUT_LEN, PRED_LENS

os.makedirs("data/eval_plots", exist_ok=True)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def inv(arr, scaler):
    return scaler.inverse_transform(np.array(arr).reshape(-1, 1)).flatten()


def load_model(ticker, model_name, pred_len):
    from models import MODEL_REGISTRY
    ModelClass = MODEL_REGISTRY[model_name]
    model      = ModelClass(INPUT_LEN, pred_len)
    path       = f"data/models/{ticker}_{model_name}_pred{pred_len}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def make_sequences(series, input_len, pred_len):
    X, y = [], []
    for i in range(len(series) - input_len - pred_len + 1):
        X.append(series[i: i + input_len])
        y.append(series[i + input_len: i + input_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_model(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def diebold_mariano(e1, e2, h=1):
    """
    DM test: H0: hai mô hình có cùng độ chính xác.
    e1, e2: mảng sai số (y_true - y_pred) của mô hình 1 và 2.
    Trả về (dm_stat, p_value).
    """
    d  = e1**2 - e2**2          # loss differential
    T  = len(d)
    dbar = np.mean(d)

    # Newey-West variance (bandwidth = h)
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for lag in range(1, h + 1):
        gamma_l = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        nw_var += 2 * (1 - lag / (h + 1)) * gamma_l
    nw_var = max(nw_var, 1e-12)

    dm_stat = dbar / np.sqrt(nw_var / T)
    p_val   = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


def label_regimes(prices):
    """
    Gán nhãn regime dựa trên rolling volatility (std 21 ngày).
    - low:    volatility < 33rd percentile
    - normal: 33rd ≤ volatility ≤ 67th percentile
    - crisis: volatility > 67th percentile
    """
    returns  = pd.Series(prices).pct_change().fillna(0)
    vol      = returns.rolling(21, min_periods=5).std().fillna(0).values
    p33, p67 = np.percentile(vol, 33), np.percentile(vol, 67)
    regimes  = np.where(vol > p67, "crisis",
               np.where(vol < p33, "low", "normal"))
    return regimes


# ════════════════════════════════════════════════════════════
# WALK-FORWARD CV
# ════════════════════════════════════════════════════════════
def walk_forward_cv(prices_scaled, prices_raw, pred_len, models_dict,
                    n_folds=4, train_ratio=0.6):
    """
    Walk-forward CV với n_folds folds.
    Mỗi fold: train trên [0, train_end], test trên [train_end, test_end].
    train_end tăng dần, test window = (1 - train_ratio) / n_folds * N.

    Trả về DataFrame với MAE từng fold từng mô hình.
    """
    N          = len(prices_scaled)
    fold_size  = int((N * (1 - train_ratio)) / n_folds)
    train_base = int(N * train_ratio)

    records = []
    for fold in range(n_folds):
        train_end = train_base + fold * fold_size
        test_end  = min(train_end + fold_size, N)

        if train_end < INPUT_LEN + pred_len:
            continue
        if test_end <= train_end:
            break

        # Scaler fit chỉ trên fold train
        from sklearn.preprocessing import MinMaxScaler
        fold_scaler = MinMaxScaler()
        fold_scaler.fit(prices_raw[:train_end].reshape(-1, 1))

        fold_train = fold_scaler.transform(prices_raw[:train_end].reshape(-1, 1)).flatten()
        fold_all   = fold_scaler.transform(prices_raw[:test_end].reshape(-1, 1)).flatten()

        # Test sequences: ghép context cuối train
        test_series = np.concatenate([fold_train[-INPUT_LEN:],
                                      fold_all[train_end:test_end]])
        X_test, y_test = make_sequences(test_series, INPUT_LEN, pred_len)

        if len(X_test) == 0:
            continue

        y_true_raw = inv(y_test.flatten(), fold_scaler)
        y_train_raw = prices_raw[:train_end]

        for name, model in models_dict.items():
            preds     = run_model(model, X_test)
            y_pred_raw = inv(preds.flatten(), fold_scaler)
            mae_val   = mean_absolute_error(y_true_raw, y_pred_raw)
            records.append({"fold": fold + 1, "model": name,
                            "pred_len": pred_len, "MAE": mae_val})

        # Naive
        naive_preds = np.tile(X_test[:, -1:], (1, pred_len))
        y_naive_raw = inv(naive_preds.flatten(), fold_scaler)
        records.append({"fold": fold + 1, "model": "Naive",
                        "pred_len": pred_len,
                        "MAE": mean_absolute_error(y_true_raw, y_naive_raw)})

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════
# ĐÁNH GIÁ CHÍNH (holdout + metrics mở rộng)
# ════════════════════════════════════════════════════════════

def evaluate_one(ticker, model_name, pred_len, X_test, y_test, scaler, y_train_raw):
    model = load_model(ticker, model_name, pred_len)
    preds = run_model(model, X_test)

    y_true_all = inv(y_test.flatten(), scaler)
    y_pred_all = inv(preds.flatten(), scaler)

    rmse_val  = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae_val   = float(mean_absolute_error(y_true_all, y_pred_all))
    r2_val    = float(r2_score(y_true_all, y_pred_all))
   

    mae_by_step = []
    for step in range(pred_len):
        yt = inv(y_test[:, step], scaler)
        yp = inv(preds[:, step], scaler)
        mae_by_step.append(float(mean_absolute_error(yt, yp)))

    # Variance Ratio: std(pred) / std(true)
    vr = float(np.std(y_pred_all) / (np.std(y_true_all) + 1e-8))

    errors = y_true_all - y_pred_all

    return {
        "ticker": ticker, "model": model_name, "pred_len": pred_len,
        "RMSE": rmse_val, "MAE": mae_val, "R2": r2_val,
        "VarianceRatio": vr,
        "mae_by_step": mae_by_step,
        "errors": errors,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "preds_raw": preds,
    }


def evaluate_naive(X_test, y_test, scaler, y_train_raw, pred_len, ticker):
    naive_preds = np.tile(X_test[:, -1:], (1, pred_len))
    y_true_all  = inv(y_test.flatten(), scaler)
    y_pred_all  = inv(naive_preds.flatten(), scaler)

    mae_by_step = []
    for step in range(pred_len):
        yt = inv(y_test[:, step], scaler)
        yp = inv(naive_preds[:, step], scaler)
        mae_by_step.append(float(mean_absolute_error(yt, yp)))

    vr     = float(np.std(y_pred_all) / (np.std(y_true_all) + 1e-8))
    errors = y_true_all - y_pred_all

    return {
        "ticker": ticker, "model": "Naive", "pred_len": pred_len,
        "RMSE": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "MAE":  float(mean_absolute_error(y_true_all, y_pred_all)),
        "R2":   float(r2_score(y_true_all, y_pred_all)),
        "VarianceRatio": vr,
        "mae_by_step": mae_by_step,
        "errors": errors,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


# ════════════════════════════════════════════════════════════
# CONDITIONAL MAE PER REGIME
# ════════════════════════════════════════════════════════════

def conditional_mae_per_regime(y_true, y_pred, regimes_test):
    """
    Tính MAE riêng cho 3 regime: low / normal / crisis.
    regimes_test có thể có độ dài N_test (per-sequence),
    còn y_true/y_pred có độ dài N_test × pred_len (flattened).
    → Tile regimes_test để khớp với y_true.
    """
    n_seq  = len(regimes_test)
    n_flat = len(y_true)
    if n_flat > n_seq:
        # pred_len = n_flat // n_seq (làm tròn an toàn)
        pred_len = max(1, round(n_flat / n_seq))
        regimes_expanded = np.repeat(regimes_test[:n_flat // pred_len], pred_len)
        # Cắt/pad cho khớp đúng độ dài
        min_len = min(len(regimes_expanded), n_flat)
        regimes_expanded = regimes_expanded[:min_len]
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    else:
        regimes_expanded = regimes_test[:n_flat]
        y_true = y_true[:len(regimes_expanded)]
        y_pred = y_pred[:len(regimes_expanded)]

    result = {}
    for regime in ["low", "normal", "crisis"]:
        mask = regimes_expanded == regime
        if mask.sum() > 0:
            result[regime] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        else:
            result[regime] = np.nan
    return result


# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ
# ════════════════════════════════════════════════════════════

def plot_mae_by_step(rows, ticker, pred_len):
    # Sử dụng style dark_background để tự động chuyển text/ticks sang màu sáng
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#121212')
        ax.set_facecolor('#121212') # Màu nền phía trong đồ thị
        
        colors = {"Linear": "#38bdf8", "DLinear": "#f59e0b",
                  "NLinear": "#34d399", "Naive": "#f87171"}
        
        for row in rows:
            ax.plot(range(1, pred_len + 1), row["mae_by_step"],
                    marker="o", 
                    markersize=4,
                    linewidth=1.5,
                    label=row["model"],
                    color=colors.get(row["model"], "gray"))
        
        # Tinh chỉnh tiêu đề và nhãn
        ax.set_xlabel("Bước dự báo (ngày)", color="#e0e0e0")
        ax.set_ylabel("MAE (VNĐ)", color="#e0e0e0")
        ax.set_title(f"{ticker} — MAE theo từng bước | horizon={pred_len} ngày", 
                     color="white", pad=15)
        
        # Tinh chỉnh chú thích và lưới
        ax.legend(facecolor='#1e1e1e', edgecolor='#444444')
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Đảm bảo trục X hiển thị số nguyên (bước dự báo)
        ax.set_xticks(range(1, pred_len + 1))
        
        plt.tight_layout()
        
        path = f"data/eval_plots/{ticker}_mae_step_pred{pred_len}.png"
        
        # Lưu ảnh với tham số facecolor để không bị nền trắng khi xuất file
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        
    return path
def plot_walk_forward(wf_df, ticker, pred_len):
    subset = wf_df[wf_df["pred_len"] == pred_len]
    if subset.empty:
        return
    
    pivot = subset.pivot(index="fold", columns="model", values="MAE")
    colors = {"Linear": "#38bdf8", "DLinear": "#f59e0b",
              "NLinear": "#34d399", "Naive": "#f87171"}

    # Sử dụng style dark_background
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#121212')
        ax.set_facecolor('#121212') # Đổi màu nền bên trong biểu đồ

        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", markersize=5,
                    linewidth=1.5, label=col, color=colors.get(col, "gray"))
        
        ax.set_xlabel("Fold", color="#e0e0e0")
        ax.set_ylabel("MAE (VNĐ)", color="#e0e0e0")
        ax.set_title(f"{ticker} — Walk-forward CV | horizon={pred_len} ngày", color="white", pad=15)
        
        ax.legend(facecolor='#1e1e1e', edgecolor='#444444')
        ax.grid(True, alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        path = f"data/eval_plots/{ticker}_wf_pred{pred_len}.png"
        # Đảm bảo facecolor được giữ nguyên khi lưu
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
def plot_regime_mae(regime_records, ticker, pred_len):
    df = pd.DataFrame(regime_records)
    if df.empty:
        return

    regimes = ["low", "normal", "crisis"]
    colors = {"Linear": "#38bdf8", "DLinear": "#f59e0b",
              "NLinear": "#34d399", "Naive": "#f87171"}

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#121212')
        ax.set_facecolor('#121212')

        x = np.arange(len(regimes))
        width = 0.15
        
        for i, row in df.iterrows():
            vals = [row.get(r, np.nan) for r in regimes]
            # Tính toán vị trí bar để căn giữa
            pos = x + (i - len(df)/2 + 0.5) * width
            ax.bar(pos, vals, width, label=row["model"], 
                   color=colors.get(row["model"], "gray"), alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(["Low vol", "Normal", "Crisis"], color="#e0e0e0")
        ax.set_ylabel("MAE (VNĐ)", color="#e0e0e0")
        ax.set_title(f"{ticker} — MAE theo Regime | horizon={pred_len} ngày", color="white", pad=15)
        
        ax.legend(facecolor='#1e1e1e', edgecolor='#444444')
        ax.grid(axis="y", alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        path = f"data/eval_plots/{ticker}_regime_pred{pred_len}.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def evaluate_all(data_path="data/processed/clean_data.csv"):
    print("=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH (Walk-forward CV + DM test + Regime MAE)")
    print("=" * 60)

    df      = pd.read_csv(data_path, parse_dates=["time"])
    results = []        # holdout metrics (không có mae_by_step)
    dm_results   = []   # DM test
    regime_rows  = []   # conditional MAE per regime
    all_wf_dfs   = []   # walk-forward CV

    for ticker in df["ticker"].unique():
        sub    = df[df["ticker"] == ticker].sort_values("time")
        prices = sub["close"].values

        with open(f"data/models/{ticker}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Regime labels (trên toàn chuỗi giá gốc)
        regimes_all = label_regimes(prices)

        for pred_len in PRED_LENS:
            # ── Load test sequences từ build_model ──────────────────
            X_test = np.load(f"data/models/{ticker}_Xtest_pred{pred_len}.npy")
            y_test = np.load(f"data/models/{ticker}_ytest_pred{pred_len}.npy")

            price_split  = int(len(prices) * 0.8)
            y_train_raw  = prices[:price_split]

            # Regime cho test (lấy bước đầu của mỗi cửa sổ test)
            # test sequences bắt đầu từ index price_split
            regime_test_idx = np.arange(price_split,
                                        price_split + len(X_test))
            regime_test_idx = np.clip(regime_test_idx, 0, len(regimes_all) - 1)
            regimes_test    = regimes_all[regime_test_idx]

            step_rows    = []   # cho mae-by-step plot
            regime_block = []   # cho regime plot

            # ── Walk-forward CV ──────────────────────────────────────
            models_dict = {}
            for name in MODEL_REGISTRY:
                m = load_model(ticker, name, pred_len)
                if m:
                    models_dict[name] = m

            wf_df = walk_forward_cv(
                prices_scaled=scaler.transform(prices.reshape(-1, 1)).flatten(),
                prices_raw=prices,
                pred_len=pred_len,
                models_dict=models_dict,
                n_folds=4,
            )
            wf_df["ticker"] = ticker
            all_wf_dfs.append(wf_df)
            plot_walk_forward(wf_df, ticker, pred_len)

            # ── Naive baseline ───────────────────────────────────────
            naive_res = evaluate_naive(X_test, y_test, scaler,
                                       y_train_raw, pred_len, ticker)
            step_rows.append(naive_res)
            results.append({k: v for k, v in naive_res.items()
                            if k not in ("mae_by_step", "errors",
                                         "y_true", "y_pred")})
            cond_naive = conditional_mae_per_regime(
                naive_res["y_true"], naive_res["y_pred"], regimes_test[:len(naive_res["y_true"])])
            regime_block.append({"model": "Naive", **cond_naive})

            # ── Mỗi mô hình ─────────────────────────────────────────
            best_res = None
            for model_name in MODEL_REGISTRY:
                res = evaluate_one(ticker, model_name, pred_len,
                                   X_test, y_test, scaler, y_train_raw)
                step_rows.append(res)
                results.append({k: v for k, v in res.items()
                                if k not in ("mae_by_step", "errors",
                                             "y_true", "y_pred",
                                             "preds_raw")})

                # Conditional MAE per regime
                cond = conditional_mae_per_regime(
                    res["y_true"], res["y_pred"],
                    regimes_test[:len(res["y_true"])])
                regime_block.append({"model": model_name, **cond})
                regime_rows.append({"ticker": ticker, "pred_len": pred_len,
                                    "model": model_name, **cond})

                # Chọn mô hình tốt nhất (MAE thấp nhất) để DM test
                if best_res is None or res["MAE"] < best_res["MAE"]:
                    best_res = res

            # ── Diebold-Mariano: best model vs Naive ────────────────
            if best_res is not None:
                dm_stat, p_val = diebold_mariano(
                    best_res["errors"], naive_res["errors"], h=pred_len)
                dm_results.append({
                    "ticker":    ticker,
                    "pred_len":  pred_len,
                    "best_model": best_res["model"],
                    "DM_stat":   round(dm_stat, 4),
                    "p_value":   round(p_val, 4),
                    "significant": "✅" if p_val < 0.05 else "❌",
                    "Naive_MAE": round(naive_res["MAE"], 2),
                    "Best_MAE":  round(best_res["MAE"], 2),
                })

            # ── Plots ────────────────────────────────────────────────
            plot_mae_by_step(step_rows, ticker, pred_len)
            plot_regime_mae(regime_block, ticker, pred_len)

            print(f"  ✅ {ticker} | pred_len={pred_len} "
                  f"| best={best_res['model'] if best_res else '?'} "
                  f"| VR={best_res['VarianceRatio']:.3f}" if best_res else "")

    # ── Lưu kết quả ─────────────────────────────────────────
    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv("data/model_evaluation.csv", index=False)

    dm_df = pd.DataFrame(dm_results)
    dm_df.to_csv("data/dm_test_results.csv", index=False)

    regime_df = pd.DataFrame(regime_rows).round(2)
    regime_df.to_csv("data/regime_mae.csv", index=False)

    if all_wf_dfs:
        wf_all = pd.concat(all_wf_dfs, ignore_index=True).round(2)
        wf_all.to_csv("data/walkforward_cv.csv", index=False)

    # ── In tổng quan ─────────────────────────────────────────
    print("\n📌 Kết quả Holdout (trung bình toàn ngân hàng):")
    summary = (results_df
               .groupby(["model", "pred_len"])[["RMSE", "MAE", "VarianceRatio"]]
               .mean().round(4))
    print(summary.to_string())

    print("\n🔬 Diebold-Mariano Test (best model vs Naive):")
    print(dm_df.to_string(index=False))

    print("\n⚠️  Variance Ratio < 0.4 (over-smoothing):")
    low_vr = results_df[results_df["VarianceRatio"] < 0.4][
        ["ticker", "model", "pred_len", "VarianceRatio"]]
    print(low_vr.to_string(index=False) if not low_vr.empty else "  Không có trường hợp nào.")

    print("\n📊 Conditional MAE theo Regime (trung bình toàn ngân hàng):")
    print(regime_df.groupby(["model", "pred_len"])[["low", "normal", "crisis"]]
          .mean().round(2).to_string())

    print("\n💾 Files đã lưu:")
    print("   data/model_evaluation.csv")
    print("   data/dm_test_results.csv")
    print("   data/regime_mae.csv")
    print("   data/walkforward_cv.csv")
    print("   data/eval_plots/  (MAE-by-step, walk-forward, regime)")

    return results_df, dm_df, regime_df


if __name__ == "__main__":
    evaluate_all()