"""
train.py — Quy trình train 2 pha cho hệ thống dự báo cổ phiếu ngân hàng.

═══════════════════════════════════════════════════════════════
  PHA 1 — PRETRAIN trên mã đại diện (VCB)
    • Train đầy đủ 3 model × 3 window trên VCB
    • So sánh val_loss → chọn model tốt nhất cho mỗi window
    • Lưu: models/pretrain_<model>_<window>d.pth
    • Lưu: logs/pretrain_selection.json  ← kết quả chọn model

  PHA 2 — FINETUNE lớp cuối trên 29 mã còn lại
    • Load weight pretrain của model tốt nhất
    • Đóng băng toàn bộ, chỉ mở lớp Linear cuối
    • Finetune FINETUNE_EPOCHS epoch với LR nhỏ
    • Lưu: models/finetune_<model>_<symbol>_<window>d.pth
═══════════════════════════════════════════════════════════════

Cách dùng:
    python train.py                  # chạy đầy đủ cả 2 pha
    python train.py --phase 1        # chỉ pretrain VCB
    python train.py --phase 2        # chỉ finetune 29 mã (cần pha 1 xong trước)
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from models_def import build_model
from data_utils import fetch_data, explore_data, preprocess_data, StockDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "", width: int = 65, char: str = "─") -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{char*pad} {title} {char*pad}")
    else:
        print(char * width)


def _pretrain_path(model_name: str, window: int) -> str:
    return f"models/pretrain_{model_name.lower()}_{window}d.pth"


def _finetune_path(model_name: str, symbol: str, window: int) -> str:
    return f"models/finetune_{model_name.lower()}_{symbol}_{window}d.pth"


def _selection_path() -> str:
    return "logs/pretrain_selection.json"


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_epochs(
    model, train_loader, val_loader, device,
    n_epochs: int, lr: float, freeze_backbone: bool = False,
) -> tuple:
    """
    Train loop tổng quát — dùng cho cả pretrain lẫn finetune.

    Args:
        freeze_backbone : Nếu True, chỉ cập nhật tham số có requires_grad=True.
    Returns:
        (model, train_losses, val_losses)
    """
    criterion = nn.MSELoss()

    # Chỉ optimize các tham số đang mở
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_train = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx).view(-1)
            loss = criterion(pred, by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_train += loss.item()
        train_losses.append(epoch_train / max(1, len(train_loader)))

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                epoch_val += criterion(model(bx).view(-1), by.view(-1)).item()
        val_losses.append(epoch_val / max(1, len(val_loader)))

        scheduler.step()

        log_every = max(1, n_epochs // 5)
        if (epoch + 1) % log_every == 0:
            print(f"      Epoch {epoch+1:3d}/{n_epochs} — "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    return model, train_losses, val_losses


def _make_loaders(data: np.ndarray, window: int):
    """Tạo train/val DataLoader với split 80/20."""
    split        = int(0.8 * len(data))
    train_ds     = StockDataset(data[:split], window)
    val_ds       = StockDataset(data[split:], window)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, drop_last=False)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def _save_loss_plot(tag: str, window: int, loss_dict: dict) -> None:
    """Lưu loss curves của 3 model vào 1 figure."""
    fig, axes  = plt.subplots(1, 3, figsize=(15, 4))
    colors     = {"LSTM": "#e74c3c", "DLinear": "#2ecc71", "NLinear": "#3498db"}

    for ax, (name, (tr, va)) in zip(axes, loss_dict.items()):
        c = colors.get(name, "gray")
        ax.plot(tr, color=c, linewidth=1.5, label="Train")
        ax.plot(va, color=c, linewidth=1.5, linestyle="--", alpha=0.7, label="Val")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{tag} — Window {window}d | Loss Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"logs/loss_{tag}_{window}d.png"
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"      💾 Loss plot → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PHA 1 — PRETRAIN trên VCB
# ═════════════════════════════════════════════════════════════════════════════

def phase1_pretrain(device: str = "cpu") -> dict:
    """
    Train đầy đủ 3 model × 3 window trên mã đại diện (config.PRETRAIN_SYMBOL).
    Chọn model tốt nhất cho mỗi window dựa trên val_loss thấp nhất.

    Returns:
        selection: dict {window: {"best_model": str, "val_loss": float, ...}}
    """
    symbol = config.PRETRAIN_SYMBOL
    _sep(f"PHA 1 — PRETRAIN trên {symbol}", char="═")
    print(f"  Mã đại diện : {symbol}")
    print(f"  Epochs      : {config.EPOCHS}")
    print(f"  Windows     : {config.INPUT_WINDOWS}")
    print(f"  Models      : LSTM, DLinear, NLinear")
    print(f"  Device      : {device}")

    # Thu thập & xử lý dữ liệu VCB
    _sep("Thu thập dữ liệu")
    df = fetch_data(symbol, config.START_DATE, config.END_DATE, verbose=True)
    if df is None:
        raise RuntimeError(f"Không thể tải dữ liệu cho {symbol}. Dừng pha 1.")

    explore_data(df, symbol=symbol)

    _sep("Tiền xử lý")
    data = preprocess_data(df, verbose=True)
    n_features = data.shape[1]
    print(f"\n  ✅ Data shape: {data.shape}  ({n_features} features, cột 0 = log_return)")

    if len(data) < 150:
        raise RuntimeError(f"Dữ liệu quá ngắn ({len(data)} điểm).")

    selection = {}   # {window: {best_model, val_loss, all_results}}

    for window in config.INPUT_WINDOWS:
        _sep(f"Window = {window}d")

        if len(data) < window + 50:
            print(f"  ⚠️  Bỏ qua window={window}d — dữ liệu quá ngắn.")
            continue

        train_loader, val_loader, n_train, n_val = _make_loaders(data, window)
        print(f"  Train: {n_train} mẫu  |  Val: {n_val} mẫu")

        window_results = {}   # {model_name: val_loss}
        loss_dict      = {}   # cho plot

        for name in ["LSTM", "DLinear", "NLinear"]:
            print(f"\n  ▶ {name}...")
            model = build_model(name, window, config.HIDDEN_SIZE,
                                n_features=n_features)
            trained, tr_loss, va_loss = _run_epochs(
                model, train_loader, val_loader, device,
                n_epochs=config.EPOCHS, lr=config.LEARNING_RATE,
            )

            # Lưu pretrain weight
            path = _pretrain_path(name, window)
            torch.save(trained.state_dict(), path)

            window_results[name] = va_loss[-1]
            loss_dict[name]      = (tr_loss, va_loss)

            print(f"  ✅ {name:8s} | val_loss={va_loss[-1]:.6f} | saved: {path}")

        _save_loss_plot(f"pretrain_{symbol}", window, loss_dict)

        # Chọn model tốt nhất cho window này
        best_name = min(window_results, key=window_results.get)
        selection[window] = {
            "best_model" : best_name,
            "val_loss"   : window_results[best_name],
            "all_results": window_results,
        }

        print(f"\n  🏆 Window {window}d — Model tốt nhất: {best_name} "
              f"(val_loss={window_results[best_name]:.6f})")
        print(f"     So sánh: " +
              "  |  ".join(f"{k}: {v:.6f}" for k, v in window_results.items()))

    # Lưu selection JSON
    with open(_selection_path(), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False,
                  default=lambda x: int(x) if isinstance(x, np.integer) else x)

    _sep("KẾT QUẢ PHA 1", char="═")
    print(f"  Pretrain symbol : {symbol}")
    print(f"  Selection file  : {_selection_path()}")
    print()
    for w, info in selection.items():
        print(f"  Window {w:2d}d → Model tốt nhất: {info['best_model']:8s} "
              f"| val_loss = {info['val_loss']:.6f}")

    return selection


# ═════════════════════════════════════════════════════════════════════════════
# PHA 2 — FINETUNE lớp cuối trên 29 mã còn lại
# ═════════════════════════════════════════════════════════════════════════════

def _freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    """
    Đóng băng toàn bộ tham số, sau đó mở lại chỉ lớp Linear cuối cùng.

    Chiến lược theo kiến trúc:
        LSTM    → chỉ mở model.fc
        DLinear → chỉ mở model.Linear_S, model.Linear_T
        NLinear → chỉ mở model.Linear
    """
    # Đóng băng tất cả
    for p in model.parameters():
        p.requires_grad = False

    name = model_name.upper()
    if name == "LSTM":
        for p in model.fc.parameters():
            p.requires_grad = True
        unfrozen = "fc"
    elif name == "DLINEAR":
        for layer in [model.Linear_S, model.Linear_T]:
            for p in layer.parameters():
                p.requires_grad = True
        unfrozen = "Linear_S, Linear_T"
    else:  # NLINEAR
        for p in model.Linear.parameters():
            p.requires_grad = True
        unfrozen = "Linear"

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_open   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Frozen: {n_frozen:,} params  |  Open ({unfrozen}): {n_open:,} params")
    return model


def phase2_finetune(selection: dict, device: str = "cpu") -> None:
    """
    Load weight pretrain tốt nhất → đóng băng backbone → finetune lớp cuối
    trên từng mã trong SYMBOLS (trừ PRETRAIN_SYMBOL).

    Args:
        selection : dict từ phase1_pretrain() hoặc load từ JSON.
    """
    symbols   = [s for s in config.SYMBOLS if s != config.PRETRAIN_SYMBOL]
    n_symbols = len(symbols)

    _sep(f"PHA 2 — FINETUNE {n_symbols} mã còn lại", char="═")
    print(f"  Mã finetune     : {symbols}")
    print(f"  Finetune epochs : {config.FINETUNE_EPOCHS}")
    print(f"  Finetune LR     : {config.FINETUNE_LR}")
    print(f"  Device          : {device}")

    for sym_idx, symbol in enumerate(symbols, 1):
        t0 = time.time()
        _sep(f"[{sym_idx}/{n_symbols}] {symbol}")

        # Tải & xử lý dữ liệu
        df = fetch_data(symbol, config.START_DATE, config.END_DATE, verbose=True)
        if df is None:
            print(f"  ⚠️  Bỏ qua {symbol} — không tải được dữ liệu.")
            continue

        try:
            data = preprocess_data(df, verbose=False)
        except Exception as e:
            print(f"  ❌ Lỗi xử lý dữ liệu [{symbol}]: {e}")
            continue

        n_features = data.shape[1]
        print(f"  Data: {data.shape}  ({n_features} features)")

        if len(data) < 100:
            print(f"  ⚠️  Quá ít dữ liệu ({len(data)} điểm), bỏ qua.")
            continue

        for window, info in selection.items():
            window    = int(window)
            best_name = info["best_model"]
            pretrain  = _pretrain_path(best_name, window)

            if not os.path.exists(pretrain):
                print(f"  ⚠️  Thiếu pretrain weight: {pretrain}")
                continue

            if len(data) < window + 30:
                print(f"  ⚠️  Bỏ qua window={window}d — dữ liệu quá ngắn.")
                continue

            print(f"\n  Window={window}d | Model={best_name}")

            # Load pretrain weight
            model = build_model(best_name, window, config.HIDDEN_SIZE,
                                n_features=n_features)
            state = torch.load(pretrain, map_location="cpu", weights_only=True)
            try:
                model.load_state_dict(state)
            except RuntimeError as e:
                print(f"    ⚠️  Không load được state_dict: {e}")
                continue

            # Đóng băng backbone, mở lớp cuối
            model = _freeze_backbone(model, best_name)

            # Finetune
            train_loader, val_loader, n_train, n_val = _make_loaders(data, window)
            print(f"    Train: {n_train} mẫu  |  Val: {n_val} mẫu")

            trained, tr_loss, va_loss = _run_epochs(
                model, train_loader, val_loader, device,
                n_epochs=config.FINETUNE_EPOCHS,
                lr=config.FINETUNE_LR,
                freeze_backbone=True,
            )

            # Lưu finetune weight
            out_path = _finetune_path(best_name, symbol, window)
            torch.save(trained.state_dict(), out_path)

            print(f"    ✅ val_loss={va_loss[-1]:.6f} | saved: {out_path}")

        elapsed = time.time() - t0
        print(f"\n  [{symbol}] 🏁 Xong trong {elapsed:.1f}s")

    _sep("KẾT QUẢ PHA 2", char="═")
    print(f"  Finetune weights → models/finetune_<model>_<symbol>_<window>d.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train: Pretrain VCB → Finetune 29 mã")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=0,
        help="1=chỉ pretrain, 2=chỉ finetune, bỏ trống=cả 2 pha",
    )
    args   = parser.parse_args()
    device = config.DEVICE

    for folder in ["models", "logs"]:
        os.makedirs(folder, exist_ok=True)

    t_start = time.time()

    print("\n" + "═"*65)
    print("  HỆ THỐNG DỰ BÁO CỔ PHIẾU NGÂN HÀNG — TRAIN PIPELINE")
    print("═"*65)
    print(f"  Pretrain symbol : {config.PRETRAIN_SYMBOL}")
    print(f"  Finetune symbols: {len(config.SYMBOLS)-1} mã còn lại")
    print(f"  Windows         : {config.INPUT_WINDOWS}")
    print(f"  Pretrain epochs : {config.EPOCHS}")
    print(f"  Finetune epochs : {config.FINETUNE_EPOCHS}")
    print(f"  Device          : {device}")
    print("═"*65)

    # ── Pha 1 ────────────────────────────────────────────────────────────────
    if args.phase in [0, 1]:
        selection = phase1_pretrain(device=device)
    else:
        # Load selection từ file JSON nếu bỏ qua pha 1
        sel_path = _selection_path()
        if not os.path.exists(sel_path):
            print(f"\n❌ Không tìm thấy {sel_path}. Hãy chạy pha 1 trước: python train.py --phase 1")
            return
        with open(sel_path, encoding="utf-8") as f:
            selection_raw = json.load(f)
        # JSON key là string, chuyển lại thành int
        selection = {int(k): v for k, v in selection_raw.items()}
        print(f"\n✅ Load selection từ {sel_path}:")
        for w, info in selection.items():
            print(f"   Window {w}d → {info['best_model']} (val_loss={info['val_loss']:.6f})")

    # ── Pha 2 ────────────────────────────────────────────────────────────────
    if args.phase in [0, 2]:
        phase2_finetune(selection, device=device)

    total = (time.time() - t_start) / 60
    print(f"\n{'═'*65}")
    print(f"  ✅ HOÀN THÀNH — tổng thời gian: {total:.1f} phút")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()