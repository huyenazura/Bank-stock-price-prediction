"""
models_def.py — Định nghĩa 3 kiến trúc mô hình dự báo chuỗi thời gian.

    LSTMModel    — Mạng hồi quy với bộ nhớ dài hạn.
    DLinearModel — Phân tách xu hướng + mùa vụ, dự báo tuyến tính.
    NLinearModel — Chuẩn hóa theo giá cuối, dự báo tuyến tính.

Tham khảo: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
"""

import torch
import torch.nn as nn


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Input(1) → LSTM(hidden_size, num_layers) → Dropout → Linear → Output(1)
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.1)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, window, 1]
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)   # [B, 1]


# ── Moving Average ────────────────────────────────────────────────────────────

class _MovingAvg(nn.Module):
    """Average Pooling 1D với padding đầu/cuối để giữ độ dài chuỗi."""
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg         = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, seq_len, C]
        pad_front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        pad_back  = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([pad_front, x, pad_back], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)   # [B, seq_len, C]


class _SeriesDecomp(nn.Module):
    """Phân tách: Trend (MA) + Seasonal (Residual)."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ── DLinear ───────────────────────────────────────────────────────────────────

class DLinearModel(nn.Module):
    """
    Decomposition-Linear:
      1. Phân tách x → seasonal + trend
      2. Dự báo tuyến tính riêng biệt cho từng thành phần
      3. Cộng lại → output

    Args:
        seq_len  : Độ dài cửa sổ đầu vào.
        pred_len : Số bước dự báo (1 ngày / 5 ngày / 21 ngày).
        channels : Số feature (mặc định 1 = log return).
    """
    def __init__(self, seq_len: int, pred_len: int = 1, channels: int = 1):
        super().__init__()
        kernel            = min(25, seq_len // 2 * 2 + 1)
        self.decomp       = _SeriesDecomp(kernel_size=kernel)
        self.Linear_S     = nn.Linear(seq_len, pred_len)
        self.Linear_T     = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, seq_len, 1]
        seasonal, trend = self.decomp(x)
        s = self.Linear_S(seasonal.permute(0, 2, 1))   # [B, 1, pred_len]
        t = self.Linear_T(trend.permute(0, 2, 1))       # [B, 1, pred_len]
        out = (s + t).permute(0, 2, 1)                  # [B, pred_len, 1]
        return out.squeeze(-1)                           # [B, pred_len]


# ── NLinear ───────────────────────────────────────────────────────────────────

class NLinearModel(nn.Module):
    """
    Normalized-Linear:
      1. Trừ giá trị cuối → loại bỏ distribution shift
      2. Dự báo tuyến tính
      3. Cộng lại giá trị đã trừ

    Args:
        seq_len  : Độ dài cửa sổ đầu vào.
        pred_len : Số bước dự báo.
        channels : Số features đầu vào.
    """
    def __init__(self, seq_len: int, pred_len: int = 1, channels: int = 1):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, seq_len, C]
        seq_last = x[:, -1:, :].detach()
        x        = x - seq_last
        # Chỉ dùng cột 0 (log_return) cho dự báo
        x0       = x[:, :, 0:1]                                         # [B, seq_len, 1]
        x0       = self.Linear(x0.permute(0, 2, 1)).permute(0, 2, 1)    # [B, pred_len, 1]
        return (x0 + seq_last[:, :, 0:1]).squeeze(-1)                    # [B, pred_len]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(name: str, window: int, hidden_size: int = 64,
                n_features: int = 1) -> nn.Module:
    """
    Khởi tạo model theo tên.

    Args:
        name       : "LSTM" | "DLinear" | "NLinear"
        window     : Độ dài cửa sổ đầu vào.
        hidden_size: Số unit ẩn LSTM.
        n_features : Số features đầu vào (mặc định 1, thực tế = số cột preprocess).
    """
    name = name.upper()
    if name == "LSTM":
        return LSTMModel(input_size=n_features, hidden_size=hidden_size, output_size=1)
    elif name == "DLINEAR":
        return DLinearModel(seq_len=window, pred_len=1, channels=n_features)
    elif name == "NLINEAR":
        return NLinearModel(seq_len=window, pred_len=1, channels=n_features)
    else:
        raise ValueError(f"Tên model không hợp lệ: {name}. Chọn LSTM / DLinear / NLinear.")