import torch
import torch.nn as nn


# ── LSTM ─────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Moving Average + Decomposition (dùng cho DLinear) ────────────────────────
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg         = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back  = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x     = torch.cat([front, x, back], dim=1)
        x     = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


# ── DLinear ──────────────────────────────────────────────────────────────────
class DLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len=1, channels=1):
        super().__init__()
        self.decomp          = series_decomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend    = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend    = self.Linear_Trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1).squeeze(-1)   # [B, pred_len]


# ── NLinear ──────────────────────────────────────────────────────────────────
class NLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len=1):
        super().__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x        = x - seq_last
        x        = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return (x + seq_last).squeeze(-1)                         # [B, pred_len]