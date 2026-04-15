import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Chuyển chuỗi 1D thành các cặp (X_window, y_next) để train."""

    def __init__(self, data: np.ndarray, window_size: int):
        self.data        = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]          # [window, 1]
        y = self.data[idx + self.window_size]                # [1]
        return x, y


def preprocess_data(df) -> np.ndarray:
    """
    Tính log return từ cột 'close', bỏ NaN.
    Trả về array shape [N, 1] — nhất quán với lúc train.
    """
    df = df.copy()
    df["close_log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["close_log_return"])
    return df[["close_log_return"]].values