# ============================================================
# ĐỊNH NGHĨA MÔ HÌNH LTSF-Linear
# Ba mô hình: Linear, DLinear, NLinear
# ============================================================

import torch #
import torch.nn as nn 
'''torch là thư viện chính của PyTorch,
cung cấp các tensor và phép toán trên tensor,
tensor là cấu trúc dữ liệu cơ bản trong PyTorch,
tương tự như mảng numpy nhưng có thể chạy trên GPU để tăng tốc tính toán.
nn là module con của torch, cung cấp các lớp và hàm để xây dựng mạng nơ-ron, 
bao gồm các lớp như Linear, Conv2d, LSTM, v.v., cũng như các hàm mất mát và tối ưu hóa.'''


class LinearModel(nn.Module):
    """
    Linear đơn giản nhất:
    Ánh xạ thẳng từ chuỗi đầu vào → chuỗi dự đoán bằng 1 lớp Linear.
    """
    def __init__(self, input_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(input_len, pred_len)

    def forward(self, x):
        # x: (batch, input_len)
        return self.linear(x)


class DLinearModel(nn.Module):
    def __init__(self, input_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.linear_trend    = nn.Linear(input_len, pred_len)
        self.linear_seasonal = nn.Linear(input_len, pred_len)

        # Padding để giữ nguyên độ dài
        pad = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x):
        # x: (batch, input_len)
        x_3d = x.unsqueeze(1)                         # → (batch, 1, input_len)
        trend = self.avg_pool(x_3d).squeeze(1)        # → (batch, input_len)
        # Đảm bảo đúng độ dài sau padding
        trend    = trend[:, :x.size(1)]
        seasonal = x - trend

        out_trend    = self.linear_trend(trend)
        out_seasonal = self.linear_seasonal(seasonal)
        return out_trend + out_seasonal


class NLinearModel(nn.Module):
    """
    NLinear (Normalization Linear):
    Trừ đi giá trị cuối cùng để chuẩn hóa trước khi đưa vào Linear,
    sau đó cộng lại. Giúp mô hình ổn định hơn khi dữ liệu có drift.
    """
    def __init__(self, input_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(input_len, pred_len)

    def forward(self, x):
        # x: (batch, input_len)
        last_val = x[:, -1:]        # giá trị cuối cùng làm điểm neo
        x_norm   = x - last_val     # chuẩn hóa
        out      = self.linear(x_norm)
        return out + last_val       # cộng lại điểm neo


# Từ điển tiện dụng để lấy model theo tên
MODEL_REGISTRY = {
    "Linear":  LinearModel,
    "DLinear": DLinearModel,
    "NLinear": NLinearModel,
}