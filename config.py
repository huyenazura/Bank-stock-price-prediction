"""
config.py — Cấu hình toàn bộ hệ thống dự báo cổ phiếu ngân hàng Việt Nam.
"""

import torch
from datetime import datetime, timedelta

# ── Danh sách 30 mã ngân hàng Việt Nam ──────────────────────────────────────
SYMBOLS = [
    "VCB", "BID", "CTG", "MBB", "TCB",
    "ACB", "VPB", "STB", "HDB", "VIB",
    "MSB", "OCB", "TPB", "LPB", "SSB",
    "EIB", "SHB", "BAB", "VAB", "ABB",
    "BVB", "KLB", "NAB", "PGB", "SCB",
    "SEA", "SGN", "VBB", "WEB", "NVB",
]

BANK_NAMES = {
    "VCB": "Vietcombank",       "BID": "BIDV",
    "CTG": "VietinBank",        "MBB": "MB Bank",
    "TCB": "Techcombank",       "ACB": "ACB",
    "VPB": "VPBank",            "STB": "Sacombank",
    "HDB": "HDBank",            "VIB": "VIB",
    "MSB": "MSB",               "OCB": "OCB",
    "TPB": "TPBank",            "LPB": "LPBank",
    "SSB": "SeABank",           "EIB": "Eximbank",
    "SHB": "SHB",               "BAB": "BacA Bank",
    "VAB": "VietABank",         "ABB": "ABBank",
    "BVB": "BaoViet Bank",      "KLB": "KienLong Bank",
    "NAB": "Nam A Bank",        "PGB": "PG Bank",
    "SCB": "SCB",               "SEA": "SeABank",
    "SGN": "Saigon Bank",       "VBB": "VietBank",
    "WEB": "WEB",               "NVB": "NamViet Bank",
}

# ── Khoảng thời gian dữ liệu ─────────────────────────────────────────────────
END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

# ── Cấu hình mô hình ─────────────────────────────────────────────────────────
INPUT_WINDOWS = [1, 5, 21]          # 1 ngày, 1 tuần, 1 tháng
PRED_HORIZONS = {1: "1 ngày", 5: "1 tuần (5 phiên)", 21: "1 tháng (21 phiên)"}

HIDDEN_SIZE   = 64
EPOCHS        = 100     # pretrain VCB
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# ── Pretrain / Finetune ───────────────────────────────────────────────────────
# Pha 1: train đầy đủ 3 model trên mã đại diện → chọn model tốt nhất
# Pha 2: load weight tốt nhất → đóng băng backbone → finetune lớp cuối 29 mã
PRETRAIN_SYMBOL  = "VCB"    # mã đại diện để pretrain
FINETUNE_EPOCHS  = 20       # số epoch finetune lớp cuối (nhỏ hơn nhiều so với pretrain)
FINETUNE_LR      = 1e-4     # LR nhỏ để không phá vỡ features đã học

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Multiprocessing ───────────────────────────────────────────────────────────
NUM_WORKERS = 4

# ── NewsAPI ───────────────────────────────────────────────────────────────────
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"   # <-- thay bằng key thật của bạn
NEWS_LANGUAGE = "vi"
NEWS_PAGE_SIZE = 5