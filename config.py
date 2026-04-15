import torch
from datetime import datetime, timedelta

# ── Danh sách 30 mã ngân hàng Việt Nam ──────────────────────────────────────
SYMBOLS = [
    "VCB", "BID", "CTG", "MBB", "TCB",
    "ACB", "VPB", "STB", "HDB", "VIB",
    "MSB", "OCB", "TPB", "LPB", "SSB",
    "EIB", "SHB", "BAB", "VAB", "ABB",
    "BVB", "KLB", "NAB", "PGB", "SCB",
    "SEA", "SGN", "VBB", "VCB", "WEB",
]

# ── Khoảng thời gian dữ liệu ─────────────────────────────────────────────────
END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

# ── Cấu hình mô hình ─────────────────────────────────────────────────────────
INPUT_WINDOWS = [7, 30, 120]
HIDDEN_SIZE   = 64 #Input (1) → LSTM hidden (64) → Linear → Output (1)
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Multiprocessing ───────────────────────────────────────────────────────────
NUM_WORKERS = 4   # Số process song song khi train (chỉnh theo số CPU)