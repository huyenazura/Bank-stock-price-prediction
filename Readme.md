# 🏦  Xây dựng hệ thống Web hỗ trợ quyết định đầu tư và dự báo xu hướng nhóm cổ phiếu Ngân hàng Việt Nam dựa trên các mô hình học máy

Link github: https://github.com/huyenazura/Bank-stock-price-prediction.git

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy

python src/collect_data.py    # 1. Thu thập dữ liệu
python src/explore_data.py    # 2. EDA
python src/clean_data.py      # 3. Làm sạch
python src/build_model.py     # 4. Huấn luyện
python src/evaluate_model.py  # 5. Đánh giá + tín hiệu
streamlit run app.py          # 6. Giao diện

## Cấu trúc

├── src/                  # Pipeline (collect → clean → train → evaluate)
├── data/
│   ├── processed/        # Dữ liệu sạch
│   ├── models/           # Weights + scaler
│   └── eval_plots/       # Biểu đồ đánh giá
├── app.py                # Streamlit UI
└── requirements.txt
