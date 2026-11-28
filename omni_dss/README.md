# Omni Customer DSS — lớp phân tích & hệ hỗ trợ ra quyết định

> **Đề tài NCKH sinh viên:** *"Xây dựng hệ thống hỗ trợ ra quyết định phân tích và dự đoán hành vi khách hàng trong bán hàng đa kênh"* — Khoa CNTT, Trường Đại học Nha Trang.

Thư mục `omni_dss/` (cùng `notebooks/`, `docs/` ở thư mục gốc) chứa **phần phân tích**: làm sạch & ẩn danh hóa dữ liệu, mô hình học máy, hệ hỗ trợ ra quyết định (DSS) và đánh giá. Tất cả chạy bằng Python và notebook, không cần máy chủ web. Các lệnh bên dưới chạy từ **thư mục gốc của repo**.

Ứng dụng web (FastAPI + React, phục vụ Nội dung 5) chưa nằm trong repo này; sẽ đưa vào ở một PR riêng.

**Dữ liệu:** code đọc `Cleaned_data/` ở thư mục gốc repo (không commit, xem `.gitignore`). Dữ liệu sạch hiện có trong repo nhóm ở `data/Cleaned_data/` **chưa ẩn danh hóa** (còn số điện thoại trong `customer_id`); nên sinh lại bản ẩn danh bằng `python -m omni_dss.ml.data_cleaning` (cần `OMNI_ANON_KEY`). Có thể trỏ thư mục khác bằng biến môi trường `CLEANED_DATA_DIR`.

## Nội dung ↔ mã nguồn

| Nội dung (thuyết minh) | Mã nguồn | Notebook |
|---|---|---|
| ND2 – Thu thập & tiền xử lý, ẩn danh hóa, đặc trưng | `omni_dss/ml/data_cleaning.py`, `feature_engineering.py` | `01_noi_dung_2_du_lieu` |
| ND3 – Mô hình dự đoán, phân cụm RFM | `omni_dss/train_pipeline.py`, `omni_dss/ml/train_advanced.py` | `02_noi_dung_3_mo_hinh` |
| ND3 – Bài toán A (mua lặp lại sau đơn đầu) | `omni_dss/ml/repeat_purchase.py` | `04_noi_dung_6_danh_gia` (mục 1.5) |
| **ND4 – Hệ hỗ trợ ra quyết định** (CV 4.1–4.4) | `omni_dss/ml/dss.py`, `predict.py`, `risk_rank.py` | **`03_noi_dung_4_dss`** |
| **ND6 – Đánh giá hệ thống** (CV 6.1–6.2) | `omni_dss/ml/evaluate_models.py`, `evaluate_dss.py`, `walk_forward.py` | **`04_noi_dung_6_danh_gia`** |

## Cấu trúc

```
omni_dss/
├── ml/
│   ├── config.py            # đường dẫn kết quả, nạp omni_dss/.env
│   ├── data_cleaning.py     # Raw_data/*.xlsx -> Cleaned_data/*.csv (ẩn danh hóa HMAC)
│   ├── feature_engineering.py, train_advanced.py, explain.py, evaluation.py
│   ├── repeat_purchase.py   # Bài toán A
│   ├── walk_forward.py      # đánh giá trượt theo tuần cho Bài toán B
│   ├── risk_rank.py         # thang percentile rủi ro
│   ├── predict.py           # CV 4.2 – chấm điểm khách tại mốc thời gian bất kỳ (+ SHAP)
│   ├── dss.py               # CV 4.1/4.3/4.4 – bộ luật, 5 kịch bản, cảnh báo sớm
│   ├── evaluate_models.py   # CV 6.1
│   └── evaluate_dss.py      # CV 6.2
├── train_pipeline.py        # chạy toàn bộ: huấn luyện + đánh giá
├── saved_models/            # mô hình (.pkl, không commit) và file kết quả JSON
└── tests/
notebooks/01–04              # mỗi file = một Nội dung của thuyết minh, có kết quả sẵn
docs/                        # thuyết minh đề tài, logo
```

## Chạy lại

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r omni_dss/requirements-dev.txt

# (Chỉ khi cần làm sạch lại từ dữ liệu thô) đặt Raw_data/not_clean_data.xlsx và OMNI_ANON_KEY trong omni_dss/.env
python -m omni_dss.ml.data_cleaning

python omni_dss/train_pipeline.py       # ~1,5 phút: mô hình + toàn bộ file đánh giá trong omni_dss/saved_models/
python -m pytest omni_dss/tests -q
jupyter notebook notebooks/
```

Dữ liệu thô, dữ liệu sạch, mô hình `.pkl`, danh sách khách đã chấm điểm và `omni_dss/.env` **không** được commit (xem `.gitignore`).

## File kết quả (omni_dss/saved_models/)

| File | Nội dung |
|---|---|
| `model_metrics.json` | So sánh 4 mô hình churn, RFM K-Means |
| `risk_reference.json` | Thang percentile rủi ro (phân bố điểm trên tập kiểm tra) |
| `dss_evaluation.json` | CV 6.2 – hiệu quả cảnh báo, theo luật/kịch bản, độ nhạy ngưỡng, case study |
| `model_evaluation.json` | CV 6.1 – khoảng tin cậy AUC, so với đối chứng Recency, calibration |
| `walk_forward.json` | Đánh giá theo 4 tuần liên tiếp (mô hình + độ ổn định cảnh báo) |
| `repeat_metrics.json` | Bài toán A theo các cửa sổ W ngày |
| `early_warning_summary.json` | Tổng hợp cảnh báo sớm tại ngày cuối dữ liệu (notebook 03) |

## Nhóm nghiên cứu

- **Cán bộ hướng dẫn:** ThS. Nguyễn Thị Hương Lý
- **Chủ nhiệm:** Hoàng Lê Thảo Nhi (66.TTQL-1)
- **Thành viên:** Võ Hoài Buýt (65.TTQL-2), Võ Thanh Tòng (65.TTQL-2)

Thuyết minh: [`docs/THUYET_MINH_DE_TAI_NCKH.md`](../docs/THUYET_MINH_DE_TAI_NCKH.md)
