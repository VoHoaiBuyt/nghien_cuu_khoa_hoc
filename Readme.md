# Hệ Thống Hỗ Trợ Ra Quyết Định Phân Tích và Dự Đoán Hành Vi Khách Hàng Trong Bán Hàng Đa Kênh

> **Đề tài Nghiên cứu Khoa học của Sinh viên**  
> **Đơn vị chủ trì:** Khoa Công nghệ Thông tin – Trường Đại học Nha Trang  
> **Thời gian thực hiện:** 13 tháng (07/2026 – 07/2027)  
> **Lĩnh vực nghiên cứu:** Khoa học kỹ thuật và công nghệ (Khai phá dữ liệu, Học máy & Hệ hỗ trợ ra quyết định – DSS)

---

## 1. Thông tin chung về đề tài

* **Tên đề tài:** Xây dựng hệ thống hỗ trợ ra quyết định phân tích và dự đoán hành vi khách hàng trong bán hàng đa kênh
* **Đơn vị chủ trì:** Khoa Công nghệ thông tin – Trường Đại học Nha Trang (Trưởng đơn vị: TS. Phạm Thị Thu Thúy)
* **Giảng viên hướng dẫn:** ThS. Nguyễn Thị Hương Lý (`lynth@ntu.edu.vn`)
* **Chủ nhiệm đề tài:** Hoàng Lê Thảo Nhi – Lớp 66.TTQL-1 (`nhi.hlt.66ttql@ntu.edu.vn`)

### Nhóm nghiên cứu & Phân công nhiệm vụ

| STT | Họ và tên | Lớp / Đơn vị | Vai trò | Nhiệm vụ chính |
|:---:|:---|:---|:---|:---|
| 1 | **Hoàng Lê Thảo Nhi** | 66.TTQL-1, Khoa CNTT | Chủ nhiệm đề tài | Quản lý chung; Khảo sát phương pháp (1.2); Chia tập dữ liệu Train/Test (2.5); Phân khúc khách hàng bằng K-Means (3.3); Đánh giá & lựa chọn mô hình (3.5, 3.6); Xây dựng bộ luật DSS, chiến lược marketing & cơ chế cảnh báo (4.1, 4.3, 4.4); Thiết kế kiến trúc tổng thể & tích hợp DSS (5.1, 5.4). |
| 2 | **Võ Hoài Buýt** | 65.TTQL-2, Khoa CNTT | Thành viên chính | Nghiên cứu khai phá dữ liệu, ML & khảo sát DSS (1.3, 1.4); Thu thập, làm sạch dữ liệu & xây dựng đặc trưng (2.1, 2.2, 2.4); Xây dựng mô hình nâng cao Random Forest, XGBoost (3.2); Phát triển chức năng dự đoán hành vi churn & xu hướng mua (4.2); Xây dựng bảng tổng hợp & trực quan hóa dữ liệu (5.2, 5.3); Đánh giá hiệu quả mô hình (6.1). |
| 3 | **Võ Thanh Tòng** | 65.TTQL-2, Khoa CNTT | Thành viên chính | Tìm hiểu hành vi khách hàng số (1.1); Chuẩn hóa & đồng bộ dữ liệu đa kênh (2.3); Xây dựng mô hình cơ bản Logistic Regression, Decision Tree (3.1); Tuning & tối ưu mô hình (3.4); Đánh giá hiệu quả hệ thống DSS (6.2). |

---

## 2. Mục tiêu và Phạm vi nghiên cứu

### 2.1. Mục tiêu đề tài
1. **Thu thập và xử lý dữ liệu** khách hàng từ môi trường bán hàng đa kênh (POS, Website, Facebook Fanpage, TikTok Shop, Shopee).
2. **Phân tích hành vi khách hàng** dựa trên dữ liệu giao dịch nhằm nhận diện đặc điểm, phân khúc (RFM) và xu hướng tiêu dùng.
3. **Xây dựng mô hình học máy dự đoán hành vi khách hàng** (dự đoán mua lặp lại, dự đoán khách hàng rời bỏ – Churn Prediction, phân khúc khách hàng K-Means).
4. **Phát triển Hệ thống hỗ trợ ra quyết định (DSS)** tích hợp kết quả phân tích, mô hình dự đoán và hệ luật cảnh báo, cung cấp bảng điều khiển trực quan (Dashboard) cho nhà quản trị.
5. **Đánh giá hiệu quả** của mô hình và hệ thống, đề xuất giải pháp tối ưu chiến lược marketing và chăm sóc khách hàng.

### 2.2. Đối tượng & Phạm vi nghiên cứu
* **Dữ liệu nghiên cứu:** Dữ liệu thực tế từ hệ thống quản lý bán hàng đa kênh (hơn 21.200 giao dịch, 19.100+ khách hàng, 1.600 biến thể sản phẩm, dữ liệu hoàn trả và danh mục hành chính 34 tỉnh/thành phố Việt Nam).
* **Phương pháp & Thuật toán:**
  * Phân tích hành vi & Phân cụm: Mô hình **RFM (Recency – Frequency – Monetary)** kết hợp thuật toán **K-Means Clustering**.
  * Mô hình dự đoán có giám sát: **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**.
  * Giải thích mô hình & Hỗ trợ ra quyết định: Luật quyết định DSS, cơ chế cảnh báo sớm khách hàng nguy cơ rời bỏ và nhận diện khách hàng tiềm năng.

---

## 3. Cấu trúc thư mục dự án

```text
e:\Nghien_cuu_khoa_hoc\
├── data/
│   ├── not_clean_data.xlsx                 # Bộ dữ liệu thô đa nguồn (8 sheets: orders, products, return_orders, provinces, districts, status, tags, users)
│   └── Cleaned_data/                       # Bộ dữ liệu sạch & đặc trưng phục vụ huấn luyện mô hình và DSS
│       ├── orders_cleaned.csv              # Dữ liệu giao dịch đa kênh đã làm sạch, đồng bộ kênh và chuẩn hóa 34 tỉnh/thành
│       ├── customer_rfm_features.csv       # Bảng đặc trưng cấp độ khách hàng (RFM, hành vi đa kênh, tỷ lệ hoàn/hủy) cho K-Means & DSS
│       ├── train_dataset.csv               # Tập huấn luyện bài toán Dự đoán khách mua lặp lại (Zero Data Leakage)
│       ├── test_dataset.csv                # Tập kiểm thử bài toán Dự đoán khách mua lặp lại
│       ├── train_churn_oot.csv             # Tập huấn luyện bài toán Dự đoán khách rời bỏ theo cửa sổ thời gian (Out-of-Time Churn)
│       └── test_churn_oot.csv              # Tập kiểm thử bài toán Dự đoán khách rời bỏ (Out-of-Time Churn)
├── notebook/
│   └── processing_data.ipynb               # Notebook thực thi quy trình làm sạch, chuẩn hóa đa kênh, Feature Engineering & chia tập Train/Test
└── Readme.md                               # Tài liệu tổng quan dự án
```

---

## 4. Quy trình xử lý dữ liệu và Xây dựng đặc trưng (`notebook/processing_data.ipynb`)

Quy trình tiền xử lý dữ liệu (Nội dung 2 của đề tài) được triển khai tự động và kiểm định chất lượng (Automated QA Audit) qua 6 giai đoạn:

1. **Khảo sát dữ liệu thô (Data Profiling - Công việc 2.1):**
   * Đọc và kiểm tra cấu trúc 8 sheets từ `data/not_clean_data.xlsx`.
2. **Làm sạch dữ liệu (Data Cleaning - Công việc 2.2):**
   * Loại bỏ cột rỗng 100% (`updated`), lọc bỏ đơn đã xóa (`status = 7`) và các đơn không có định danh khách hàng.
   * Chuẩn hóa số điện thoại Việt Nam (khôi phục số `0` ở đầu, xử lý mã quốc gia `84`, số máy bàn) và tạo khóa định danh `customer_id` (`KH_0xxxxxxxxx`).
   * Khắc phục lỗi font tiếng Việt trên tên người nhận và trích xuất đặc trưng số điện thoại đặt hộ/Cộng tác viên (`is_shared_phone`).
   * Nội suy giá trị đơn hàng (`total_price_cleaned`), xử lý tồn kho âm, giá âm, chuẩn hóa `cod_cleaned` và xử lý ngoại lai (Winsorization 99th percentile).
   * Gộp bảng sản phẩm theo `product_id` tránh nhân bản bản ghi khi kết nối giao dịch.
3. **Chuẩn hóa & Đồng bộ đa kênh (Data Standardization - Công việc 2.3):**
   * Quy hoạch hơn 40 nguồn bán hàng (`source`) về 5 nhóm kênh chuẩn (`channel_group`: *Facebook Fanpage, TikTok Shop, Shopee, Website, Unknown/Other*) và 3 dòng sản phẩm (*Đầm/Váy Thời trang, Nội y, Tổng hợp*).
   * Quy đổi chính xác 100% các mã tỉnh/thành cũ sang **34 tỉnh/thành phố mới** và phân vùng địa lý (*Miền Bắc, Miền Trung, Miền Nam*).
   * Chuẩn hóa định dạng thời gian ISO (`YYYY-MM-DD HH:MM:SS`) và tính thời gian xử lý đơn hàng (`processing_hours`).
4. **Xây dựng đặc trưng (Feature Engineering - Công việc 2.4):**
   * Tổng hợp bộ đặc trưng RFM (`recency_days`, `total_orders`, `completed_orders`, `monetary_total`, `avg_order_value`) và hành vi đa kênh (`num_channels_used`, `is_omnichannel`, `num_sources_used`, `is_multi_source`, `return_rate`, `cancel_rate`, `ad_order_ratio`).
5. **Chia tập dữ liệu Train/Test chống rò rỉ nhãn (Zero Data Leakage - Công việc 2.5):**
   * **Bài toán Dự đoán mua lặp lại (`train_dataset.csv` / `test_dataset.csv`):** Chỉ sử dụng đặc trưng từ đơn hàng đầu tiên (`first_order_*`) để dự đoán `is_repeat_customer`, loại bỏ hoàn toàn các biến cộng dồn tương lai.
   * **Bài toán Dự đoán khách hàng rời bỏ (`train_churn_oot.csv` / `test_churn_oot.csv`):** Tách cửa sổ quan sát (*Observation Window*) và cửa sổ dự đoán (*Prediction Window*) theo mốc thời gian `cutoff_date` để huấn luyện mô hình dự đoán `is_churn_oot`.

---

## 5. Nội dung và Lộ trình nghiên cứu (07/2026 – 07/2027)

| Giai đoạn | Nội dung nghiên cứu | Sản phẩm đầu ra | Thời gian |
|:---:|:---|:---|:---:|
| **Nội dung 1** | Phân tích bài toán và khảo sát phương pháp (Omnichannel, RFM, ML, DSS) | Báo cáo tổng quan & khảo sát phương pháp | 07/2026 – 09/2026 |
| **Nội dung 2** | Thu thập, làm sạch, chuẩn hóa dữ liệu đa kênh và xây dựng đặc trưng | Bộ dữ liệu sạch & tập Train/Test (`data/Cleaned_data/`) | 08/2026 – 11/2026 |
| **Nội dung 3** | Xây dựng, tối ưu và đánh giá các mô hình (Logistic Regression, Decision Tree, Random Forest, XGBoost, K-Means) | Các mô hình dự đoán tối ưu, kết quả phân khúc & báo cáo đánh giá | 11/2026 – 02/2027 |
| **Nội dung 4** | Xây dựng Hệ hỗ trợ ra quyết định (DSS), hệ luật gợi ý marketing & cơ chế cảnh báo sớm | Bộ luật DSS, module dự đoán hành vi & cảnh báo khách hàng rời bỏ | 02/2027 – 04/2027 |
| **Nội dung 5** | Thiết kế kiến trúc hệ thống, xây dựng Dashboard trực quan hóa & tích hợp DSS | Phần mềm Hệ thống DSS & Bảng điều khiển trực quan | 04/2027 – 05/2027 |
| **Nội dung 6** | Đánh giá hiệu quả mô hình dự đoán và hệ thống DSS; viết báo cáo tổng kết & bài báo khoa học | Báo cáo đánh giá, 01 bài báo Hội thảo khoa học, Báo cáo tổng kết đề tài | 05/2027 – 07/2027 |

---

## 6. Hướng dẫn cài đặt và Chạy dự án

### Yêu cầu môi trường
* Python `>= 3.10`
* Các thư viện chính: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `jupyter` (và `xgboost` cho giai đoạn huấn luyện mô hình).

### Cài đặt thư viện
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl xgboost jupyter
```

### Thực thi tiền xử lý dữ liệu
Mở và chạy trực tiếp notebook [notebook/processing_data.ipynb](notebook/processing_data.ipynb) trên VS Code hoặc Jupyter Notebook, hoặc chạy qua dòng lệnh:
```bash
jupyter nbconvert --to notebook --execute --inplace notebook/processing_data.ipynb
```

---

## 7. Sản phẩm khoa học dự kiến

1. **01 Hệ thống hỗ trợ ra quyết định (DSS)** trong phân tích và dự đoán hành vi khách hàng bán hàng đa kênh.
2. **01 Bảng tổng hợp và trực quan hóa dữ liệu (Dashboard)** hỗ trợ nhà quản trị theo dõi hành vi mua sắm, phân khúc RFM và cảnh báo khách hàng rời bỏ.
3. **01 Tài liệu hướng dẫn sử dụng hệ thống**.
4. **01 Bài báo cáo Hội thảo khoa học** cấp Khoa trở lên.
5. **01 Báo cáo tổng kết đề tài** nghiên cứu khoa học sinh viên.
