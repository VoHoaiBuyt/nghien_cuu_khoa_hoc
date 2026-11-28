"""
Làm sạch & chuẩn hóa dữ liệu đa kênh (Nội dung 2 – công việc 2.1–2.5 của thuyết minh).

Tái hiện quy trình của notebook nhóm (repo VoHoaiBuyt/nghien_cuu_khoa_hoc, notebook/processing_data.ipynb)
thành module chạy được bằng một lệnh, để dự án tự sinh lại Cleaned_data/ từ dữ liệu thô:

    python -m omni_dss.ml.data_cleaning --raw Raw_data/not_clean_data.xlsx --out Cleaned_data

Cần biến môi trường OMNI_ANON_KEY (khóa bí mật ẩn danh hóa, khai báo trong omni_dss/.env).

Đầu vào : Excel 8 sheet (orders, products, huonglyc return_orders, provinces, huonglyc districts, status, tags, users).
Đầu ra  : orders_cleaned.csv, customer_rfm_features.csv, train/test_dataset.csv (bài toán mua lặp lại),
          train/test_churn_oot.csv (bài toán churn out-of-time).

Các bước (giữ nguyên logic notebook để kết quả trùng khớp — đã kiểm chứng):
  1. Đơn hàng: chuẩn hóa số điện thoại, loại đơn đã xóa (status 7) / thiếu SĐT, tạo customer_id ẨN DANH
     = "KH_" + base32(HMAC-SHA256(OMNI_ANON_KEY, SĐT))[:10] — không suy ngược được nếu không có khóa.
  2. Tên người nhận: sửa lỗi font, phát hiện SĐT dùng chung (đặt hộ / cộng tác viên).
  3. Địa lý: suy tỉnh từ huyện, bổ khuyết theo lịch sử khách, quy đổi 63 tỉnh cũ -> 34 tỉnh mới, gán vùng miền.
  4. Sản phẩm: tồn kho âm -> 0, giá <= 0 -> trung vị theo danh mục, gộp theo product_id.
  5. Giá trị đơn: COD âm -> 0, nội suy giá 0đ (giá sản phẩm -> COD -> trung vị cho đơn hoàn thành), Winsorize p99.
  6. Hoàn trả & users: bỏ trùng, nhóm lý do hoàn; bỏ cột SĐT của nhân viên.
  7. Chuẩn hóa kênh (5 nhóm), dòng sản phẩm, thời gian, nhóm trạng thái; ghép sản phẩm & hoàn trả; 0 giá trị thiếu.
  8. Đặc trưng khách hàng (RFM, đa kênh) và hai bộ train/test.

QUYỀN RIÊNG TƯ (ẩn danh hóa – Vấn đề 5 của nhận xét đợt 1):
  - Tệp thành phẩm KHÔNG chứa số điện thoại, họ tên (bỏ bill_phone_number, bill_full_name); customer_id là mã giả danh.
  - Sau khi xuất, mọi cột văn bản được quét: còn chuỗi giống số điện thoại -> báo lỗi, dừng.
  - Bảng ánh xạ mã giả danh -> SĐT (để CSKH liên hệ khi cần) chỉ ghi cạnh dữ liệu thô (Raw_data/, nằm trong .gitignore).
"""

import argparse
import base64
import hashlib
import hmac
import os
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DELETED_STATUS = 7
COMPLETED_STATUSES = (3, 16)
RETURNED_STATUSES = (4, 5, 15)
CANCELLED_STATUS = 6
COMPLETED_PRICE_FALLBACK = 300000.0  # trung vị giá trị đơn toàn bộ dữ liệu
WEBSITE_SOURCES = ("AMYRA.shopvn", "AMYRA.nt")
UNKNOWN_PROVINCE = "Chưa xác định / Khác"
ANON_KEY_ENV = "OMNI_ANON_KEY"
PII_COLUMNS = ("bill_phone_number", "bill_full_name")
MAPPING_FILE = "customer_id_map.csv"
PHONE_LIKE = re.compile(r"(?<![\d.])0\d{9,10}(?![\d.])")  # SĐT VN 10–11 số, không phải phần của số thực

# Quy đổi chính thức các tỉnh cũ sáp nhập -> tỉnh mới (34 tỉnh/thành từ 2025)
OLD_TO_NEW_PROVINCE: Dict[int, int] = {
    104: 217,  # Vĩnh Phúc -> Phú Thọ
    107: 103,  # Hải Dương -> Hải Phòng
    111: 117,  # Hà Nam -> Ninh Bình
    113: 117,  # Nam Định -> Ninh Bình
    115: 109,  # Thái Bình -> Hưng Yên
    201: 211,  # Hà Giang -> Tuyên Quang
    207: 215,  # Bắc Kạn -> Thái Nguyên
    213: 205,  # Yên Bái -> Lào Cai
    221: 106,  # Bắc Giang -> Bắc Ninh
    305: 217,  # Hòa Bình -> Phú Thọ
    407: 409,  # Quảng Bình -> Quảng Trị
    503: 501,  # Quảng Nam -> Đà Nẵng
    507: 603,  # Bình Định -> Gia Lai
    509: 605,  # Phú Yên -> Đắk Lắk
    601: 505,  # Kon Tum -> Quảng Ngãi
    606: 607,  # Đắk Nông -> Lâm Đồng
    705: 511,  # Ninh Thuận -> Khánh Hòa
    707: 713,  # Bình Phước -> Đồng Nai
    711: 701,  # Bình Dương -> Hồ Chí Minh
    715: 607,  # Bình Thuận -> Lâm Đồng
    717: 701,  # Bà Rịa - Vũng Tàu -> Hồ Chí Minh
    801: 709,  # Long An -> Tây Ninh
    807: 803,  # Tiền Giang -> Đồng Tháp
    811: 809,  # Bến Tre -> Vĩnh Long
    813: 805,  # Kiên Giang -> An Giang
    816: 815,  # Hậu Giang -> Cần Thơ
    817: 809,  # Trà Vinh -> Vĩnh Long
    819: 815,  # Sóc Trăng -> Cần Thơ
    821: 823,  # Bạc Liêu -> Cà Mau
}

# Lỗi font phổ biến trong tên người nhận (ký tự có dấu bị thay bằng "?"), áp dụng theo thứ tự
NAME_FIXES = [
    (r"nguy\?n", "nguyễn"), (r"th\?", "thị"), (r"ph\?m", "phạm"),
    (r"tr\?n", "trần"), (r"l\?", "lê"), (r"v\?", "vũ"), (r"h\?", "hồ"),
    (r"đ\?ng", "đặng"), (r"b\?i", "bùi"), (r"đ\?", "đỗ"),
    (r"ho\?ng", "hoàng"), (r"ng\?c", "ngọc"), (r"h\?\?ng", "hương"),
    (r"th\?o", "thảo"), (r"t\?", "tự"), (r"th\?\?ng", "thương"),
    (r"d\?ng", "dũng"), (r"h\?i", "hải"), (r"qu\?nh", "quỳnh"), (r"tu\?n", "tuấn"),
]


def _mode(s: pd.Series, default=np.nan):
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else default


# ---------------------------------------------------------------------------------------------- hàm làm sạch đơn lẻ
def clean_phone_number(val) -> str:
    """Chuẩn hóa SĐT Việt Nam: bỏ ký tự phi số, 84/084 -> 0, khôi phục số 0 đầu bị mất khi Excel lưu dạng số."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    if not s:
        return np.nan
    if len(s) == 12 and s.startswith("084"):
        s = "0" + s[3:]
    elif len(s) == 11 and s.startswith("84"):
        s = "0" + s[2:]
    elif len(s) == 9:
        s = "0" + s
    elif len(s) == 10 and s.startswith("2"):  # số máy bàn mất số 0 đầu
        s = "0" + s
    elif not s.startswith("0"):
        s = "0" + s
    return s


def pseudonymize(phone: str, key: bytes) -> str:
    """Mã khách hàng giả danh ổn định: KH_ + 10 ký tự base32 của HMAC-SHA256 (chữ A–Z, số 2–7 -> không giống SĐT)."""
    digest = hmac.new(key, phone.encode("utf-8"), hashlib.sha256).digest()
    return "KH_" + base64.b32encode(digest).decode("ascii")[:10]


def load_anon_key() -> bytes:
    import omni_dss.ml.config  # noqa: F401  (nạp omni_dss/.env vào biến môi trường)
    key = os.environ.get(ANON_KEY_ENV, "")
    if len(key) < 16:
        raise SystemExit(f"Thiếu khóa ẩn danh hóa: đặt {ANON_KEY_ENV} (>= 16 ký tự, giữ bí mật, không đổi giữa các lần chạy) "
                         "trong omni_dss/.env. Tạo khóa: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    return key.encode("utf-8")


def find_phone_like(frames: Dict[str, pd.DataFrame], skip: Iterable[str] = ()) -> Dict[str, int]:
    """Đếm ô chứa chuỗi giống SĐT trong các cột văn bản (kiểm tra cuối trước khi xuất)."""
    hits = {}
    for name, df in frames.items():
        for c in df.columns:
            if c in skip or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
                continue
            n = int(df[c].astype(str).str.contains(PHONE_LIKE).sum())
            if n:
                hits[f"{name}.{c}"] = n
    return hits


def clean_recipient_name(name) -> str:
    if pd.isna(name):
        return ""
    s = re.sub(r"\s+", " ", str(name).lower().strip())
    for pat, rep in NAME_FIXES:
        s = re.sub(pat, rep, s)
    return s


def map_channel_group(source) -> str:
    """Gộp hơn 40 nguồn bán về 5 nhóm kênh."""
    if pd.isna(source):
        return "Unknown/Other"
    s = str(source).strip()
    if s.lower() == "tiktok":
        return "TikTok Shop"
    if s.lower() == "shopee":
        return "Shopee"
    if s in WEBSITE_SOURCES:
        return "Website"
    return "Facebook Fanpage"


def map_product_line(source) -> str:
    if pd.isna(source):
        return "Tổng hợp"
    s = str(source).lower()
    if any(k in s for k in ["nội y", "đồ lót", "lót"]):
        return "Nội y"
    if any(k in s for k in ["váy", "đầm", "fashion", "dressy", "thời trang", "boutique", "closet"]):
        return "Đầm/Váy Thời trang"
    return "Tổng hợp"


def standardize_return_reason(reason) -> str:
    if pd.isna(reason):
        return "Không có lý do"
    r = str(reason).lower()
    if "khách hàng" in r:
        return "Do khách hàng"
    if "đơn vị vc" in r or "vận chuyển" in r:
        return "Do đơn vị vận chuyển"
    return "Lỗi sản phẩm/Shop"


def status_group(status, order_pk, returned_ids) -> str:
    if status in RETURNED_STATUSES or order_pk in returned_ids:
        return "Returned"
    if status in COMPLETED_STATUSES:
        return "Completed"
    if status == CANCELLED_STATUS:
        return "Cancelled"
    return "In_Progress"


# ---------------------------------------------------------------------------------------------- các bước theo bảng
def clean_customers_and_names(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 1–2: SĐT, lọc đơn, khóa khách hàng, tên người nhận, SĐT dùng chung.
    customer_id ở bước này là KHÓA NỘI BỘ "KH_" + SĐT (giữ đúng thứ tự sắp xếp, nên cách chia train/test và mọi
    kết quả giữ nguyên như trước khi ẩn danh hóa). run() thay bằng mã giả danh trước khi xuất — không tệp nào chứa khóa này.
    """
    o = orders.copy()
    o["bill_phone_number"] = o["bill_phone_number"].apply(clean_phone_number)
    o = o.drop(columns=[c for c in ["updated"] if c in o.columns])  # cột rỗng 100%
    o = o[(o["status"] != DELETED_STATUS) & (o["bill_phone_number"].notnull())].copy()
    o["customer_id"] = "KH_" + o["bill_phone_number"]

    o["clean_name"] = o["bill_full_name"].apply(clean_recipient_name)
    name_mode = o[o["clean_name"] != ""].groupby("customer_id")["clean_name"].agg(lambda x: _mode(x, "")).to_dict()
    o["clean_name"] = o["clean_name"].replace("", np.nan).fillna(o["customer_id"].map(name_mode)).fillna("khách hàng ẩn danh")
    o["bill_full_name"] = o["clean_name"]
    distinct = o[o["clean_name"] != "khách hàng ẩn danh"].groupby("customer_id")["clean_name"].nunique().to_dict()
    o["num_distinct_recipients"] = o["customer_id"].map(distinct).fillna(1).astype(int)
    o["is_shared_phone"] = (o["num_distinct_recipients"] >= 2).astype(int)
    return o


def standardize_geography(o: pd.DataFrame, provinces: pd.DataFrame, districts: pd.DataFrame) -> pd.DataFrame:
    """Bước 3: tỉnh/huyện -> 34 tỉnh mới và vùng miền."""
    o = o.copy()
    dist_to_prov = dict(zip(districts["id_districts"], districts["province_id"]))
    o["raw_province_id"] = o["province_id"].fillna(o["district_id"].map(dist_to_prov))

    cust_dist = o.dropna(subset=["district_id"]).groupby("customer_id")["district_id"].agg(_mode).to_dict()
    o["district_id"] = o["district_id"].fillna(o["customer_id"].map(cust_dist)).fillna(-1).astype(int)

    cust_prov = o.dropna(subset=["raw_province_id"]).groupby("customer_id")["raw_province_id"].agg(_mode).to_dict()
    o["raw_province_id"] = o["raw_province_id"].fillna(o["customer_id"].map(cust_prov)).fillna(-1)

    o["province_id"] = o["raw_province_id"].replace(OLD_TO_NEW_PROVINCE).astype(int)
    o["province_name"] = o["province_id"].map(dict(zip(provinces["id"], provinces["name"]))).fillna(UNKNOWN_PROVINCE)
    o["region_type"] = o["province_id"].map(dict(zip(provinces["id"], provinces["region_type"]))).fillna("Chưa xác định")
    o["district_name"] = o["district_id"].map(dict(zip(districts["id_districts"], districts["name"]))).fillna("Chưa xác định")
    return o


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Bước 4: tồn kho âm, giá <= 0, gộp theo product_id (tránh nhân bản đơn khi ghép)."""
    p = products.copy()
    p["is_out_of_stock"] = (p["remain_quantity"] <= 0).astype(int)
    p["remain_quantity"] = np.maximum(0, p["remain_quantity"])
    cat_median = p[p["retail_price"] > 0].groupby("categories_id")["retail_price"].median()
    for cat_id, med in cat_median.items():
        p.loc[(p["categories_id"] == cat_id) & (p["retail_price"] <= 0), "retail_price"] = med
    return p.groupby("product_id").agg(
        product_name=("name", "first"),
        display_id=("display_id", "first"),
        categories_id=("categories_id", "first"),
        categories_name=("categories_name", "first"),
        retail_price=("retail_price", "mean"),
        remain_quantity=("remain_quantity", "sum"),
        num_variations=("variation_id", "count"),
    ).reset_index()


def clean_order_values(o: pd.DataFrame, products_agg: pd.DataFrame) -> pd.DataFrame:
    """Bước 5: COD, nội suy giá trị đơn 0đ, cờ quà tặng, Winsorize p99."""
    o = o.copy()
    o["is_negative_cod"] = (o["cod"] < 0).astype(int)
    o["cod_cleaned"] = np.maximum(0, o["cod"].fillna(0))
    o["quantity"] = o["quantity"].fillna(1.0)
    o["product_id"] = o["product_id"].fillna("UNKNOWN_PRODUCT")
    o["is_from_ad"] = o["ad_id"].notnull().astype(int)
    o["assigning_care_id"] = o["assigning_care_id"].fillna(-1).astype(int)

    imputed = o["quantity"] * o["product_id"].map(dict(zip(products_agg["product_id"], products_agg["retail_price"]))).fillna(0)
    price = o["total_price"].astype(float).copy()
    m = (price <= 0) & (imputed > 0)                       # bậc 1: giá sản phẩm × số lượng
    price.loc[m] = imputed[m]
    m = (price <= 0) & (o["cod_cleaned"] > 0)              # bậc 2: tiền COD
    price.loc[m] = o["cod_cleaned"][m]
    m = (price <= 0) & (o["status"].isin(COMPLETED_STATUSES))  # bậc 3: đơn hoàn thành vẫn 0đ -> trung vị
    price.loc[m] = COMPLETED_PRICE_FALLBACK
    o["total_price_cleaned"] = price
    o["is_gift_or_free"] = (o["total_price_cleaned"] == 0).astype(int)

    o["total_price_capped"] = o["total_price_cleaned"].clip(upper=float(o["total_price_cleaned"].quantile(0.99)))
    o["quantity_capped"] = o["quantity"].clip(upper=float(o["quantity"].quantile(0.99)))
    return o


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.drop_duplicates(subset=["order_id"]).copy()
    r["detail"] = r["detail"].fillna("Không có chi tiết")
    r["reason_group"] = r["reason"].apply(standardize_return_reason)
    return r


def standardize_and_merge(o: pd.DataFrame, status: pd.DataFrame, products_agg: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Bước 7: kênh, thời gian, trạng thái; ghép sản phẩm & hoàn trả; điền mọi giá trị thiếu."""
    o = o.copy()
    o["channel_group"] = o["source"].apply(map_channel_group)
    o["channel_product_line"] = o["source"].apply(map_product_line)

    o["inserted_at"] = pd.to_datetime(o["inserted_at"])
    o["updated_at"] = pd.to_datetime(o["updated_at"])
    o["processing_hours"] = np.maximum(0, (o["updated_at"] - o["inserted_at"]).dt.total_seconds() / 3600.0)
    o["order_year"] = o["inserted_at"].dt.year
    o["order_month"] = o["inserted_at"].dt.month
    o["order_day"] = o["inserted_at"].dt.day
    o["order_dayofweek"] = o["inserted_at"].dt.day_of_week
    o["order_hour"] = o["inserted_at"].dt.hour
    o["is_weekend"] = o["order_dayofweek"].isin([5, 6]).astype(int)

    o["status_name"] = o["status"].map(dict(zip(status["id"], status["name"]))).fillna("Khác")
    returned_ids = set(returns["order_id"])
    o["status_group"] = [status_group(s, pk, returned_ids) for s, pk in zip(o["status"], o["id"])]
    o["is_returned"] = (o["status_group"] == "Returned").astype(int)

    o = o.merge(products_agg, on="product_id", how="left")
    o["product_name"] = o["product_name"].fillna("UNKNOWN_PRODUCT")
    o["display_id"] = o["display_id"].fillna("UNKNOWN_DISPLAY")
    o["categories_id"] = o["categories_id"].fillna(-1).astype(int)
    o["categories_name"] = o["categories_name"].fillna("Chưa phân loại")
    o["retail_price"] = o["retail_price"].fillna(o["total_price_cleaned"])
    o["remain_quantity"] = o["remain_quantity"].fillna(0).astype(int)
    o["num_variations"] = o["num_variations"].fillna(1).astype(int)

    info = returns[["order_id", "reason_group", "detail"]].rename(
        columns={"order_id": "id", "reason_group": "return_reason", "detail": "return_detail"})
    o = o.merge(info, on="id", how="left")
    o["page_id"] = o["page_id"].fillna(-1).astype(int)
    o["ad_id"] = o["ad_id"].fillna(-1).astype(int)
    o["district_id"] = o["district_id"].fillna(-1).astype(int)
    o["district_name"] = o["district_name"].fillna("Chưa xác định")
    o["source"] = o["source"].fillna("Unknown")
    o["return_reason"] = np.where(o["is_returned"] == 1, o["return_reason"].fillna("Hoàn trả - Không ghi nhận chi tiết"), "Không hoàn trả")
    o["return_detail"] = np.where(o["is_returned"] == 1, o["return_detail"].fillna("Không có chi tiết"), "Không hoàn trả")
    return o.drop(columns=["clean_name"])


def clean_orders(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bước 1–7: bảng đơn hàng sạch (0 giá trị thiếu). Còn khóa nội bộ & cột SĐT/tên — run() ẩn danh trước khi xuất."""
    products_agg = clean_products(sheets["products"])
    returns = clean_returns(sheets["huonglyc return_orders"])
    o = clean_customers_and_names(sheets["orders"])
    o = standardize_geography(o, sheets["provinces"], sheets["huonglyc districts"])
    o = clean_order_values(o, products_agg)
    o = standardize_and_merge(o, sheets["status"], products_agg, returns)
    assert o.isnull().sum().sum() == 0, "orders_cleaned vẫn còn giá trị thiếu"
    return o


# ---------------------------------------------------------------------------------------------- đặc trưng & tập dữ liệu
def build_customer_features(o: pd.DataFrame) -> pd.DataFrame:
    """Bảng đặc trưng khách hàng toàn kỳ (RFM, đa kênh) — dùng cho phân cụm K-Means."""
    snapshot = o["inserted_at"].max() + pd.Timedelta(days=1)
    base = o.groupby("customer_id").agg(
        total_orders=("id", "count"),
        first_order_date=("inserted_at", "min"),
        last_order_date=("inserted_at", "max"),
        total_items_bought=("quantity", "sum"),
        avg_quantity_per_order=("quantity", "mean"),
        max_order_value=("total_price_cleaned", "max"),
        ad_order_ratio=("is_from_ad", "mean"),
        num_channels_used=("channel_group", "nunique"),
        num_sources_used=("source", "nunique"),
        num_categories_bought=("categories_name", "nunique"),
        primary_channel=("channel_group", lambda x: _mode(x, "Unknown/Other")),
        region_type=("region_type", lambda x: _mode(x, "Chưa xác định")),
        num_distinct_recipients=("num_distinct_recipients", "first"),
        is_shared_phone=("is_shared_phone", "first"),
    )
    by_status = lambda g, **agg: o[o["status_group"] == g].groupby("customer_id").agg(**agg)  # noqa: E731
    c = base.join([
        by_status("Completed", completed_orders=("id", "count"), monetary_total=("total_price_cleaned", "sum")),
        by_status("Returned", return_orders_count=("id", "count")),
        by_status("Cancelled", cancel_orders_count=("id", "count")),
    ], how="left")
    for col in ("completed_orders", "return_orders_count", "cancel_orders_count"):
        c[col] = c[col].fillna(0).astype(int)
    c["monetary_total"] = c["monetary_total"].fillna(0.0)
    c["avg_order_value"] = np.where(c["completed_orders"] > 0, c["monetary_total"] / c["completed_orders"], 0.0)
    c["recency_days"] = (snapshot - c["last_order_date"]).dt.days
    c["customer_tenure_days"] = (c["last_order_date"] - c["first_order_date"]).dt.days
    c["return_rate"] = c["return_orders_count"] / c["total_orders"]
    c["cancel_rate"] = c["cancel_orders_count"] / c["total_orders"]
    c["is_omnichannel"] = (c["num_channels_used"] >= 2).astype(int)
    c["is_multi_source"] = (c["num_sources_used"] >= 2).astype(int)
    c["is_repeat_customer"] = (c["total_orders"] >= 2).astype(int)
    # Cờ MÔ TẢ theo quy tắc nghiệp vụ, tính trên toàn kỳ (đổi tên từ is_churn_behavioral theo nhận xét đợt 1, Vấn đề 2).
    # KHÔNG phải nhãn dự báo và KHÔNG dùng làm đặc trưng mô hình (kiểm thử: test_churn_rule_flag_not_a_model_feature).
    c["churn_rule_flag"] = ((c["recency_days"] > 21) | ((c["return_rate"] >= 0.5) & (c["total_orders"] >= 2))).astype(int)
    c = c.reset_index()
    assert ((c["num_channels_used"] == 1) & (c["is_omnichannel"] == 1)).sum() == 0
    assert c.isnull().sum().sum() == 0
    return c


def _scaled_split(features: pd.DataFrame, target: pd.Series, feat_cols) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia 80/20 phân tầng, StandardScaler fit trên train; giữ customer_id đầu, target cuối."""
    tr_idx, te_idx = train_test_split(features.index, test_size=0.20, random_state=RANDOM_STATE, stratify=target)
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(features.loc[tr_idx, feat_cols]), columns=feat_cols, index=tr_idx)
    Xte = pd.DataFrame(scaler.transform(features.loc[te_idx, feat_cols]), columns=feat_cols, index=te_idx)
    return (pd.concat([features.loc[tr_idx, ["customer_id"]], Xtr, target.loc[tr_idx]], axis=1),
            pd.concat([features.loc[te_idx, ["customer_id"]], Xte, target.loc[te_idx]], axis=1))


def build_repeat_purchase_datasets(o: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bài toán 1: dự đoán mua lặp lại chỉ từ đơn hàng đầu tiên (không dùng thông tin toàn kỳ)."""
    first = o.sort_values(["customer_id", "inserted_at", "id"]).reset_index(drop=True).groupby("customer_id").first().reset_index()
    n_orders = o.groupby("customer_id")["id"].count().to_dict()
    y = first["customer_id"].map(lambda c: int(n_orders[c] >= 2)).rename("target")

    f = pd.DataFrame({"customer_id": first["customer_id"]})
    f["first_order_total_price_log"] = np.log1p(first["total_price_cleaned"])
    f["first_order_retail_price_log"] = np.log1p(first["retail_price"])
    f["first_order_quantity"] = first["quantity"]
    f["first_order_price_ratio"] = first["total_price_cleaned"] / (first["quantity"] * first["retail_price"] + 1.0)
    f["first_order_cod_ratio"] = first["cod_cleaned"] / (first["total_price_cleaned"] + 1.0)
    f["first_order_processing_hours_log"] = np.log1p(first["processing_hours"])
    f["first_order_hour"] = first["inserted_at"].dt.hour
    f["first_order_dayofweek"] = first["inserted_at"].dt.day_of_week
    f["first_order_day"] = first["inserted_at"].dt.day
    f["first_order_month"] = first["inserted_at"].dt.month
    f["first_order_is_weekend"] = first["inserted_at"].dt.day_of_week.isin([5, 6]).astype(int)
    f["first_order_is_from_ad"] = first["is_from_ad"]
    f["first_order_num_variations"] = first["num_variations"]
    for g, col in (("Completed", "first_order_is_completed"), ("Returned", "first_order_is_returned"), ("Cancelled", "first_order_is_cancelled")):
        f[col] = (first["status_group"] == g).astype(int)
    f = pd.concat([f, pd.get_dummies(first[["channel_group", "channel_product_line", "region_type"]],
                                      prefix=["channel", "prod_line", "region"], dtype=int)], axis=1)

    # Frequency encoding tính trên train để tránh rò rỉ -> cần chỉ số train trước
    tr_idx, _ = train_test_split(f.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    prov_freq = first.loc[tr_idx, "province_id"].value_counts(normalize=True).to_dict()
    cat_freq = first.loc[tr_idx, "categories_id"].value_counts(normalize=True).to_dict()
    f["province_order_freq"] = first["province_id"].map(prov_freq).fillna(0)
    f["category_order_freq"] = first["categories_id"].map(cat_freq).fillna(0)
    return _scaled_split(f, y, [c for c in f.columns if c != "customer_id"])


def build_churn_oot_datasets(o: pd.DataFrame, cutoff_quantile: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bài toán 2: churn out-of-time — đặc trưng trong cửa sổ quan sát, nhãn = không mua lại sau cutoff."""
    cutoff = o["inserted_at"].quantile(cutoff_quantile)
    obs, later = o[o["inserted_at"] < cutoff], o[o["inserted_at"] >= cutoff]
    ids = obs["customer_id"].unique()
    returned = set(later["customer_id"].unique())
    y = pd.Series([int(c not in returned) for c in ids], index=ids, name="target")

    base = obs.groupby("customer_id").agg(
        obs_total_orders=("id", "count"),
        obs_first_date=("inserted_at", "min"),
        obs_last_date=("inserted_at", "max"),
        obs_processing_hours_mean=("processing_hours", "mean"),
        obs_ad_order_ratio=("is_from_ad", "mean"),
        obs_num_channels=("channel_group", "nunique"),
        obs_primary_channel=("channel_group", lambda x: _mode(x, "Unknown/Other")),
        obs_region_type=("region_type", lambda x: _mode(x, "Chưa xác định")),
    )
    by_status = lambda g, **agg: obs[obs["status_group"] == g].groupby("customer_id").agg(**agg)  # noqa: E731
    d = base.join([
        by_status("Completed", obs_completed_orders=("id", "count"), obs_monetary_total=("total_price_cleaned", "sum")),
        by_status("Returned", obs_return_orders_count=("id", "count")),
        by_status("Cancelled", obs_cancel_orders_count=("id", "count")),
    ], how="left")
    for col in ("obs_completed_orders", "obs_return_orders_count", "obs_cancel_orders_count"):
        d[col] = d[col].fillna(0).astype(int)
    d["obs_monetary_total"] = d["obs_monetary_total"].fillna(0.0)
    d["obs_avg_order_value"] = np.where(d["obs_completed_orders"] > 0, d["obs_monetary_total"] / d["obs_completed_orders"], 0.0)
    d["obs_recency_days"] = (cutoff - d["obs_last_date"]).dt.total_seconds() / 86400.0
    d["obs_tenure_days"] = (d["obs_last_date"] - d["obs_first_date"]).dt.total_seconds() / 86400.0
    d["obs_return_rate"] = d["obs_return_orders_count"] / d["obs_total_orders"]
    d["obs_cancel_rate"] = d["obs_cancel_orders_count"] / d["obs_total_orders"]
    d["obs_is_omnichannel"] = (d["obs_num_channels"] >= 2).astype(int)
    d["obs_monetary_log"] = np.log1p(d["obs_monetary_total"])
    d["obs_avg_order_value_log"] = np.log1p(d["obs_avg_order_value"])

    num_cols = ["obs_recency_days", "obs_total_orders", "obs_completed_orders", "obs_monetary_log", "obs_avg_order_value_log",
                "obs_return_rate", "obs_cancel_rate", "obs_ad_order_ratio", "obs_processing_hours_mean", "obs_num_channels",
                "obs_is_omnichannel", "obs_tenure_days"]
    feats = pd.concat([d[num_cols], pd.get_dummies(d[["obs_primary_channel", "obs_region_type"]], prefix=["channel", "region"], dtype=int)], axis=1)
    feats["target"] = y
    feats = feats.reset_index().rename(columns={"index": "customer_id"})
    feat_cols = [c for c in feats.columns if c not in ("customer_id", "target")]
    return _scaled_split(feats.drop(columns="target"), feats["target"], feat_cols)


# ---------------------------------------------------------------------------------------------- chạy toàn bộ
def run(raw_path: str, out_dir: str, anon_key: bytes, mapping_path: str = None) -> Dict[str, Tuple[int, int]]:
    sheets = pd.read_excel(raw_path, sheet_name=None)
    orders = clean_orders(sheets)
    customers = build_customer_features(orders)
    tr1, te1 = build_repeat_purchase_datasets(orders)
    tr2, te2 = build_churn_oot_datasets(orders)

    export = orders.copy()
    for col in ("inserted_at", "updated_at"):
        export[col] = export[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    export = export.drop(columns=list(PII_COLUMNS))
    export = export[["customer_id"] + [c for c in export.columns if c != "customer_id"]]

    outputs = {"orders_cleaned": export, "customer_rfm_features": customers, "train_dataset": tr1,
               "test_dataset": te1, "train_churn_oot": tr2, "test_churn_oot": te2}

    # ẨN DANH HÓA: khóa nội bộ (KH_ + SĐT) -> mã giả danh ở MỌI tệp thành phẩm
    phones = orders[["customer_id", "bill_phone_number"]].drop_duplicates("customer_id")
    to_pseudo = {k: pseudonymize(ph, anon_key) for k, ph in zip(phones["customer_id"], phones["bill_phone_number"])}
    if len(set(to_pseudo.values())) != len(to_pseudo):
        raise RuntimeError("Trùng mã giả danh — tăng độ dài mã")
    for df in outputs.values():
        df["customer_id"] = df["customer_id"].map(to_pseudo)
    # Bảng ánh xạ giả danh -> SĐT chỉ lưu cạnh dữ liệu thô (không nằm trong Cleaned_data/)
    mapping_path = mapping_path or os.path.join(os.path.dirname(os.path.abspath(raw_path)), MAPPING_FILE)
    pd.DataFrame({"customer_id": phones["customer_id"].map(to_pseudo), "bill_phone_number": phones["bill_phone_number"]}) \
        .sort_values("customer_id").to_csv(mapping_path, index=False, encoding="utf-8-sig")

    leaked = find_phone_like(outputs)
    if leaked:
        raise RuntimeError(f"Dữ liệu thành phẩm còn chuỗi giống số điện thoại: {leaked}")
    os.makedirs(out_dir, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
    return {k: v.shape for k, v in outputs.items()}


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description="Làm sạch dữ liệu đa kênh -> Cleaned_data/")
    ap.add_argument("--raw", default=os.path.join(root, "Raw_data", "not_clean_data.xlsx"))
    ap.add_argument("--out", default=os.path.join(root, "Cleaned_data"))
    ap.add_argument("--mapping", default=None, help="Nơi lưu bảng ánh xạ mã giả danh -> SĐT (mặc định: cạnh tệp thô)")
    args = ap.parse_args()
    for name, shape in run(args.raw, args.out, load_anon_key(), args.mapping).items():
        print(f"✓ {name}.csv  {shape[0]:,} dòng × {shape[1]} cột")


if __name__ == "__main__":
    main()
