"""Quy trình làm sạch dữ liệu (omni_dss/ml/data_cleaning.py)."""
import os

import re

import numpy as np
import pandas as pd
import pytest

from omni_dss.ml import data_cleaning as dc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW = os.path.join(ROOT, "Raw_data", "not_clean_data.xlsx")
CLEANED = os.path.join(ROOT, "Cleaned_data")


@pytest.mark.parametrize("raw,expected", [
    (912345678, "0912345678"),          # Excel làm mất số 0 đầu
    ("84912345678", "0912345678"),      # mã quốc gia 84
    ("084912345678", "0912345678"),
    ("2713819289", "02713819289"),      # số máy bàn
    ("0912 345 678", "0912345678"),     # khoảng trắng
    (912345678.0, "0912345678"),        # float từ Excel
    ("abc", np.nan),
    (None, np.nan),
])
def test_clean_phone_number(raw, expected):
    out = dc.clean_phone_number(raw)
    assert (pd.isna(out) and pd.isna(expected)) or out == expected


def test_pseudonymize_stable_opaque_and_not_phone_like():
    key = b"test-key-0123456789abcdef"
    a = dc.pseudonymize("0912345678", key)
    assert a == dc.pseudonymize("0912345678", key)                     # ổn định
    assert a != dc.pseudonymize("0912345679", key)                     # khác khách -> khác mã
    assert a != dc.pseudonymize("0912345678", b"another-key-000000000")  # phụ thuộc khóa bí mật
    assert re.fullmatch(r"KH_[A-Z2-7]{10}", a) and "912345678" not in a
    assert not dc.PHONE_LIKE.search(a)


def test_load_anon_key_requires_secret(monkeypatch):
    monkeypatch.setenv(dc.ANON_KEY_ENV, "short")
    with pytest.raises(SystemExit):
        dc.load_anon_key()
    monkeypatch.setenv(dc.ANON_KEY_ENV, "x" * 32)
    assert dc.load_anon_key() == b"x" * 32


def test_find_phone_like_detects_leaks():
    frames = {"t": pd.DataFrame({"note": ["gọi 0912345678 nhé", "ok"], "x": [0.0912345678, 1.5]})}
    assert dc.find_phone_like(frames) == {"t.note": 1}  # số thực không bị tính


def test_clean_recipient_name_fixes_encoding():
    assert dc.clean_recipient_name("  NGUY?N   Th?  Lan ") == "nguyễn thị lan"
    assert dc.clean_recipient_name(None) == ""


@pytest.mark.parametrize("source,channel", [
    ("tiktok", "TikTok Shop"), ("Shopee", "Shopee"), ("AMYRA.shopvn", "Website"),
    ("Váy Amyra", "Facebook Fanpage"), (None, "Unknown/Other"),
])
def test_map_channel_group(source, channel):
    assert dc.map_channel_group(source) == channel


def test_status_group_priority():
    returned = {99}
    assert dc.status_group(3, 1, returned) == "Completed"
    assert dc.status_group(3, 99, returned) == "Returned"  # có trong bảng hoàn trả -> ưu tiên Returned
    assert dc.status_group(4, 1, returned) == "Returned"
    assert dc.status_group(6, 1, returned) == "Cancelled"
    assert dc.status_group(1, 1, returned) == "In_Progress"


def test_clean_customers_filters_and_shared_phone():
    orders = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "bill_phone_number": [912345678, 912345678, None, 987654321, 987654321],
        "bill_full_name": ["Nguy?n A", "Tr?n B", "C", "D", "D"],
        "status": [3, 3, 3, 7, 3],   # đơn 4 đã xóa
        "updated": [np.nan] * 5,
    })
    o = dc.clean_customers_and_names(orders)
    assert list(o["id"]) == [1, 2, 5]                      # bỏ đơn thiếu SĐT và đơn đã xóa
    assert set(o["customer_id"]) == {"KH_0912345678", "KH_0987654321"}
    assert "updated" not in o.columns
    shared = o.groupby("customer_id")["is_shared_phone"].first()
    assert shared["KH_0912345678"] == 1 and shared["KH_0987654321"] == 0  # 2 tên người nhận khác nhau


def test_clean_order_values_imputation_tiers():
    products = pd.DataFrame({"product_id": ["P1"], "retail_price": [150000.0]})
    o = pd.DataFrame({
        "total_price": [0, 0, 0, 0, 500000],
        "cod": [0, 200000, 0, -5, 500000],
        "quantity": [2, 1, 1, np.nan, 1],
        "product_id": ["P1", "PX", "PX", "PX", "P1"],
        "status": [1, 1, 3, 1, 3],
        "ad_id": [np.nan, 1, np.nan, np.nan, np.nan],
        "assigning_care_id": [np.nan] * 5,
    })
    r = dc.clean_order_values(o, products)
    assert list(r["total_price_cleaned"]) == [300000.0, 200000.0, 300000.0, 0.0, 500000.0]
    # bậc 1: 2 × giá SP; bậc 2: COD; bậc 3: đơn hoàn thành -> trung vị; đơn chưa hoàn thành vẫn 0đ = quà tặng
    assert list(r["is_gift_or_free"]) == [0, 0, 0, 1, 0]
    assert r.loc[3, "is_negative_cod"] == 1 and r.loc[3, "cod_cleaned"] == 0
    assert (r["total_price_capped"] <= r["total_price_cleaned"]).all()


def test_province_merge_to_34_provinces():
    provinces = pd.DataFrame({"id": [701, 511], "name": ["Hồ Chí Minh", "Khánh Hòa"], "region_type": ["Miền Nam", "Miền Trung"]})
    districts = pd.DataFrame({"id_districts": [71101], "province_id": [711], "name": ["Thủ Dầu Một"]})
    o = pd.DataFrame({"customer_id": ["KH_1", "KH_2", "KH_2"], "province_id": [np.nan, 705, np.nan], "district_id": [71101, np.nan, np.nan]})
    g = dc.standardize_geography(o, provinces, districts)
    # Bình Dương (711, suy từ huyện) -> Hồ Chí Minh; Ninh Thuận (705) -> Khánh Hòa; thiếu -> lấy theo lịch sử khách
    assert list(g["province_name"]) == ["Hồ Chí Minh", "Khánh Hòa", "Khánh Hòa"]
    assert list(g["region_type"]) == ["Miền Nam", "Miền Trung", "Miền Trung"]


@pytest.mark.skipif(not (os.path.exists(RAW) and os.path.exists(os.path.join(CLEANED, "orders_cleaned.csv"))),
                    reason="cần Raw_data/not_clean_data.xlsx và Cleaned_data/")
def test_full_pipeline_reproduces_cleaned_data(tmp_path):
    try:
        key = dc.load_anon_key()
    except SystemExit:
        pytest.skip(f"thiếu {dc.ANON_KEY_ENV}")
    mapping = tmp_path / "map.csv"
    shapes = dc.run(RAW, str(tmp_path), key, str(mapping))
    m = pd.read_csv(mapping, dtype=str, encoding="utf-8-sig")
    assert m["customer_id"].is_unique and m["bill_phone_number"].is_unique and len(m) == shapes["customer_rfm_features"][0]
    for name in shapes:
        a = pd.read_csv(tmp_path / f"{name}.csv", encoding="utf-8-sig", low_memory=False, dtype={"bill_phone_number": str})
        b = pd.read_csv(os.path.join(CLEANED, f"{name}.csv"), encoding="utf-8-sig", low_memory=False, dtype={"bill_phone_number": str})
        assert list(a.columns) == list(b.columns) and a.shape == b.shape, name
        # ẩn danh hóa: không cột cá nhân, mã giả danh, không chuỗi giống SĐT
        assert not set(dc.PII_COLUMNS) & set(a.columns), name
        assert a["customer_id"].str.fullmatch(r"KH_[A-Z2-7]{10}").all(), name
        assert not dc.find_phone_like({name: a}), name
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, rtol=1e-9, atol=1e-9), (name, c)
            else:
                assert (a[c].astype(str) == b[c].astype(str)).all(), (name, c)


def test_churn_rule_flag_not_a_model_feature():
    """Cờ luật nghiệp vụ (toàn kỳ) không được lọt vào đặc trưng của mô hình dự báo."""
    from omni_dss.ml import feature_engineering as fe
    for name in ("churn_rule_flag", "is_churn_behavioral"):
        assert name not in fe.MODEL_FEATURES and name not in fe.AGGREGATE_COLUMNS
