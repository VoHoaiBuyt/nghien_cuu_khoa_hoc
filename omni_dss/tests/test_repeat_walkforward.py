"""Bài toán A (mua lặp lại) và đánh giá walk-forward Bài toán B."""
import numpy as np
import pandas as pd
import pytest

from omni_dss.ml import repeat_purchase as rp
from omni_dss.ml import walk_forward as wf

T0 = pd.Timestamp("2022-10-01 10:00")


def _orders(rows):
    """rows: (customer_id, order_id, ngày lệch so với T0, trạng thái)."""
    recs = []
    for cid, oid, d, st in rows:
        t = T0 + pd.Timedelta(days=d)
        recs.append({
            "customer_id": cid, "id": oid, "order_id": oid, "order_time": t, "update_time": t + pd.Timedelta(days=3),
            "status_group": st, "total_price_cleaned": 300000.0, "retail_price": 150000.0, "quantity": 2.0,
            "cod_cleaned": 300000.0, "is_from_ad": 1, "num_variations": 3, "categories_name": "VÁY", "province_id": 701,
            "channel_group": "Facebook Fanpage", "channel_product_line": "Tổng hợp", "region_type": "Miền Nam", "source": "S1",
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- Bài toán A
def test_repeat_labels_window_and_right_censoring():
    o = _orders([
        ("A", 1, 0, "Completed"), ("A", 2, 5, "Completed"),    # mua lại trong 14 ngày -> 1
        ("B", 3, 1, "Completed"), ("B", 4, 30, "Completed"),   # mua lại sau 14 ngày -> 0
        ("C", 5, 2, "Cancelled"),                              # không mua lại -> 0
        ("D", 6, 35, "Completed"),                             # đơn đầu quá gần cuối dữ liệu -> bị loại
        ("E", 7, 40, "Completed"),                             # đơn cuối dữ liệu
    ])
    y, last_t0 = rp.build_labels(o, rp.first_orders(o), 14)
    assert last_t0 == T0 + pd.Timedelta(days=26)
    assert set(y.index) == {"A", "B", "C"}  # D, E chưa đủ 14 ngày theo dõi
    assert y.to_dict() == {"A": 1, "B": 0, "C": 0}


def test_repeat_features_use_first_order_only_and_exclude_post_order_info():
    o = _orders([("A", 1, 0, "Returned"), ("A", 2, 3, "Completed"), ("B", 3, 1, "Completed")])
    o.loc[o["id"] == 2, "total_price_cleaned"] = 9_999_999.0  # đơn thứ hai không được ảnh hưởng đặc trưng
    firsts = rp.first_orders(o)
    X = rp.build_features(firsts, firsts.index)
    assert X.loc["A", "price_log"] == pytest.approx(np.log1p(300000.0))
    assert not any(c.startswith(rp.FORBIDDEN) for c in X.columns)
    assert not {"status_group", "processing_hours", "update_time"} & set(X.columns)


def test_repeat_temporal_split_tests_on_later_customers():
    o = _orders([(f"K{i}", i, i, "Completed") for i in range(10)])
    firsts = rp.first_orders(o)
    tr, te, start = rp.temporal_split(firsts, firsts.index, 0.2)
    assert list(te) == ["K8", "K9"] and firsts.loc[tr, "order_time"].max() < start


def test_repeat_window_without_enough_data_is_reported_not_trained():
    o = _orders([("A", 1, 0, "Completed"), ("B", 2, 1, "Completed"), ("C", 3, 90, "Completed")])
    res = rp.run_window(o, rp.first_orders(o), 60)
    assert res["status"].startswith("Không đủ dữ liệu") and "models" not in res


# ---------------------------------------------------------------- Walk-forward Bài toán B
def test_walk_forward_cutoffs_leave_full_label_window():
    o = _orders([("A", 1, 0, "Completed"), ("B", 2, 60, "Completed")])
    cuts = wf.cutoffs(o, n_test_folds=3)
    assert cuts[-1] == o["order_time"].max() - pd.Timedelta(days=wf.HORIZON_DAYS)
    assert all((b - a) == pd.Timedelta(days=wf.STEP_DAYS) for a, b in zip(cuts, cuts[1:]))
    # nhãn của mốc huấn luyện gần nhất kết thúc không muộn hơn mốc kiểm tra -> không dùng thông tin tương lai
    assert all(a + pd.Timedelta(days=wf.HORIZON_DAYS) <= b for a, b in zip(cuts, cuts[1:]))


def test_walk_forward_snapshot_active_customers_and_labels():
    o = _orders([
        ("A", 1, 0, "Completed"), ("A", 2, 12, "Completed"),   # mua trong cửa sổ nhãn -> không churn
        ("B", 3, 5, "Completed"),                              # không mua lại -> churn
        ("C", 4, -60, "Completed"),                            # không active trong 30 ngày -> loại
    ])
    agg, y = wf.snapshot(o, T0 + pd.Timedelta(days=10))
    assert set(agg.index) == {"A", "B"}
    assert y.to_dict() == {"A": 0, "B": 1}


def test_purchase_occasions_merge_orders_within_session_gap():
    h = 1 / 24
    o = _orders([("A", 1, 0, "Completed"), ("A", 2, 5 * h, "Completed"), ("A", 3, 12 * h, "Completed"),  # chuỗi cách 5h, 7h -> 1 lần
                 ("A", 4, 3, "Completed"),                                                                 # 3 ngày sau -> lần 2
                 ("B", 5, 0, "Completed"), ("B", 6, 9 * h, "Completed")])                                 # cách 9h -> 2 lần
    occ = rp.purchase_occasions(o).set_axis(o["id"]).to_dict()
    assert occ == {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1}


def test_repeat_label_counts_new_purchase_occasions_only():
    h = 1 / 24
    o = _orders([("A", 1, 0, "Completed"), ("A", 2, 0.5 * h, "Completed"),   # thêm món sau 30 phút: KHÔNG phải mua lại
                 ("B", 3, 1, "Completed"), ("B", 4, 1.5, "Completed"),       # quay lại sau 12 giờ: mua lại
                 ("C", 5, 2, "Completed"), ("C", 6, 2, "Completed"),         # 2 đơn cùng thời điểm: không phải mua lại
                 ("D", 7, 40, "Completed")])
    firsts = rp.first_orders(o)
    y, _ = rp.build_labels(o, firsts, 14)
    assert y.to_dict() == {"A": 0, "B": 1, "C": 0}
    y_raw, _ = rp.build_labels(o, firsts, 14, gap_hours=0.0)   # không gộp: A bị tính là mua lại
    assert y_raw.to_dict() == {"A": 1, "B": 1, "C": 0}
    assert rp.next_order_gap_hours(o, firsts, y[y == 1].index).loc["B"] == pytest.approx(12.0)
