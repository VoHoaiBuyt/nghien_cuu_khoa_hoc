"""
Bài toán A — Phân loại khách hàng MUA LẶP LẠI từ đơn hàng đầu tiên (nhận xét đợt 1, Vấn đề 1 & 4).

Thiết kế:
  - Thời điểm dự đoán: ngay sau khi khách đặt đơn đầu tiên (t0).
  - Lần mua (purchase occasion): mỗi đơn chỉ có 1 sản phẩm, nên mua nhiều món một lúc tạo nhiều đơn cách nhau vài phút.
    Các đơn nối tiếp nhau cách <= SESSION_GAP_HOURS được gộp thành MỘT lần mua. Ngưỡng 8 giờ chọn theo điểm gãy của phân bố
    khoảng cách giữa hai đơn liên tiếp (rất ít cặp đơn cách 8–12 giờ; từ 12–24 giờ tỷ lệ tăng lại = khách quay lại hôm sau).
  - Nhãn: 1 nếu khách có một LẦN MUA KHÁC (không thuộc lần mua đầu) bắt đầu trong (t0, t0 + W ngày]; 0 nếu không.
  - Cắt right-censoring: chỉ giữ khách có t0 <= ngày cuối dữ liệu − W (đủ W ngày theo dõi).
  - Đặc trưng: CHỈ thông tin có tại t0 (giá trị, số lượng, giá bán lẻ, COD, quảng cáo, thời điểm đặt, kênh, dòng sản phẩm,
    vùng, ngành hàng). KHÔNG dùng trạng thái / thời gian xử lý đơn đầu (chốt sau trung vị ~118 giờ).
  - Chia tập THEO THỜI GIAN: 80% khách có t0 sớm nhất -> train, 20% muộn nhất -> test. Mã hóa tần suất chỉ học trên train.
  - So sánh Logistic Regression / Random Forest / XGBoost (GridSearchCV 5-fold trên train, tiêu chí ROC-AUC).

Chạy riêng: python -m omni_dss.ml.repeat_purchase
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from omni_dss.ml.config import SAVED_MODELS_DIR
from omni_dss.ml import feature_engineering as fe
from omni_dss.ml.evaluation import evaluate

RANDOM_STATE = 42
PRIMARY_WINDOW = 14
SESSION_GAP_HOURS = 8.0   # đơn cách đơn trước <= 8 giờ thuộc cùng một lần mua
WINDOWS = (7, 14, 21, 60)
MIN_ELIGIBLE = 500        # dưới ngưỡng này: không đủ dữ liệu để huấn luyện/kiểm tra
MIN_POSITIVES = 30
TEST_SHARE = 0.20
OUTPUT_PATH = os.path.join(SAVED_MODELS_DIR, "repeat_metrics.json")

CATEGORICAL = {"channel_group": "channel", "channel_product_line": "prod_line", "region_type": "region"}
FIRST_ORDER_FEATURES: List[str] = [
    "price_log", "retail_price_log", "quantity", "price_ratio", "cod_ratio", "is_from_ad",
    "hour", "dayofweek", "is_weekend", "day", "num_variations", "category_freq", "province_freq",
]
# Thông tin CHƯA có tại thời điểm đặt đơn đầu -> cấm dùng (kiểm thử: test_repeat_features_exclude_post_order_info)
FORBIDDEN = ("status", "status_group", "status_name", "is_returned", "processing_hours", "updated_at", "return_reason")


def first_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Đơn hàng đầu tiên của mỗi khách (theo thời điểm đặt, hòa -> mã đơn nhỏ hơn)."""
    o = orders.sort_values(["customer_id", "order_time", "id"])
    return o.groupby("customer_id", sort=False).head(1).set_index("customer_id")


def purchase_occasions(orders: pd.DataFrame, gap_hours: float = SESSION_GAP_HOURS) -> pd.Series:
    """Số thứ tự lần mua (0 = lần đầu) của từng đơn: đơn mới bắt đầu lần mua mới nếu cách đơn trước > gap_hours."""
    o = orders.sort_values(["customer_id", "order_time", "id"])
    gap = o.groupby("customer_id")["order_time"].diff().dt.total_seconds() / 3600.0
    return (gap > gap_hours).astype(int).groupby(o["customer_id"]).cumsum().reindex(orders.index)


def build_labels(orders: pd.DataFrame, firsts: pd.DataFrame, window_days: int, gap_hours: float = SESSION_GAP_HOURS):
    """(nhãn cho khách đủ điều kiện, mốc cắt t0 tối đa). Khách có t0 quá gần cuối dữ liệu bị loại (right-censoring)."""
    end = orders["order_time"].max()
    last_t0 = end - pd.Timedelta(days=window_days)
    eligible = firsts[firsts["order_time"] <= last_t0]
    o = orders[orders["customer_id"].isin(eligible.index)][["customer_id", "order_time", "id"]].copy()
    o["occasion"] = purchase_occasions(o, gap_hours)
    o = o.join(eligible["order_time"].rename("t0"), on="customer_id")
    # đơn mở đầu một lần mua mới (occasion >= 1) trong (t0, t0 + W]
    later = o[(o["occasion"] >= 1) & (o["order_time"] > o["t0"]) & (o["order_time"] <= o["t0"] + pd.Timedelta(days=window_days))]
    y = pd.Series(eligible.index.isin(later["customer_id"].unique()).astype(int), index=eligible.index, name="target")
    return y, last_t0


def next_order_gap_hours(orders: pd.DataFrame, firsts: pd.DataFrame, customers: pd.Index) -> pd.Series:
    """Số giờ từ đơn đầu đến LẦN MUA kế tiếp (chỉ khách có lần mua kế tiếp)."""
    o = orders[orders["customer_id"].isin(customers)].copy()
    o["occasion"] = purchase_occasions(o)
    nxt = o[o["occasion"] >= 1].groupby("customer_id")["order_time"].min()
    return (nxt - firsts.loc[nxt.index, "order_time"]).dt.total_seconds() / 3600.0


def build_features(firsts: pd.DataFrame, train_ids: pd.Index) -> pd.DataFrame:
    """Đặc trưng chỉ từ đơn đầu tiên. Mã hóa tần suất (tỉnh, ngành hàng) học trên train để tránh rò rỉ."""
    f = firsts
    X = pd.DataFrame(index=f.index)
    price = f["total_price_cleaned"].astype(float).clip(lower=0)
    retail = f["retail_price"].astype(float).clip(lower=0)
    qty = f["quantity"].astype(float).clip(lower=1)
    X["price_log"] = np.log1p(price)
    X["retail_price_log"] = np.log1p(retail)
    X["quantity"] = qty
    X["price_ratio"] = price / (qty * retail + 1.0)
    X["cod_ratio"] = f["cod_cleaned"].astype(float).clip(lower=0) / (price + 1.0)
    X["is_from_ad"] = f["is_from_ad"].astype(float)
    X["hour"] = f["order_time"].dt.hour
    X["dayofweek"] = f["order_time"].dt.day_of_week
    X["is_weekend"] = X["dayofweek"].isin([5, 6]).astype(float)
    X["day"] = f["order_time"].dt.day
    X["num_variations"] = f["num_variations"].astype(float)
    tr = f.loc[f.index.intersection(train_ids)]
    X["category_freq"] = f["categories_name"].map(tr["categories_name"].value_counts(normalize=True)).fillna(0.0)
    X["province_freq"] = f["province_id"].map(tr["province_id"].value_counts(normalize=True)).fillna(0.0)
    X = pd.concat([X, pd.get_dummies(f[list(CATEGORICAL)].rename(columns=CATEGORICAL), dtype=float)], axis=1)
    assert not any(c.startswith(FORBIDDEN) for c in X.columns)
    return X


def temporal_split(firsts: pd.DataFrame, ids: pd.Index, test_share: float = TEST_SHARE):
    """Khách có đơn đầu sớm -> train; muộn -> test (kiểm tra trên giai đoạn sau, không chia ngẫu nhiên)."""
    t0 = firsts.loc[ids, "order_time"].sort_values(kind="stable")
    n_test = int(round(len(t0) * test_share))
    return t0.index[:-n_test], t0.index[-n_test:], t0.iloc[-n_test]


def _search_space(y_train: pd.Series):
    pos, neg = int(y_train.sum()), int((1 - y_train).sum())
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))]),
            {"model__C": [0.01, 0.1, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE),
            {"max_depth": [4, 8], "min_samples_leaf": [20, 50]},
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          scale_pos_weight=neg / max(pos, 1), n_jobs=1, random_state=RANDOM_STATE),
            {"n_estimators": [200], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]},
        ),
    }


def run_window(orders: pd.DataFrame, firsts: pd.DataFrame, window_days: int) -> Dict:
    y, last_t0 = build_labels(orders, firsts, window_days)
    res = {"window_days": window_days, "eligible_customers": int(len(y)), "positives": int(y.sum()),
           "positive_rate": round(float(y.mean()), 4) if len(y) else None,
           "first_order_cutoff": str(last_t0)[:19], "data_end": str(orders["order_time"].max())[:19]}
    if len(y) < MIN_ELIGIBLE or y.sum() < MIN_POSITIVES:
        res["status"] = (f"Không đủ dữ liệu: cần >= {MIN_ELIGIBLE} khách đủ điều kiện và >= {MIN_POSITIVES} khách mua lại "
                         f"(hiện {len(y)} khách, {int(y.sum())} mua lại).")
        return res

    # So sánh với nhãn cũ (không gộp lần mua) để thấy số "mua lại" thực chất là cùng một lần mua nhiều món
    y_raw, _ = build_labels(orders, firsts, window_days, gap_hours=0.0)
    gap = next_order_gap_hours(orders, firsts, y[y == 1].index)
    res["label_quality"] = {
        "session_gap_hours": SESSION_GAP_HOURS,
        "positives_without_session_merge": int(y_raw.sum()),
        "positives_removed_by_session_merge": int(y_raw.sum() - y.sum()),
        "positives_next_purchase_within_24h": round(float((gap <= 24).mean()), 4),
    }

    tr_ids, te_ids, test_start = temporal_split(firsts, y.index)
    X = build_features(firsts.loc[y.index], tr_ids)
    Xtr, Xte, ytr, yte = X.loc[tr_ids], X.loc[te_ids], y.loc[tr_ids], y.loc[te_ids]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = []
    for name, (est, grid) in _search_space(ytr).items():
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
        p = gs.best_estimator_.predict_proba(Xte)[:, 1]
        # evaluate() tính chỉ số cho lớp dương = 1 (ở đây: MUA LẠI)
        m = evaluate(yte.to_numpy(), p)
        models.append({"model": name, "cv_roc_auc": round(float(gs.best_score_), 4), "best_params": gs.best_params_,
                       "test": {k: m[k] for k in ("roc_auc", "balanced_accuracy", "precision", "recall", "f1_score", "accuracy")},
                       "test_pr_auc": round(float(_pr_auc(yte.to_numpy(), p)), 4)})
    best = max(models, key=lambda d: d["cv_roc_auc"])
    res.update({
        "status": "ok", "n_train": int(len(tr_ids)), "n_test": int(len(te_ids)),
        "positive_rate_train": round(float(ytr.mean()), 4), "positive_rate_test": round(float(yte.mean()), 4),
        "test_period_start": str(test_start)[:19], "features": list(X.columns),
        "models": models, "best_model": best["model"], "selection": "CV ROC-AUC (5-fold, chỉ tập train)",
    })
    return res


def _pr_auc(y, p) -> float:
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, p)


def run(orders: Optional[pd.DataFrame] = None, windows=WINDOWS, out: str = OUTPUT_PATH) -> Dict:
    orders = orders if orders is not None else fe.load_orders()
    firsts = first_orders(orders)
    result = {
        "problem": "Bài toán A – Phân loại khách hàng mua lặp lại trong W ngày kể từ đơn hàng đầu tiên",
        "label": (f"1 nếu khách có lần mua mới (đơn cách đơn trước > {SESSION_GAP_HOURS:g} giờ) trong (t0, t0 + W]; "
                  "chỉ giữ khách có đủ W ngày theo dõi (cắt right-censoring)"),
        "features_policy": "Chỉ thông tin có tại thời điểm đặt đơn đầu; loại trạng thái và thời gian xử lý đơn",
        "primary_window_days": PRIMARY_WINDOW,
        "windows": [run_window(orders, firsts, w) for w in windows],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result


if __name__ == "__main__":
    r = run()
    for w in r["windows"]:
        head = f"W={w['window_days']:>2}: {w['eligible_customers']:,} khách, mua lại {w['positives']:,} ({(w['positive_rate'] or 0):.2%})"
        if w["status"] != "ok":
            print(head, "->", w["status"]); continue
        print(head, f"| train {w['n_train']:,} / test {w['n_test']:,} (test từ {w['test_period_start'][:10]})")
        for m in w["models"]:
            t = m["test"]
            print(f"     {m['model']:<20} CV AUC {m['cv_roc_auc']:.4f} | test AUC {t['roc_auc']:.4f} PR-AUC {m['test_pr_auc']:.4f} "
                  f"Bal.Acc {t['balanced_accuracy']:.4f} Recall {t['recall']:.4f} Precision {t['precision']:.4f}")
    print("💾", OUTPUT_PATH)
