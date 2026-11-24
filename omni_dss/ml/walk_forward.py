"""
Đánh giá WALK-FORWARD (trượt theo thời gian) cho Bài toán B — dự đoán rời bỏ (nhận xét đợt 1, Vấn đề 3).

Mỗi kỳ k:
  - Mốc cắt c_k cách nhau STEP_DAYS; nhãn churn = không đặt đơn nào trong (c_k, c_k + HORIZON_DAYS].
  - Khách đủ điều kiện (active): có >= 1 đơn trong ACTIVE_DAYS ngày trước c_k.
  - Đặc trưng point-in-time tại c_k (cùng hàm với pipeline chính: feature_engineering.build_customer_aggregates).
  - Huấn luyện trên các mốc TRƯỚC (cửa sổ mở rộng), kiểm tra trên mốc c_k. Nhãn của dữ liệu huấn luyện kết thúc tại
    c_{k-1} + HORIZON_DAYS <= c_k nên không dùng thông tin tương lai.
  - Mô hình: XGBoost với siêu tham số đã chọn ở pipeline chính (không tinh chỉnh lại theo kỳ); đối chứng: chỉ Recency.

Chạy riêng: python -m omni_dss.ml.walk_forward
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from omni_dss.ml.config import SAVED_MODELS_DIR
from omni_dss.ml import feature_engineering as fe
from omni_dss.ml.evaluation import evaluate

HORIZON_DAYS = 7
STEP_DAYS = 7
ACTIVE_DAYS = 30
N_TEST_FOLDS = 4
MIN_TRAIN_CUSTOMERS = 300
OUTPUT_PATH = os.path.join(SAVED_MODELS_DIR, "walk_forward.json")
DEFAULT_PARAMS = {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 1, "n_estimators": 200}


def cutoffs(orders: pd.DataFrame, n_test_folds: int = N_TEST_FOLDS) -> List[pd.Timestamp]:
    """Các mốc cắt cách đều STEP_DAYS, mốc cuối = ngày cuối dữ liệu − HORIZON_DAYS (đủ cửa sổ nhãn)."""
    last = orders["order_time"].max() - pd.Timedelta(days=HORIZON_DAYS)
    return [last - pd.Timedelta(days=STEP_DAYS * j) for j in range(n_test_folds, -1, -1)]


def snapshot(orders: pd.DataFrame, cutoff: pd.Timestamp, fill: Optional[float] = None):
    """Đặc trưng point-in-time + nhãn churn cho khách active tại mốc cắt."""
    agg = fe.build_customer_aggregates(orders, cutoff)
    agg = agg[agg["recency_days"] <= ACTIVE_DAYS]
    y = fe.build_labels(orders, cutoff, cutoff + pd.Timedelta(days=HORIZON_DAYS), agg.index)
    return agg, y.loc[agg.index].astype(int)


def rank_alert_stats(y: np.ndarray, score: np.ndarray, top: float = 10.0, bottom: float = 20.0) -> Dict:
    """
    Độ ổn định của cảnh báo theo thứ hạng trong một kỳ (CV 6.2): top `top`% rủi ro có churn cao hơn cơ sở không,
    `bottom`% rủi ro thấp nhất có mua lại cao hơn cơ sở không. Percentile tính trong chính kỳ đó.
    """
    pct = 100.0 * pd.Series(score).rank(method="average", pct=True).to_numpy()
    hi, lo = pct > 100 - top, pct <= bottom
    base_churn, base_rep = float(y.mean()), float(1 - y.mean())
    rep_lo = float(1 - y[lo].mean()) if lo.any() else float("nan")
    return {"top_percent": top, "top_churn_rate": round(float(y[hi].mean()), 4), "base_churn_rate": round(base_churn, 4),
            "top_churn_lift": round(float(y[hi].mean()) / base_churn, 3),
            "bottom_percent": bottom, "bottom_repurchase_rate": round(rep_lo, 4), "base_repurchase_rate": round(base_rep, 4),
            "bottom_repurchase_lift": round(rep_lo / base_rep, 2) if base_rep > 0 else None}


def run(orders: Optional[pd.DataFrame] = None, params: Optional[Dict] = None, out: str = OUTPUT_PATH) -> Dict:
    orders = orders if orders is not None else fe.load_orders()
    params = params or _best_params()
    cuts = cutoffs(orders)
    snaps = [snapshot(orders, c) for c in cuts]
    folds = []
    for k in range(1, len(cuts)):
        tr_agg = pd.concat([snaps[j][0] for j in range(k)])
        y_tr = pd.concat([snaps[j][1] for j in range(k)])
        te_agg, y_te = snaps[k]
        fold = {"test_cutoff": str(cuts[k])[:19], "train_cutoffs": [str(cuts[j])[:10] for j in range(k)],
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                "churn_rate_train": round(float(y_tr.mean()), 4), "churn_rate_test": round(float(y_te.mean()), 4)}
        if len(y_tr) < MIN_TRAIN_CUSTOMERS or y_tr.nunique() < 2 or y_te.nunique() < 2:
            fold["status"] = "Không đủ dữ liệu cho kỳ này"
            folds.append(fold)
            continue
        fill = float(tr_agg["processing_hours_mean"].median())
        Xtr, Xte = fe.build_model_features(tr_agg, fill), fe.build_model_features(te_agg, fill)
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              scale_pos_weight=float((1 - y_tr).sum() / max(y_tr.sum(), 1)), n_jobs=1, random_state=42, **params)
        model.fit(Xtr, y_tr)
        p = model.predict_proba(Xte)[:, 1]
        m = evaluate(y_te.to_numpy(), p)
        fold.update({"status": "ok", "roc_auc": m["roc_auc"], "balanced_accuracy": m["balanced_accuracy"],
                     "recall_churn": m["recall"], "recall_repurchase": m["retained_recall"],
                     "recency_only_roc_auc": round(float(roc_auc_score(y_te, te_agg["recency_days"])), 4),
                     "dss_alerts": rank_alert_stats(y_te.to_numpy(), p)})
        folds.append(fold)

    ok = [f for f in folds if f["status"] == "ok"]
    aucs = np.array([f["roc_auc"] for f in ok])
    result = {
        "problem": "Bài toán B – Dự đoán rời bỏ, đánh giá walk-forward",
        "design": {"horizon_days": HORIZON_DAYS, "step_days": STEP_DAYS, "active_days": ACTIVE_DAYS,
                   "training": "cửa sổ mở rộng (mọi mốc trước mốc kiểm tra)", "model": "XGBoost", "params": params},
        "folds": folds,
        "summary": {
            "n_folds": len(ok),
            "roc_auc_mean": round(float(aucs.mean()), 4) if len(ok) else None,
            "roc_auc_std": round(float(aucs.std(ddof=1)), 4) if len(ok) > 1 else None,
            "roc_auc_min": round(float(aucs.min()), 4) if len(ok) else None,
            "roc_auc_max": round(float(aucs.max()), 4) if len(ok) else None,
            "beats_recency_only_in": sum(f["roc_auc"] > f["recency_only_roc_auc"] for f in ok),
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def _best_params() -> Dict:
    """Siêu tham số XGBoost đã chọn ở pipeline chính (model_metrics.json); mặc định nếu chưa có."""
    from omni_dss.ml.predict import load_metrics
    for x in (load_metrics() or {}).get("churn", {}).get("models", []):
        if x.get("model") == "XGBoost" and x.get("best_params"):
            return {k: v for k, v in x["best_params"].items() if k in DEFAULT_PARAMS}
    return dict(DEFAULT_PARAMS)


if __name__ == "__main__":
    r = run()
    print("Thiết kế:", r["design"])
    for f in r["folds"]:
        if f["status"] != "ok":
            print(f"  test {f['test_cutoff'][:10]}: {f['status']} (train {f['n_train']}, test {f['n_test']})"); continue
        print(f"  test {f['test_cutoff'][:10]} | train {f['n_train']:>6,} (mốc {', '.join(f['train_cutoffs'])}) | test {f['n_test']:>6,} "
              f"churn {f['churn_rate_test']:.1%} | AUC {f['roc_auc']:.4f} (chỉ Recency {f['recency_only_roc_auc']:.4f}) "
              f"| Bal.Acc {f['balanced_accuracy']:.4f} | recall mua lại {f['recall_repurchase']:.4f}")
    print("Tổng hợp:", r["summary"]); print("💾", OUTPUT_PATH)
