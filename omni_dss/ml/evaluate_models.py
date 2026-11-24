"""
CV 6.1 — Đánh giá hiệu quả mô hình dự đoán trong môi trường thực nghiệm.

Tổng hợp và bổ sung các phép đo mà một lần chia train/test đơn lẻ chưa trả lời được:
  1. So sánh mô hình (CV trên train + test out-of-time)              <- model_metrics.json
  2. Độ bất định: khoảng tin cậy bootstrap của ROC-AUC trên tập test, và chênh lệch so với đối chứng "chỉ Recency"
  3. Hiệu chỉnh xác suất (calibration): xác suất dự đoán so với tỷ lệ thực tế theo nhóm -> căn cứ cảnh báo theo thứ hạng
  4. Độ ổn định theo thời gian (walk-forward)                         <- walk_forward.json
  5. Bài toán A – mua lặp lại sau đơn đầu                              <- repeat_metrics.json

Đầu ra : saved_models/model_evaluation.json (chỉ số liệu gộp).
Chạy riêng: python -m omni_dss.ml.evaluate_models
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from omni_dss.ml.config import SAVED_MODELS_DIR

SCORED_PATH = os.path.join(SAVED_MODELS_DIR, "scored_customers.csv")
OUTPUT_PATH = os.path.join(SAVED_MODELS_DIR, "model_evaluation.json")
N_BOOT = 2000


def _r(x, d: int = 4) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(SAVED_MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_auc(y: np.ndarray, score: np.ndarray, baseline: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> Dict[str, Any]:
    """CI 95% của ROC-AUC mô hình, của đối chứng, và của chênh lệch (bootstrap ghép cặp trên cùng mẫu khách)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    a_m, a_b, diff = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        m, b = roc_auc_score(y[idx], score[idx]), roc_auc_score(y[idx], baseline[idx])
        a_m.append(m); a_b.append(b); diff.append(m - b)

    def ci(a):
        return [_r(np.percentile(a, 2.5)), _r(np.percentile(a, 97.5))]

    return {"model_auc": _r(roc_auc_score(y, score)), "model_auc_ci95": ci(a_m),
            "baseline_auc": _r(roc_auc_score(y, baseline)), "baseline_auc_ci95": ci(a_b),
            "difference_ci95": ci(diff), "share_model_better": _r(np.mean(np.array(diff) > 0), 3), "n_boot": len(diff)}


def calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    """Chia theo thập phân vị điểm dự đoán: xác suất TB vs tỷ lệ churn thực tế; ECE = sai lệch TB có trọng số."""
    q = pd.qcut(pd.Series(proba).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"bin": q, "p": proba, "y": y})
    g = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
    ece = float((g["n"] * (g["mean_predicted"] - g["actual_rate"]).abs()).sum() / g["n"].sum())
    return {
        "bins": [{"decile": int(i) + 1, "customers": int(r.n), "mean_predicted": _r(r.mean_predicted), "actual_rate": _r(r.actual_rate)}
                 for i, r in g.iterrows()],
        "expected_calibration_error": _r(ece),
        "brier_model": _r(brier_score_loss(y, proba)),
        "brier_constant_base_rate": _r(brier_score_loss(y, np.full(len(y), y.mean()))),
        "mean_predicted": _r(proba.mean()), "actual_rate": _r(y.mean()),
        "monotonic_actual_by_decile": bool(np.all(np.diff(g["actual_rate"].to_numpy()) >= -0.02)),
    }


def evaluate(scored: pd.DataFrame) -> Dict[str, Any]:
    y = scored["actual_churn"].to_numpy().astype(int)
    p = scored["churn_prob"].to_numpy()
    metrics = _load("model_metrics.json") or {}
    churn = metrics.get("churn", {})

    comparison = [{
        "model": m["model"], "cv_roc_auc_mean": m["cv_roc_auc_mean"], "cv_roc_auc_std": m["cv_roc_auc_std"],
        **{f"test_{k}": m["test"][k] for k in ("roc_auc", "balanced_accuracy", "recall", "retained_recall", "retained_pr_auc")},
    } for m in churn.get("models", [])]

    wf = _load("walk_forward.json")
    wf_ok = [f for f in (wf or {}).get("folds", []) if f.get("status") == "ok"]
    rp = _load("repeat_metrics.json")
    repeat = None
    if rp:
        repeat = [{k: w.get(k) for k in ("window_days", "status", "eligible_customers", "positive_rate", "best_model")}
                  | {f"label_{k}": v for k, v in (w.get("label_quality") or {}).items()}
                  | ({"test_roc_auc": next(m["test"]["roc_auc"] for m in w["models"] if m["model"] == w["best_model"]),
                      "test_pr_auc": next(m["test_pr_auc"] for m in w["models"] if m["model"] == w["best_model"])}
                     if w.get("status") == "ok" else {})
                  for w in rp.get("windows", [])]

    return {
        "test_set": {"customers": int(len(y)), "churn_rate": _r(y.mean()), "model": churn.get("best_model"),
                     "outcome_window": churn.get("outcome_window")},
        "model_comparison": comparison,
        "best_by_cv": churn.get("best_overall_by_cv"),
        "uncertainty_vs_recency_baseline": bootstrap_auc(y, p, scored["recency"].to_numpy()),
        "repurchase_ranking": {  # lớp thiểu số (mua lại) là lớp doanh nghiệp quan tâm
            "pr_auc": _r(average_precision_score(1 - y, 1 - p)), "base_rate": _r(1 - y.mean()),
        },
        "calibration": calibration_table(y, p),
        "walk_forward": None if not wf else {
            "design": wf["design"], "summary": wf["summary"],
            "folds": [{k: f.get(k) for k in ("test_cutoff", "n_train", "n_test", "churn_rate_test", "roc_auc", "recency_only_roc_auc")}
                      for f in wf_ok],
        },
        "repeat_purchase": repeat,
        "conclusions_inputs": {  # số liệu gốc cho phần nhận xét (người viết báo cáo tự diễn giải)
            "oot_test_auc": churn.get("best_model_test", {}).get("roc_auc"),
            "walk_forward_auc_mean": (wf or {}).get("summary", {}).get("roc_auc_mean"),
            "walk_forward_auc_std": (wf or {}).get("summary", {}).get("roc_auc_std"),
        },
    }


def run(path: str = SCORED_PATH, out: str = OUTPUT_PATH) -> Dict[str, Any]:
    res = evaluate(pd.read_csv(path, dtype={"customer_id": str}))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    r = run()
    u, c = r["uncertainty_vs_recency_baseline"], r["calibration"]
    print(f"ROC-AUC {u['model_auc']} CI95 {u['model_auc_ci95']} | chỉ Recency {u['baseline_auc']} | chênh lệch CI95 {u['difference_ci95']}")
    print(f"Calibration: dự đoán TB {c['mean_predicted']} vs thực tế {c['actual_rate']} | ECE {c['expected_calibration_error']}")
    print(f"💾 {OUTPUT_PATH}")
