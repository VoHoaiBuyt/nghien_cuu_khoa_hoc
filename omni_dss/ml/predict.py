"""
Module dự đoán hành vi khách hàng (CV 4.2) — dùng mô hình đã huấn luyện bởi omni_dss/train_pipeline.py.

Đầu vào : số liệu hành vi point-in-time của khách (feature_engineering.build_customer_aggregates) hoặc file đơn hàng.
Đầu ra  : xác suất churn / mua lại, percentile rủi ro, bậc rủi ro, phân khúc RFM và lý do chính (SHAP) từng khách.

    from omni_dss.ml.predict import Predictor
    p = Predictor.load()
    scored = p.score_orders(fe.load_orders(), as_of=pd.Timestamp("2022-11-02"))
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from omni_dss.ml import feature_engineering as fe
from omni_dss.ml.config import SAVED_MODELS_DIR
from omni_dss.ml.explain import explain_rows, make_explainer
from omni_dss.ml.risk_rank import REFERENCE_FILE, load_reference, percentile_of, tier_of

ARTIFACTS = {
    "kmeans": "kmeans_rfm.pkl",
    "scaler_rfm": "scaler_rfm.pkl",
    "segment_map": "segment_map.pkl",
    "churn_model": "churn_model.pkl",
    "churn_features": "churn_features.pkl",
}


class ModelNotReadyError(RuntimeError):
    pass


def load_metrics(saved_dir: str = SAVED_MODELS_DIR) -> Optional[Dict[str, Any]]:
    path = os.path.join(saved_dir, "model_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def models_ready(saved_dir: str = SAVED_MODELS_DIR) -> bool:
    return all(os.path.exists(os.path.join(saved_dir, f)) for f in (*ARTIFACTS.values(), REFERENCE_FILE))


@dataclass
class Predictor:
    kmeans: Any
    scaler_rfm: Any
    segment_map: Dict[int, str]
    churn_model: Any
    churn_features: list
    risk_reference: np.ndarray
    model_name: str = "unknown"
    processing_hours_fill: float = 0.0
    explainer: Any = field(default=None, repr=False)

    @classmethod
    def load(cls, saved_dir: str = SAVED_MODELS_DIR, with_shap: bool = True) -> "Predictor":
        if not models_ready(saved_dir):
            raise ModelNotReadyError("Chưa có mô hình. Hãy chạy: python omni_dss/train_pipeline.py")
        art = {k: joblib.load(os.path.join(saved_dir, f)) for k, f in ARTIFACTS.items()}
        churn = (load_metrics(saved_dir) or {}).get("churn", {})
        p = cls(**art, risk_reference=load_reference(saved_dir),
                model_name=churn.get("best_model", type(art["churn_model"]).__name__),
                processing_hours_fill=float(churn.get("inference_defaults", {}).get("processing_hours_mean", 0.0)))
        if with_shap:
            try:
                p.explainer = make_explainer(p.churn_model)
            except Exception:  # SHAP không khả dụng -> vẫn dự đoán, không có lý do
                p.explainer = None
        return p

    def segment(self, agg: pd.DataFrame) -> pd.Series:
        X = fe.rfm_feature_frame(agg["recency_days"], agg["completed_orders"], agg["monetary"])  # F = đơn thành công
        clusters = self.kmeans.predict(self.scaler_rfm.transform(X))
        return pd.Series([self.segment_map.get(int(c), f"Cluster {int(c)}") for c in clusters], index=agg.index)

    def score_aggregates(self, agg: pd.DataFrame, top_k: int = 3, explain_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Chấm điểm nhiều khách. Trả DataFrame (index = customer_id): churn_prob, repurchase_prob, risk_percentile,
        risk_level, rfm_segment, top_factors. SHAP chỉ tính cho `explain_limit` khách rủi ro cao nhất (None = tất cả).
        """
        X = fe.build_model_features(agg, processing_hours_fill=self.processing_hours_fill)[self.churn_features]
        proba = self.churn_model.predict_proba(X)[:, 1] if len(X) else np.array([])
        out = pd.DataFrame(index=agg.index)
        out["churn_prob"] = proba
        out["repurchase_prob"] = 1.0 - proba
        out["risk_percentile"] = percentile_of(self.risk_reference, proba) if len(X) else []
        out["risk_level"] = [tier_of(p) for p in out["risk_percentile"]]
        out["rfm_segment"] = self.segment(agg) if len(X) else []
        out["top_factors"] = [[] for _ in range(len(out))]
        if self.explainer is not None and len(out) and explain_limit != 0:
            order = np.argsort(-proba)
            idx = order if explain_limit is None else order[:max(explain_limit, 0)]
            for pos, f in zip(idx, explain_rows(self.explainer, X.iloc[idx], top_k=top_k)):
                out.iat[pos, out.columns.get_loc("top_factors")] = f
        return out

    def score_orders(self, orders: pd.DataFrame, as_of: pd.Timestamp, **kw) -> pd.DataFrame:
        """Chấm điểm mọi khách có đơn trước mốc `as_of` (chỉ dùng thông tin đã biết tại as_of)."""
        agg = fe.build_customer_aggregates(orders, as_of)
        return agg.join(self.score_aggregates(agg, **kw))
