"""Module dự đoán (CV 4.2) cho cùng kết quả với pipeline huấn luyện: xác suất, percentile, phân khúc."""
import os

import pandas as pd
import pytest

from omni_dss.tests.conftest import requires_models

SCORED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "scored_customers.csv")
ORDERS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Cleaned_data", "orders_cleaned.csv")


@requires_models
@pytest.mark.skipif(not (os.path.exists(SCORED) and os.path.exists(ORDERS)), reason="thiếu dữ liệu thật")
def test_predictor_matches_training_scores():
    from omni_dss.ml import feature_engineering as fe
    from omni_dss.ml.predict import Predictor, load_metrics
    cutoff = pd.Timestamp(load_metrics()["churn"]["observation_window"]["cutoff"])
    ref = pd.read_csv(SCORED).set_index("customer_id")
    agg = fe.build_customer_aggregates(fe.load_orders(), cutoff)
    sc = Predictor.load(with_shap=False).score_aggregates(agg.loc[ref.index], explain_limit=0)
    assert (sc["churn_prob"] - ref["churn_prob"]).abs().max() < 1e-4
    assert (sc["risk_percentile"] - ref["risk_percentile"]).abs().max() < 0.05
    assert (sc["rfm_segment"] == ref["rfm_segment"]).all()
    assert set(sc["risk_level"]) <= {"High Risk", "Medium Risk", "Low Risk"}
