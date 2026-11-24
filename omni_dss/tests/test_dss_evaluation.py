import os
import re

import numpy as np
import pandas as pd
import pytest

from omni_dss.ml.evaluate_dss import OUTPUT_PATH, bootstrap_rate_ci, cumulative_gains, evaluate

PHONE_ID = re.compile(r"KH_\d{9,}")


def test_cumulative_gains_perfect_and_random_ranking():
    pos = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = cumulative_gains(np.arange(10, 0, -1), pos, points=10)
    assert perfect[1]["captured_share"] == 1.0 and perfect[1]["precision"] == 1.0  # top 20% bắt đủ 2/2
    assert perfect[-1]["captured_share"] == 1.0
    worst = cumulative_gains(np.arange(1, 11), pos, points=10)
    assert worst[7]["captured_share"] == 0.0  # 2 dương tính bị xếp cuối


def test_evaluate_groups_and_no_raw_ids():
    df = pd.DataFrame({
        "customer_id": [f"KH_09000000{i:02d}" for i in range(6)],
        "recency": [5, 6, 30, 40, 2, 3], "frequency": [2, 1, 0, 1, 2, 1], "total_orders": [2, 1, 1, 1, 2, 1],
        "monetary": [600000, 300000, 0, 300000, 700000, 300000],
        "fav_channel": ["Website"] * 6, "region": ["Miền Nam"] * 6,
        "churn_prob": [0.80, 0.75, 0.85, 0.50, 0.15, 0.30],
        "risk_percentile": [95.0, 92.0, 96.0, 70.0, 10.0, 15.0],
        "rfm_segment": ["VIP", "Tiềm năng", "Rủi ro", "Cần chăm sóc", "VIP", "Tiềm năng"],
        "actual_churn": [1, 1, 1, 1, 0, 0],
    })
    r = evaluate(df)
    assert r["data"]["customers"] == 6 and r["data"]["base_actual_repurchase_rate"] == pytest.approx(2 / 6, abs=1e-4)
    opp = r["opportunity_alerts"]["alerted"]
    assert opp["customers"] == 2 and opp["actual_repurchase_rate"] == 1.0 and opp["repurchase_lift"] == 3.0
    assert r["risk_alerts"]["alerted"]["actual_churn_rate"] == 1.0
    assert sum(g["customers"] for g in r["by_alert_level"].values()) == 6
    assert not PHONE_ID.search(str(r))  # case study chỉ dùng ID đã che


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    mask = rng.random(2000) < 0.2
    mask[y == 1] |= rng.random(int(y.sum())) < 0.5  # nhóm được chọn giàu dương tính hơn
    ci = bootstrap_rate_ci(mask, y, n_boot=300)
    lo, hi = ci["rate_ci95"]
    assert lo <= y[mask].mean() <= hi and ci["lift_ci95"][0] > 1.0


def test_model_evaluation_calibration_and_bootstrap():
    from omni_dss.ml.evaluate_models import bootstrap_auc, calibration_table
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.5).astype(int)
    score = y * 0.3 + rng.random(1000) * 0.7
    cal = calibration_table(y, score)
    assert len(cal["bins"]) == 10 and cal["expected_calibration_error"] >= 0
    b = bootstrap_auc(y, score, rng.random(1000), n_boot=200)
    assert b["model_auc_ci95"][0] > 0.6 and b["difference_ci95"][0] > 0 and b["share_model_better"] == 1.0


@pytest.mark.skipif(not os.path.exists(OUTPUT_PATH), reason="chưa chạy train_pipeline.py")
def test_saved_evaluation_has_no_raw_ids():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        text = f.read()
    assert not PHONE_ID.search(text) and "threshold_sensitivity" in text and "by_strategy" in text
