"""Nội dung 4 — bộ luật DSS (CV 4.1), kịch bản (CV 4.3), cảnh báo sớm (CV 4.4), thang percentile rủi ro."""
import numpy as np
import pandas as pd
import pytest

from omni_dss.ml import risk_rank as rr
from omni_dss.ml.dss import (
    AT_RISK, HIGH_MIN, LEVEL_PRIORITY, MEDIUM_MIN, NEEDS_ATTENTION, POTENTIAL, RULES, STRATEGIES, UNKNOWN, VIP,
    VIP_LOYAL_MAX, apply_dss, describe_rules, describe_strategies, early_warning_list, evaluate_customer,
    mask_customer_id, normalize_segment,
)
from omni_dss.ml.feature_engineering import SEGMENT_LABELS

SEGMENTS = [VIP, POTENTIAL, NEEDS_ATTENTION, AT_RISK, UNKNOWN]
assert (VIP_LOYAL_MAX, MEDIUM_MIN, HIGH_MIN) == (20.0, 60.0, 90.0)
# Phủ toàn bộ biên ngưỡng: 0, <20, 20, <60, 60, giữa, 90, >90, 100
GRID = [0.0, 0.5, 10.0, 19.99, 20.0, 40.0, 59.99, 60.0, 75.0, 90.0, 90.01, 95.0, 100.0]


def expected_level(segment: str, pct: float) -> str:
    """Bảng kỳ vọng viết độc lập với engine."""
    high, medium = pct > 90, 60 <= pct <= 90
    if segment == VIP:
        return "CRITICAL" if high else ("MEDIUM" if pct >= 20 else "LOW")
    if segment == AT_RISK:
        return "HIGH" if (high or medium) else "MEDIUM"
    if segment == NEEDS_ATTENTION:
        return "HIGH" if high else "MEDIUM"
    return "HIGH" if high else ("MEDIUM" if medium else "LOW")  # Tiềm năng & Không xác định


@pytest.mark.parametrize("segment", SEGMENTS)
@pytest.mark.parametrize("pct", GRID)
def test_alert_level_full_grid(segment, pct):
    res = evaluate_customer(segment, pct)
    assert res["alert_level"] == expected_level(segment, pct)
    assert res["recommendations"], "luôn phải có ít nhất 1 khuyến nghị"


@pytest.mark.parametrize("segment", SEGMENTS)
@pytest.mark.parametrize("pct", GRID)
def test_level_is_max_of_matched_rules_and_strategy_follows_top_rule(segment, pct):
    res = evaluate_customer(segment, pct)
    rules = {r.rule_id: r for r in RULES}
    if res["matched_rules"]:
        levels = [LEVEL_PRIORITY[rules[i].level] for i in res["matched_rules"]]
        assert LEVEL_PRIORITY[res["alert_level"]] == max(levels) and levels == sorted(levels, reverse=True)
        assert res["strategy"] == rules[res["matched_rules"][0]].strategy
    else:
        assert res["alert_level"] == "LOW" and res["strategy"] is None


def test_at_risk_is_never_low():
    assert all(LEVEL_PRIORITY[evaluate_customer(AT_RISK, p)["alert_level"]] >= 2 for p in GRID)


def test_percentile_below_one_is_not_read_as_probability():
    assert evaluate_customer(VIP, 0.8)["alert_level"] == "LOW"


def test_every_rule_has_a_valid_strategy_and_all_five_strategies_are_used():
    assert len(STRATEGIES) == 5
    assert {r.strategy for r in RULES} == set(STRATEGIES)
    assert len(describe_rules()) == len(RULES) and len(describe_strategies()) == 5


def test_opportunity_definition():
    assert evaluate_customer(VIP, 15)["is_opportunity"] and evaluate_customer(POTENTIAL, 20)["is_opportunity"]
    assert not evaluate_customer(POTENTIAL, 20.01)["is_opportunity"]
    assert not evaluate_customer(NEEDS_ATTENTION, 5)["is_opportunity"]


@pytest.mark.parametrize("raw,expected", [
    ("VIP", VIP), (" Vip ", VIP), ("TIỀM NĂNG", POTENTIAL), ("tiem nang", POTENTIAL), ("can_cham_soc", NEEDS_ATTENTION),
    ("RỦI RO", AT_RISK), ("at-risk", AT_RISK), ("xyz", UNKNOWN), (None, UNKNOWN),
])
def test_normalize_segment(raw, expected):
    assert normalize_segment(raw) == expected


def test_kmeans_labels_are_all_known_to_dss():
    assert all(normalize_segment(label) == label for label in SEGMENT_LABELS)


def test_apply_dss_and_early_warning_order():
    df = pd.DataFrame({"customer_id": list("ABCD"), "rfm_segment": [AT_RISK, VIP, POTENTIAL, VIP],
                       "risk_percentile": [70.0, 95.0, 99.0, 5.0]})
    out = apply_dss(df)
    assert list(out["alert_level"]) == ["HIGH", "CRITICAL", "HIGH", "LOW"]
    assert list(out["strategy"]) == ["KB2", "KB1", "KB1", "KB5"] and list(out["is_opportunity"]) == [False, False, False, True]
    ew = early_warning_list(out)
    assert list(ew["customer_id"]) == ["B", "C", "A"]  # CRITICAL trước, rồi HIGH theo percentile giảm dần
    assert list(early_warning_list(out, top_n=1)["customer_id"]) == ["B"]


def test_mask_customer_id():
    assert mask_customer_id("KH_0912345678") == "KH_0912***678"
    assert mask_customer_id("KH_GLW7HGXLYM") == "KH_GLW7HGXLYM"  # mã giả danh giữ nguyên


# ------------------------------------------------------------------ thang percentile (risk_rank.py)
def test_percentile_of_and_tiers():
    ref = np.arange(1, 101) / 100.0  # 100 điểm tham chiếu 0.01..1.00
    assert list(rr.percentile_of(ref, [-1.0, 0.01, 0.505, 1.0, 2.0])) == pytest.approx([0.0, 1.0, 50.0, 100.0, 100.0])
    assert np.all(np.diff(rr.percentile_of(ref, np.linspace(0, 1, 50))) >= 0)
    assert [rr.tier_of(x) for x in (95, 90, 75, 60, 10)] == ["High Risk", "Medium Risk", "Medium Risk", "Low Risk", "Low Risk"]
    assert rr.tier_counts(rr.percentile_of(ref, ref)) == {"High": 10, "Medium": 30, "Low": 60}


def test_risk_reference_roundtrip(tmp_path):
    rr.save_reference(np.array([0.3, 0.1, 0.2]), str(tmp_path), "XGBoost")
    assert list(rr.load_reference(str(tmp_path))) == [0.1, 0.2, 0.3]
    assert rr.load_reference(str(tmp_path / "khong_co")) is None
