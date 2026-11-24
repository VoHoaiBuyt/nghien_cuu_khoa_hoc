"""
CV 6.2 — Đánh giá hiệu quả hỗ trợ ra quyết định của hệ DSS (và case study minh họa) trên TẬP KIỂM TRA OUT-OF-TIME.

Tập kiểm tra có kết quả thực tế (actual_churn: khách có mua lại trong cửa sổ kết quả hay không), nên đo trực tiếp được:
nhóm khách mà DSS cảnh báo / gợi ý có thực sự khác biệt so với phần còn lại không.

Lưu ý phương pháp: đây là đánh giá KHẢ NĂNG NHẮM ĐÚNG ĐỐI TƯỢNG (targeting) của cảnh báo. Hiệu quả nhân quả của hành động
CSKH (khách được liên hệ có mua lại nhiều hơn vì được liên hệ không) cần thử nghiệm A/B có nhóm đối chứng — ngoài phạm vi
dữ liệu lịch sử.

Đầu vào : saved_models/scored_customers.csv (sinh bởi train_pipeline.py)
Đầu ra  : saved_models/dss_evaluation.json — chỉ số liệu gộp + case study với ID đã che.
Chạy riêng: python -m omni_dss.ml.evaluate_dss
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from omni_dss.ml.config import SAVED_MODELS_DIR
from omni_dss.ml.dss import LEVELS, OPPORTUNITY_CRITERION, RULES, STRATEGIES, apply_dss, mask_customer_id

SCORED_PATH = os.path.join(SAVED_MODELS_DIR, "scored_customers.csv")
OUTPUT_PATH = os.path.join(SAVED_MODELS_DIR, "dss_evaluation.json")
N_BOOT = 2000


def _r(x: float, d: int = 4) -> float:
    return round(float(x), d)


def _group(df: pd.DataFrame, base_repurchase: float) -> Dict[str, Any]:
    n = len(df)
    rep = float(1 - df["actual_churn"].mean()) if n else float("nan")
    return {
        "customers": int(n),
        "actual_churn_rate": _r(df["actual_churn"].mean()) if n else None,
        "actual_repurchase_rate": _r(rep) if n else None,
        "repurchased_customers": int((df["actual_churn"] == 0).sum()),
        "repurchase_lift": _r(rep / base_repurchase, 2) if n and base_repurchase > 0 else None,
        "mean_risk_percentile": _r(df["risk_percentile"].mean(), 2) if n else None,
    }


def bootstrap_rate_ci(mask: np.ndarray, outcome: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> Dict[str, Any]:
    """
    Khoảng tin cậy 95% (bootstrap theo khách) cho tỷ lệ `outcome` trong nhóm `mask` và lift so với toàn bộ tập.
    Lấy mẫu lại cả tập (không chỉ trong nhóm) để phản ánh cả sai số của tỷ lệ cơ sở.
    """
    rng = np.random.default_rng(seed)
    n = len(outcome)
    rates, lifts = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m, y = mask[idx], outcome[idx]
        if m.sum() == 0 or y.mean() == 0:
            continue
        rates.append(y[m].mean())
        lifts.append(y[m].mean() / y.mean())

    def q(a):
        return [_r(np.percentile(a, 2.5)), _r(np.percentile(a, 97.5))] if a else None

    return {"rate_ci95": q(rates), "lift_ci95": q(lifts), "n_boot": len(rates)}


def cumulative_gains(score: np.ndarray, positive: np.ndarray, points: int = 20) -> List[Dict[str, float]]:
    """Xếp khách theo score giảm dần; tại mỗi % đầu danh sách: tỷ lệ số dương tính bắt được."""
    order = np.argsort(-score, kind="stable")
    pos_sorted = positive[order]
    total_pos = max(pos_sorted.sum(), 1)
    out = []
    for i in range(1, points + 1):
        k = int(round(len(score) * i / points))
        captured = pos_sorted[:k].sum()
        out.append({"population_share": _r(i / points, 3), "captured_share": _r(captured / total_pos),
                    "precision": _r(captured / max(k, 1))})
    return out


def threshold_sensitivity(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Đánh đổi quy mô danh sách – độ chính xác khi đổi ngưỡng percentile (căn cứ chọn ngưỡng)."""
    base_churn, base_rep = df["actual_churn"].mean(), 1 - df["actual_churn"].mean()
    risk = []
    for top in (5, 10, 20, 30, 40):
        g = df[df["risk_percentile"] > 100 - top]
        risk.append({"top_percent": top, "customers": int(len(g)), "actual_churn_rate": _r(g["actual_churn"].mean()),
                     "churn_lift": _r(g["actual_churn"].mean() / base_churn, 3)})
    opp = []
    valuable = df["rfm_segment"].isin(["VIP", "Tiềm năng"])
    for low in (10, 20, 30, 40, 50):
        g = df[valuable & (df["risk_percentile"] <= low)]
        rep = 1 - g["actual_churn"].mean() if len(g) else float("nan")
        opp.append({"max_percentile": low, "customers": int(len(g)),
                    "actual_repurchase_rate": _r(rep) if len(g) else None,
                    "repurchase_lift": _r(rep / base_rep, 2) if len(g) else None})
    return {"risk_top_percent": risk, "opportunity_max_percentile": opp}


def evaluate(scored: pd.DataFrame) -> Dict[str, Any]:
    df = apply_dss(scored)
    df["repurchased"] = (df["actual_churn"] == 0).astype(int)
    df["is_risk_alert"] = df["alert_level"].isin(["CRITICAL", "HIGH"])
    base_rep = float(df["repurchased"].mean())
    churn, rep = df["actual_churn"].to_numpy(), df["repurchased"].to_numpy()

    rule_coverage = []
    for rule in RULES:
        g = df[df["matched_rules"].apply(lambda m, rid=rule.rule_id: rid in m)]
        rule_coverage.append({"rule_id": rule.rule_id, "level": rule.level, "strategy": rule.strategy, **_group(g, base_rep)})

    by_strategy = {code: {"name": s.name, **_group(df[df["strategy"] == code], base_rep)} for code, s in STRATEGIES.items()}

    # Xếp theo rủi ro THẤP dần = khả năng mua lại giảm dần
    gains = cumulative_gains(-df["risk_percentile"].to_numpy(), rep)
    top = {f"top_{int(p * 100)}pct": next(g for g in gains if abs(g["population_share"] - p) < 1e-9) for p in (0.05, 0.10, 0.20)}

    return {
        "data": {
            "source": "Tập kiểm tra out-of-time (scored_customers.csv), nhãn = hành vi thực tế trong cửa sổ kết quả",
            "customers": int(len(df)),
            "base_actual_churn_rate": _r(df["actual_churn"].mean()),
            "base_actual_repurchase_rate": _r(base_rep),
        },
        "risk_alerts": {
            "definition": "Mức cảnh báo DSS CRITICAL hoặc HIGH (luật DSS trên percentile rủi ro)",
            "alerted": _group(df[df["is_risk_alert"]], base_rep),
            "not_alerted": _group(df[~df["is_risk_alert"]], base_rep),
            "churn_rate_bootstrap": bootstrap_rate_ci(df["is_risk_alert"].to_numpy(), churn),
        },
        "opportunity_alerts": {
            "definition": OPPORTUNITY_CRITERION[0].upper() + OPPORTUNITY_CRITERION[1:],
            "alerted": _group(df[df["is_opportunity"]], base_rep),
            "not_alerted": _group(df[~df["is_opportunity"]], base_rep),
            "repurchase_bootstrap": bootstrap_rate_ci(df["is_opportunity"].to_numpy(), rep),
        },
        "by_alert_level": {lvl: _group(df[df["alert_level"] == lvl], base_rep) for lvl in LEVELS},
        "by_segment": {seg: _group(g, base_rep) for seg, g in df.groupby("rfm_segment")},
        "by_strategy": by_strategy,
        "rule_coverage": rule_coverage,
        "threshold_sensitivity": threshold_sensitivity(df),
        "repurchase_ranking": {"cumulative_gains": gains, **top},
        "case_studies": case_studies(df),
        "limitations": [
            "Đo khả năng nhắm đúng đối tượng, chưa đo hiệu quả nhân quả của hành động CSKH (cần thử nghiệm A/B).",
            "Nhóm cảnh báo khách tiềm năng nhỏ nên khoảng tin cậy rộng (xem repurchase_bootstrap).",
            "Cửa sổ kết quả ngắn (vài ngày) vì dữ liệu chỉ kéo dài khoảng 5 tháng.",
            "Ngưỡng cảnh báo cơ hội (20%) được chọn sau khi xem kết quả trên chính tập kiểm tra này, nên lift ở đây có thể "
            "lạc quan; ước lượng độc lập hơn là đánh giá walk-forward (không chỉnh ngưỡng theo từng kỳ).",
            "Mô hình K-Means được huấn luyện trên RFM của toàn bộ giai đoạn (gồm cả sau mốc cắt), chỉ việc gán phân khúc "
            "cho khách kiểm tra mới dùng số liệu tại mốc cắt.",
        ],
    }


def case_studies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Một vài khách tiêu biểu (ID đã che) minh họa luồng dữ liệu -> dự đoán -> luật DSS -> khuyến nghị."""
    picks = []
    crit = df[df["alert_level"] == "CRITICAL"].sort_values("risk_percentile", ascending=False)
    opp = df[df["is_opportunity"]].sort_values("risk_percentile")
    risk_seg = df[(df["rfm_segment"] == "Rủi ro") & df["alert_level"].isin(["HIGH", "MEDIUM"])].sort_values("risk_percentile", ascending=False)
    for title, sub in (("VIP có nguy cơ rời bỏ cao", crit), ("Khách tiềm năng có xu hướng mua lại", opp),
                       ("Khách nhóm Rủi ro (chưa có/ít đơn thành công)", risk_seg)):
        if len(sub):
            r = sub.iloc[0]
            picks.append({
                "title": title,
                "display_id": mask_customer_id(str(r["customer_id"])),
                "segment": r["rfm_segment"],
                "recency_days": _r(r["recency"], 1),
                "completed_orders": int(r["frequency"]),
                "total_orders": int(r["total_orders"]) if "total_orders" in r else None,
                "monetary": float(r["monetary"]),
                "channel": r["fav_channel"],
                "region": r["region"],
                "churn_prob_pct": _r(r["churn_prob"] * 100, 2),
                "risk_percentile": _r(r["risk_percentile"], 2),
                "alert_level": r["alert_level"],
                "strategy": f"{r['strategy']} – {STRATEGIES[r['strategy']].name}" if r["strategy"] else None,
                "matched_rules": list(r["matched_rules"]),
                "recommendation": r["recommendation"],
                "actual_outcome": "Đã mua lại trong cửa sổ kết quả" if r["actual_churn"] == 0 else "Không mua lại (churn)",
            })
    return picks


def run(path: str = SCORED_PATH, out: str = OUTPUT_PATH) -> Dict[str, Any]:
    res = evaluate(pd.read_csv(path, dtype={"customer_id": str}))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


if __name__ == "__main__":
    r = run()
    d = r["data"]
    print(f"Tập kiểm tra: {d['customers']:,} khách, mua lại thực tế {d['base_actual_repurchase_rate']:.1%}")
    for k, b in (("risk_alerts", "churn_rate_bootstrap"), ("opportunity_alerts", "repurchase_bootstrap")):
        a = r[k]["alerted"]
        print(f"{k:20s} n={a['customers']:,} mua lại {a['actual_repurchase_rate']:.1%} (lift {a['repurchase_lift']}) | CI95 {r[k][b]}")
    print(f"💾 {OUTPUT_PATH}")
