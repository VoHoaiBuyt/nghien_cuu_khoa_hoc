"""
Hệ hỗ trợ ra quyết định (DSS) — Nội dung 4 của thuyết minh.

  CV 4.1  Tiêu chí & bộ luật : RULES — mỗi luật = (phân khúc RFM, khoảng percentile rủi ro, mức cảnh báo, kịch bản).
  CV 4.2  Dự đoán hành vi    : omni_dss/ml/predict.py (xác suất churn/mua lại + SHAP) -> percentile rủi ro (risk_rank.py).
  CV 4.3  5 kịch bản CSKH     : STRATEGIES — mỗi luật trỏ tới đúng một kịch bản.
  CV 4.4  Cảnh báo sớm        : apply_dss() + early_warning_list() — danh sách ưu tiên kèm lý do (SHAP) và hành động.

Nguyên tắc suy luận:
  - Đánh giá TẤT CẢ luật; mức cảnh báo cuối = mức CAO NHẤT trong các luật khớp (CRITICAL > HIGH > MEDIUM > LOW),
    không phụ thuộc thứ tự viết luật.
  - Đầu vào rủi ro là PERCENTILE 0–100 (thứ hạng điểm mô hình so với tập tham chiếu), không phải xác suất thô:
    mô hình huấn luyện có trọng số lớp nên xác suất bị lệch, chỉ khả năng xếp hạng là đáng tin (xem risk_rank.py).
"""

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

import pandas as pd

from omni_dss.ml.risk_rank import HIGH_PERCENTILE, MEDIUM_PERCENTILE, OPPORTUNITY_MAX_PERCENTILE

LEVEL_PRIORITY: Dict[str, int] = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

# Phân khúc: đồng bộ với nhãn K-Means (feature_engineering.SEGMENT_LABELS)
VIP, POTENTIAL, NEEDS_ATTENTION, AT_RISK, UNKNOWN = "VIP", "Tiềm năng", "Cần chăm sóc", "Rủi ro", "Không xác định"
CANONICAL_SEGMENTS = (VIP, POTENTIAL, NEEDS_ATTENTION, AT_RISK)


def _fold(text: str) -> str:
    """Chuẩn hóa để so khớp: NFC, chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng/gạch dưới."""
    text = unicodedata.normalize("NFC", str(text)).strip().lower().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return " ".join(text.replace("_", " ").replace("-", " ").split())


_SEGMENT_ALIASES = {
    "vip": VIP, "tiem nang": POTENTIAL, "potential": POTENTIAL,
    "can cham soc": NEEDS_ATTENTION, "needs attention": NEEDS_ATTENTION, "need attention": NEEDS_ATTENTION,
    "rui ro": AT_RISK, "at risk": AT_RISK, "risk": AT_RISK,
}


def normalize_segment(raw: Any) -> str:
    return UNKNOWN if raw is None else _SEGMENT_ALIASES.get(_fold(raw), UNKNOWN)


# ---------------------------------------------------------------------------
# CV 4.3 — 5 kịch bản Marketing & CSKH cá nhân hóa
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    code: str
    name: str
    objective: str
    actions: str
    kpi: str  # chỉ số để đo hiệu quả khi triển khai thật (cần nhóm đối chứng, xem notebook 04)


STRATEGIES: Dict[str, Strategy] = {s.code: s for s in (
    Strategy("KB1", "Cứu vãn khẩn cấp",
             "Giữ lại khách thuộc nhóm rủi ro rời bỏ cao nhất trước khi mất hẳn",
             "CSKH gọi điện/nhắn tin trong 24h; voucher 15–20%; VIP có chuyên viên riêng và quà tri ân",
             "Tỷ lệ đặt đơn mới trong 14 ngày sau liên hệ so với nhóm không liên hệ"),
    Strategy("KB2", "Win-back nhóm Rủi ro",
             "Kích hoạt đơn thành công đầu tiên cho khách có đơn hủy/hoàn hoặc giá trị thấp",
             "Khảo sát trải nghiệm; mã giảm 10–15% kèm freeship cho đơn kế tiếp",
             "Tỷ lệ khách có đơn thành công đầu tiên trong 30 ngày"),
    Strategy("KB3", "Giữ chân khách giá trị đang giảm tương tác",
             "Ngăn VIP/Tiềm năng trượt xuống nhóm rủi ro",
             "Quà tặng cá nhân hóa, nhắc giỏ hàng, ưu đãi freeship",
             "Tỷ lệ mua lại và giá trị đơn trung bình của nhóm VIP/Tiềm năng"),
    Strategy("KB4", "Kích hoạt lại nhu cầu",
             "Đưa khách Cần chăm sóc / chưa rõ phân khúc quay lại mua",
             "Khảo sát dịch vụ, mã giảm 10%, theo dõi tương tác 14 ngày",
             "Tỷ lệ mở/nhấp chiến dịch và tỷ lệ đặt đơn"),
    Strategy("KB5", "Tri ân & bán thêm",
             "Khai thác khách giá trị có rủi ro thấp (cơ hội tăng doanh thu)",
             "Chương trình tri ân đặc quyền, gợi ý cross-sell/up-sell, điểm thưởng kép",
             "Doanh thu bán thêm, tỷ lệ mua lại của nhóm khách tiềm năng"),
)}


# ---------------------------------------------------------------------------
# CV 4.1 — Bộ luật (ngưỡng theo percentile rủi ro, đồng bộ risk_rank.py)
#   Cao = (90, 100] (top 10%) · Trung bình = [60, 90] (30% kế tiếp) · Thấp = [0, 60) · VIP trung thành = [0, 20)
# ---------------------------------------------------------------------------
HIGH_MIN = HIGH_PERCENTILE
MEDIUM_MIN, MEDIUM_MAX = MEDIUM_PERCENTILE, HIGH_PERCENTILE
VIP_LOYAL_MAX = 20.0


@dataclass(frozen=True)
class Rule:
    rule_id: str
    level: str
    strategy: str                          # mã kịch bản CV 4.3
    segments: Optional[FrozenSet[str]]     # None = mọi phân khúc
    pct_min: float
    pct_max: float
    recommendation: str
    warning: Optional[str] = None
    exclude_segments: FrozenSet[str] = frozenset()
    min_exclusive: bool = False            # True: pct > pct_min ; False: pct >= pct_min
    max_exclusive: bool = False            # True: pct < pct_max ; False: pct <= pct_max

    def matches(self, segment: str, pct: float) -> bool:
        if segment in self.exclude_segments or (self.segments is not None and segment not in self.segments):
            return False
        above = pct > self.pct_min if self.min_exclusive else pct >= self.pct_min
        below = pct < self.pct_max if self.max_exclusive else pct <= self.pct_max
        return above and below


RULES: List[Rule] = [
    Rule("R01_VIP_CRITICAL", "CRITICAL", "KB1", frozenset({VIP}), HIGH_MIN, 100.0, min_exclusive=True,
         warning="Khách hàng VIP có nguy cơ rời bỏ rất cao",
         recommendation="Chuyên viên VIP liên hệ trong 24h, tặng voucher 20% và quà tri ân cá nhân hóa"),
    Rule("R02_HIGH_CHURN", "HIGH", "KB1", None, HIGH_MIN, 100.0, min_exclusive=True, exclude_segments=frozenset({VIP}),
         warning="Nguy cơ rời bỏ cao",
         recommendation="Gửi voucher giảm giá 15% và CSKH liên hệ trực tiếp"),
    Rule("R03_AT_RISK_MEDIUM_CHURN", "HIGH", "KB2", frozenset({AT_RISK}), MEDIUM_MIN, MEDIUM_MAX,
         warning="Khách hàng nhóm Rủi ro đang có dấu hiệu rời bỏ",
         recommendation="Kích hoạt chiến dịch win-back: mã giảm giá 15% kèm freeship cho đơn kế tiếp"),
    Rule("R04_AT_RISK_FLOOR", "MEDIUM", "KB2", frozenset({AT_RISK}), 0.0, MEDIUM_MIN, max_exclusive=True,
         warning="Khách hàng nhóm Rủi ro (giá trị/tần suất thấp) cần được theo dõi",
         recommendation="Gửi khảo sát trải nghiệm & mã giảm giá 10% để kích hoạt đơn thành công đầu tiên"),
    Rule("R05_VIP_DECLINING", "MEDIUM", "KB3", frozenset({VIP}), VIP_LOYAL_MAX, MEDIUM_MAX,
         warning="Khách hàng VIP đang giảm tần suất mua sắm",
         recommendation="Gửi quà tặng cá nhân hóa & gán chuyên viên hỗ trợ riêng"),
    Rule("R06_POTENTIAL_DECLINING", "MEDIUM", "KB3", frozenset({POTENTIAL}), MEDIUM_MIN, MEDIUM_MAX,
         warning="Khách hàng tiềm năng có dấu hiệu sụt giảm tương tác",
         recommendation="Gửi email nhắc giỏ hàng kèm ưu đãi freeship"),
    Rule("R07_NEEDS_ATTENTION", "MEDIUM", "KB4", frozenset({NEEDS_ATTENTION}), 0.0, MEDIUM_MAX,
         warning="Cần kích hoạt lại nhu cầu mua sắm",
         recommendation="Khảo sát trải nghiệm dịch vụ & gửi mã giảm giá 10%"),
    Rule("R08_UNKNOWN_MEDIUM_CHURN", "MEDIUM", "KB4", frozenset({UNKNOWN}), MEDIUM_MIN, MEDIUM_MAX,
         warning="Nguy cơ rời bỏ trung bình (chưa xác định phân khúc)",
         recommendation="Chạy phân cụm RFM để xác định phân khúc và theo dõi tương tác trong 14 ngày"),
    Rule("R09_VIP_LOYAL", "LOW", "KB5", frozenset({VIP}), 0.0, VIP_LOYAL_MAX, max_exclusive=True,
         recommendation="Mời tham gia chương trình tri ân đặc quyền"),
    Rule("R10_POTENTIAL_GROWTH", "LOW", "KB5", frozenset({POTENTIAL}), 0.0, MEDIUM_MIN, max_exclusive=True,
         recommendation="Gợi ý sản phẩm cross-sell/up-sell và tích điểm thưởng kép"),
]
DEFAULT_RECOMMENDATION = "Duy trì chăm sóc theo quy chuẩn Marketing định kỳ"

# Cảnh báo cơ hội (CV 4.4): khách giá trị thuộc 20% có điểm rủi ro THẤP NHẤT
OPPORTUNITY_SEGMENTS = (VIP, POTENTIAL)
OPPORTUNITY_CRITERION = (f"phân khúc {' / '.join(OPPORTUNITY_SEGMENTS)} và thuộc "
                         f"{OPPORTUNITY_MAX_PERCENTILE:.0f}% khách có điểm rủi ro thấp nhất")


def is_opportunity(segment: str, risk_percentile: float) -> bool:
    return segment in OPPORTUNITY_SEGMENTS and risk_percentile <= OPPORTUNITY_MAX_PERCENTILE


def evaluate_customer(rfm_segment: Any, risk_percentile: float) -> Dict[str, Any]:
    """Suy luận DSS cho một khách: phân khúc RFM + percentile rủi ro (0–100)."""
    segment = normalize_segment(rfm_segment)
    pct = min(max(float(risk_percentile), 0.0), 100.0)
    matched = sorted((r for r in RULES if r.matches(segment, pct)), key=lambda r: -LEVEL_PRIORITY[r.level])
    top = matched[0] if matched else None
    return {
        "rfm_segment": segment,
        "risk_percentile": round(pct, 2),
        "alert_level": top.level if top else "LOW",
        "strategy": top.strategy if top else None,
        "warnings": [r.warning for r in matched if r.warning],
        "recommendations": [r.recommendation for r in matched] or [DEFAULT_RECOMMENDATION],
        "matched_rules": [r.rule_id for r in matched],
        "is_opportunity": is_opportunity(segment, pct),
    }


def apply_dss(df: pd.DataFrame, segment_col: str = "rfm_segment", pct_col: str = "risk_percentile") -> pd.DataFrame:
    """Áp DSS cho cả bảng khách; thêm cột alert_level, strategy, matched_rules, recommendation, is_opportunity."""
    res = [evaluate_customer(s, p) for s, p in zip(df[segment_col], df[pct_col])]
    out = df.copy()
    out["rfm_segment"] = [r["rfm_segment"] for r in res]
    for col in ("alert_level", "strategy", "matched_rules", "is_opportunity"):
        out[col] = [r[col] for r in res]
    out["recommendation"] = [r["recommendations"][0] for r in res]
    return out


def early_warning_list(df: pd.DataFrame, levels=("CRITICAL", "HIGH"), top_n: Optional[int] = None) -> pd.DataFrame:
    """
    CV 4.4 — danh sách cảnh báo sớm: khách có mức cảnh báo trong `levels`, sắp theo (mức, percentile) giảm dần.
    `df` là đầu ra của apply_dss(); nếu có cột `top_factors` (SHAP) thì giữ làm lý do cảnh báo.
    """
    out = df[df["alert_level"].isin(levels)].copy()
    out["_prio"] = out["alert_level"].map(LEVEL_PRIORITY)
    out = out.sort_values(["_prio", "risk_percentile"], ascending=False).drop(columns="_prio")
    return out if top_n is None else out.head(top_n)


def describe_rules() -> pd.DataFrame:
    """Bảng luật dạng đọc được (cho notebook/báo cáo)."""
    rows = []
    for r in RULES:
        seg = ("Mọi phân khúc" + (f" trừ {', '.join(sorted(r.exclude_segments))}" if r.exclude_segments else "")
               if r.segments is None else ", ".join(sorted(r.segments)))
        lo, hi = ("(" if r.min_exclusive else "["), (")" if r.max_exclusive else "]")
        rows.append({"Luật": r.rule_id, "Mức": r.level, "Phân khúc": seg,
                     "Percentile rủi ro": f"{lo}{r.pct_min:g}, {r.pct_max:g}{hi}",
                     "Kịch bản": f"{r.strategy} – {STRATEGIES[r.strategy].name}",
                     "Cảnh báo": r.warning or "—", "Khuyến nghị": r.recommendation})
    return pd.DataFrame(rows)


def describe_strategies() -> pd.DataFrame:
    by = {}
    for r in RULES:
        by.setdefault(r.strategy, []).append(r.rule_id)
    return pd.DataFrame([{"Mã": s.code, "Kịch bản": s.name, "Mục tiêu": s.objective, "Hành động": s.actions,
                          "Chỉ số đo hiệu quả": s.kpi, "Luật kích hoạt": ", ".join(by.get(s.code, []))}
                         for s in STRATEGIES.values()])


def mask_customer_id(customer_id: str) -> str:
    """Mã giả danh (KH_ + base32) giữ nguyên; mã dạng số điện thoại (dữ liệu cũ) bị che: KH_0912***678."""
    prefix, sep, body = str(customer_id).rpartition("_")
    if not body.isdigit():
        return str(customer_id)
    head = f"{prefix}{sep}"
    if len(body) <= 7:
        return head + body[:1] + "***" if body else str(customer_id)
    return f"{head}{body[:4]}***{body[-3:]}"
