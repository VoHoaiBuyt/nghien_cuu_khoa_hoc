"""
Điểm rủi ro theo THỨ HẠNG (percentile) — dùng chung cho dự đoán, DSS, danh sách cảnh báo và đánh giá.

Vì sao không dùng ngưỡng xác suất cố định (0,7 / 0,4)?
  Mô hình huấn luyện với trọng số lớp nên xác suất đầu ra bị kéo lệch (trung bình ~0,55 trong khi tỷ lệ churn thật
  ~0,96). Hiệu chỉnh isotonic đưa xác suất về đúng thực tế nhưng gần như mọi khách đều > 0,9, nên ngưỡng cố định không
  còn phân biệt được ai. Mô hình chỉ đáng tin ở khả năng XẾP HẠNG (ROC-AUC), vì vậy cảnh báo dùng thứ hạng.

Định nghĩa:
  percentile(p) = % khách trong tập tham chiếu có điểm mô hình <= p  (0 = rủi ro thấp nhất, 100 = cao nhất).
  Tập tham chiếu = điểm mô hình trên tập kiểm tra out-of-time (lưu ở saved_models/risk_reference.json).
  Bậc rủi ro: Cao = top 10% (percentile > 90), Trung bình = 30% kế tiếp (60–90), Thấp = 60% còn lại.
  Cảnh báo khách tiềm năng: VIP/Tiềm năng thuộc 20% có điểm rủi ro thấp nhất (percentile <= 20).
"""

import json
import os
from typing import Iterable, Optional

import numpy as np

HIGH_PERCENTILE = 90.0         # Cao: top 10%
MEDIUM_PERCENTILE = 60.0       # Trung bình: 30% kế tiếp
OPPORTUNITY_MAX_PERCENTILE = 20.0  # tiềm năng: 20% rủi ro thấp nhất
REFERENCE_FILE = "risk_reference.json"


SCORE_DECIMALS = 6  # mô hình cây nông cho nhiều điểm TRÙNG nhau: làm tròn để sai số dấu phẩy động không đổi thứ hạng


def percentile_of(reference_sorted: np.ndarray, scores) -> np.ndarray:
    """Percentile (0–100) của từng điểm so với tập tham chiếu đã sắp xếp tăng dần."""
    ref = np.round(np.asarray(reference_sorted, dtype=float), SCORE_DECIMALS)
    s = np.round(np.atleast_1d(np.asarray(scores, dtype=float)), SCORE_DECIMALS)
    return np.round(100.0 * np.searchsorted(ref, s, side="right") / len(ref), 2)


def tier_of(percentile: float) -> str:
    if percentile > HIGH_PERCENTILE:
        return "High Risk"
    if percentile > MEDIUM_PERCENTILE:
        return "Medium Risk"
    return "Low Risk"


def tier_counts(percentiles: Iterable[float]) -> dict:
    tiers = [tier_of(p).split()[0] for p in percentiles]
    return {k: int(sum(t == k for t in tiers)) for k in ("High", "Medium", "Low")}


def save_reference(scores, saved_models_dir: str, model_name: str) -> str:
    path = os.path.join(saved_models_dir, REFERENCE_FILE)
    ref = np.sort(np.asarray(scores, dtype=float))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Điểm mô hình churn trên tập kiểm tra out-of-time, sắp tăng dần — thang đo percentile rủi ro",
            "model": model_name, "n": int(len(ref)),
            "tiers": {"high_above_percentile": HIGH_PERCENTILE, "medium_above_percentile": MEDIUM_PERCENTILE,
                      "opportunity_max_percentile": OPPORTUNITY_MAX_PERCENTILE},
            "scores": [round(float(x), SCORE_DECIMALS) for x in ref],
        }, f, ensure_ascii=False)
    return path


def load_reference(saved_models_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(saved_models_dir, REFERENCE_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return np.asarray(json.load(f)["scores"], dtype=float)
