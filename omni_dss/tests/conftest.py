import os
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from omni_dss.ml.predict import models_ready  # noqa: E402

MODELS_READY = models_ready()
requires_models = pytest.mark.skipif(not MODELS_READY, reason="Chưa huấn luyện mô hình (python omni_dss/train_pipeline.py)")
