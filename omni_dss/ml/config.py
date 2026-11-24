"""Cấu hình chung của lớp phân tích: đường dẫn kết quả và nạp biến môi trường từ omni_dss/.env (OMNI_ANON_KEY)."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../omni_dss
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def _load_env(path: str = os.path.join(BASE_DIR, ".env")) -> None:
    """Nạp KEY=VALUE từ .env (không ghi đè biến đã có trong môi trường)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
