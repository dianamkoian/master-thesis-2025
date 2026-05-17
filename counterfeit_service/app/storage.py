"""
Локальное хранилище изображений входных запросов

Архитектурный выбор. В production-системе аналогом этого модуля выступает
объектное хранилище (S3 / MinIO); в текущем прототипе используется локальная
файловая система с volume-маунтом в Docker. Контракт `save_image` остаётся
тем же и при последующей миграции на S3 — изменится только реализация.

Структура каталога:
  <STORAGE_ROOT>/
    YYYY-MM-DD/
      <task_id>.<ext>

Это даёт O(сутки) объектов в одной директории и облегчает ротацию старых
данных через `find -mtime +N -delete`.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Где складывать. По умолчанию — рядом с сервисом; в docker-compose монтируется
# как volume, чтобы переживать рестарты контейнера.
STORAGE_ROOT = Path(os.getenv("AUDIT_IMAGE_STORAGE", "/tmp/counterfeit_audit_images"))
ENABLE_IMAGE_STORAGE = os.getenv("AUDIT_IMAGE_STORAGE_ENABLE", "1") not in ("0", "false", "False")
MAX_IMAGE_BYTES = int(os.getenv("AUDIT_IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB


def _guess_ext(content_type: str | None) -> str:
    if not content_type:
        return ".bin"
    ct = content_type.lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "webp" in ct:
        return ".webp"
    return ".bin"


def save_image(task_id: str, image_bytes: bytes, content_type: str | None) -> str | None:
    """Сохранить изображение на диск. Возвращает относительный путь
    (без `STORAGE_ROOT` префикса) либо None, если хранение выключено или
    превышен размер.
    """
    if not ENABLE_IMAGE_STORAGE:
        return None
    size = len(image_bytes)
    if size > MAX_IMAGE_BYTES:
        logger.warning(
            "audit-image rejected: task_id=%s size=%d exceeds MAX_IMAGE_BYTES=%d",
            task_id, size, MAX_IMAGE_BYTES,
        )
        return None
    today = datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = STORAGE_ROOT / today
    day_dir.mkdir(parents=True, exist_ok=True)
    ext = _guess_ext(content_type)
    rel_path = f"{today}/{task_id}{ext}"
    abs_path = STORAGE_ROOT / rel_path
    abs_path.write_bytes(image_bytes)
    return rel_path
