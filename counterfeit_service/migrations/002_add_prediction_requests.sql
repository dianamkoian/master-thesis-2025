-- Миграция 002: audit-log входных запросов (§ 7.5 ВКР).
-- Создаёт таблицу prediction_requests для хранения raw input запросов и
-- ссылок на изображения. Связывается с predictions_async через task_id.
--
-- На свежей БД таблица создаётся автоматически из ORM-метаданных
-- (app/db/models.py) при первом старте; этот SQL нужен для обновления
-- уже задеплоенной инстанции.

BEGIN;

CREATE TABLE IF NOT EXISTS prediction_requests (
  task_id          VARCHAR(36) PRIMARY KEY,
  mode             VARCHAR(8)  NOT NULL,        -- 'sync' | 'async'
  name             VARCHAR(512),
  description      TEXT,
  brand            VARCHAR(256),
  tab_inputs       JSONB,
  image_path       VARCHAR(512),
  image_size_bytes INTEGER,
  created_at       TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Индекс для аналитических выборок по дате (типичный запрос: «все запросы за день»).
CREATE INDEX IF NOT EXISTS idx_prediction_requests_created_at
  ON prediction_requests (created_at DESC);

-- Индекс для джойна с predictions_async через task_id уже неявно есть (PK).

COMMIT;
