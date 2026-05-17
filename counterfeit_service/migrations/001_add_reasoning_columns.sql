-- Миграция 001: добавление LLM-канала reasoning в predictions_async.
-- Связано с § 4.4.9.7 ВКР и реализацией ReasoningPredictor.
--
-- Применять на существующей БД production:
--   psql $DATABASE_URL -f migrations/001_add_reasoning_columns.sql
--
-- На свежей БД новые колонки создаются автоматически из ORM-метаданных
-- (app/db/models.py); этот SQL нужен только для обновления уже задеплоенной
-- инстанции, где таблица predictions_async уже существует без новых полей.

BEGIN;

ALTER TABLE predictions_async
  ADD COLUMN IF NOT EXISTS reasoning      TEXT,
  ADD COLUMN IF NOT EXISTS reasoning_mode VARCHAR(48);

-- Индекс для CRM-выгрузки автоблокировок:
--   SELECT task_id, probability, reasoning, created_at FROM predictions_async
--   WHERE reasoning_mode = 'blocking_explanation' AND created_at >= NOW() - INTERVAL '1 day';
CREATE INDEX IF NOT EXISTS idx_predictions_async_reasoning_mode
  ON predictions_async (reasoning_mode)
  WHERE reasoning_mode IS NOT NULL;

COMMIT;
