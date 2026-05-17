#!/usr/bin/env bash
# redeploy_timeweb.sh — hot-swap counterfeit_service на Timeweb VPS
# Запуск:
#   ssh root@5.129.242.72
#   curl -fsSL https://<твой-gist>/redeploy_timeweb.sh -o redeploy.sh
#   chmod +x redeploy.sh && bash redeploy.sh
# Либо скопировать тело скрипта вручную.

set -euo pipefail

SERVICE_DIR=${SERVICE_DIR:-/opt/counterfeit_service}
GIT_BRANCH=${GIT_BRANCH:-main}
TS=$(date +%Y%m%d_%H%M%S)

echo "=== [1/7] cd $SERVICE_DIR"
if [ ! -d "$SERVICE_DIR" ]; then
  echo "ERROR: $SERVICE_DIR не найден. Поиск:"
  find / -maxdepth 4 -name docker-compose.yml -path "*counterfeit*" 2>/dev/null | head -5
  echo "Переопредели путь: SERVICE_DIR=/your/path bash $0"
  exit 1
fi
cd "$SERVICE_DIR"

echo "=== [2/7] backup .env"
if [ -f .env ]; then
  cp .env ".env.backup_$TS"
  echo "saved: .env.backup_$TS"
fi

echo "=== [3/7] git status / pull"
if [ -d .git ]; then
  git fetch --all
  echo "Current commit:"
  git log --oneline -1
  echo "Pulling $GIT_BRANCH..."
  git checkout "$GIT_BRANCH"
  git pull --ff-only
  echo "New commit:"
  git log --oneline -1
else
  echo "WARNING: not a git repo. Update code manually (rsync from local Mac)."
fi

echo "=== [4/7] docker compose down (БЕЗ удаления volumes)"
docker compose down

echo "=== [5/7] проверка новых env-переменных"
if [ -f .env.example ]; then
  echo "Vars in .env.example but NOT in .env:"
  comm -23 \
    <(grep -E '^[A-Z_]+=' .env.example | cut -d= -f1 | sort) \
    <(grep -E '^[A-Z_]+=' .env | cut -d= -f1 | sort 2>/dev/null) \
    || true
  echo "Если что-то перечислено выше — добавь в .env вручную и нажми Enter (Ctrl+C — отмена):"
  read -r _
fi

echo "=== [6/7] docker compose up -d --build"
docker compose up -d --build

echo "Жду healthcheck'и (max 90 сек)..."
for i in $(seq 1 30); do
  sleep 3
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "/health OK"
    break
  fi
  echo "  ...waiting ($((i*3))s)"
done

echo "=== [7/7] статус и smoke"
docker compose ps
echo ""
echo "=== /health ==="
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null | head -25 || echo "WARNING: /health not responding yet"
echo ""
echo "=== логи API (последние 30 строк) ==="
docker compose logs --tail=30 counterfeit-api

echo ""
echo "=== DONE ==="
echo "Если /predict ещё не отвечает — подожди 30 сек (lazy-load CLIP+e5),"
echo "потом проверь: curl -s -X POST http://localhost:8000/predict -F 'image=@test.png' ..."
echo ""
echo "Откат: docker compose down && git reset --hard <prev-commit> && cp .env.backup_$TS .env && docker compose up -d --build"
