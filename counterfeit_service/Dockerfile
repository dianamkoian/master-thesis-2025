FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheel: экономит ~1,5 ГБ (без nvidia_cublas, triton, cuDNN).
# Используем PyTorch CPU index — wheels оптимизированы для inference без GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Download CLIP weights at build time so first request is fast
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP cached')"

# Copy application code. artifacts/ НЕ копируется в image — приходит через
# volume mount в docker-compose.yml (`./artifacts:/app/artifacts`), что
# экономит ~1 ГБ context на каждом билде и позволяет менять веса моделей
# без пересборки образа.
COPY app/ ./app/
COPY static/ ./static/

ENV ARTIFACTS_DIR=/app/artifacts
ENV COUNTERFEIT_THRESHOLD=0.5

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
