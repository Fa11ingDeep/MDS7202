FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY backend/pyproject.toml /app/pyproject.toml
COPY backend/uv.lock /app/uv.lock

RUN uv sync --frozen

COPY backend /app/backend
COPY modelo_final.pkl /app/modelo_final.pkl

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
