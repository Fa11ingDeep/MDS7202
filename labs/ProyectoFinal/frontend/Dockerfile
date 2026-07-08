FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY frontend/pyproject.toml /app/pyproject.toml
COPY frontend/uv.lock /app/uv.lock

RUN uv sync --frozen

COPY frontend /app/frontend

CMD ["uv", "run", "python", "frontend/app.py"]
