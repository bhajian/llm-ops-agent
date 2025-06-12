# Dockerfile  (repo root)
FROM python:3.11-slim

WORKDIR /app

# ─── 1. copy metadata + src so editable install succeeds ───────────
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# ─── 2. copy the rest of the app directory (ui/, mcp-*, etc.) ──────
COPY . .

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
