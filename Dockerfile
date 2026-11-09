FROM python:3.12-slim AS backend

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY backend ./backend

RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "aetheris", "serve", "--host", "0.0.0.0", "--port", "8000"]

