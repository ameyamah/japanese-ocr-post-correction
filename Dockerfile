FROM ghcr.io/astral-sh/uv:0.10.8 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY.md ./
RUN uv sync --locked --no-install-project --extra cpu
COPY app ./app
RUN useradd --create-home --uid 10001 demo
USER demo
ENV PATH="/opt/venv/bin:$PATH" MODEL_DIR=/model MODEL_DEVICE=cpu
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
