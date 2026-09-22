FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir mlflow==3.16.1 \
    && useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlflow/state \
    && chown -R mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow
