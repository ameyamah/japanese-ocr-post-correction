FROM ghcr.io/astral-sh/uv:0.10.8 AS uv
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
