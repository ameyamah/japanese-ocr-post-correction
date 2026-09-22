#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/ocr-text-corrector
mkdir -p .cache/tmp .cache/uv .cache/huggingface workbench
if [[ ! -f pyproject.toml || ! -f uv.lock ]]; then
  printf '%s\n' 'Missing pyproject.toml or uv.lock. Follow the devcontainer guide, then rerun: bash .devcontainer/bootstrap.sh'
  exit 0
fi
# Install libraries only: this also works before the learner writes any src files.
uv sync --frozen --no-install-project --extra dev --extra ml
printf '%s\n' 'Libraries ready. In a new terminal: source .venv/bin/activate'
