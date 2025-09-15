#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-help}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
UV_ENV_CMD=(env UV_PROJECT_ENVIRONMENT="${VENV_DIR}")

usage() {
  cat <<USAGE
Usage: $0 <command>

Commands:
  bootstrap   Create uv-managed virtualenv and install project in editable mode
  install     Sync dependencies using uv based on pyproject/requirements
  lock        Export lockfile via uv pip compile to requirements.lock
  test        Run pytest inside the uv environment
  cli         Execute the vectr CLI with given arguments, e.g. "$0 cli backtest run"
  help        Show this message

This helper expects https://github.com/astral-sh/uv to be installed and available on PATH.
USAGE
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install from https://github.com/astral-sh/uv" >&2
    exit 1
  fi
}

ensure_env() {
  ensure_uv
  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating uv virtualenv in ${VENV_DIR}";
    uv venv "${VENV_DIR}"
  fi
}

run_in_env() {
  ensure_env
  "${UV_ENV_CMD[@]}" uv "$@"
}

case "${COMMAND}" in
  bootstrap)
    ensure_uv
    uv venv "${VENV_DIR}"
    "${UV_ENV_CMD[@]}" uv pip install -e "${PROJECT_DIR}[dev]"
    ;;
  install)
    run_in_env pip install -e "${PROJECT_DIR}[dev]"
    ;;
  lock)
    ensure_uv
    uv pip compile pyproject.toml --output-file requirements.lock
    ;;
  test)
    run_in_env run pytest
    ;;
  cli)
    shift || true
    run_in_env run vectr "$@"
    ;;
  help|*)
    usage
    ;;
esac
