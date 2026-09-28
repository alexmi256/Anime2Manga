#!/usr/bin/env bash
#
# Run the Anime2Manga pipeline with the project virtual environment.
#
# Usage:
#   scripts/anime2manga.sh input.mkv -o output
#   scripts/anime2manga.sh input.mkv -o output --start-at 02:00 --end-at 20:00
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${repo_root}/.venv/bin/python"

if [[ ! -x "${python}" ]]; then
    echo "error: virtual environment not found at ${repo_root}/.venv" >&2
    echo "       run 'just install' (or 'python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt')" >&2
    exit 1
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python}" -m anime2manga "$@"
