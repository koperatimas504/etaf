#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-keyplans/main.cfg}"
OUT="${2:-runs/main}"

python -m etaf_agent.keyring train --config "${CONFIG}" --out "${OUT}"
