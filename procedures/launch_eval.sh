#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-keyplans/main.cfg}"
STEPS="${2:-150}"

python -m etaf_agent.keyring evaluate --config "${CONFIG}" --steps "${STEPS}"
python -m etaf_agent.keyring ablate --config "${CONFIG}" --steps "${STEPS}"
python -m etaf_agent.keyring dossier --config "${CONFIG}" --steps "${STEPS}"
