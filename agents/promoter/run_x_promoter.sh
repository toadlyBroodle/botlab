#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (two directories up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Activate virtual environment (required)
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Error: .venv not found at $REPO_ROOT/.venv. Create it or adjust path." >&2
  exit 1
fi
source .venv/bin/activate

# Randomization window in seconds (default 2 hours)
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"

# Random sleep within the window
SLEEP_FOR=$(( RANDOM % MAX_WAIT_SECONDS ))
printf "%s random-sleep=%s s\n" "$(date -Is)" "$SLEEP_FOR" >> agents/promoter/data/x_promoter_cron.log
sleep "$SLEEP_FOR"

# Run the promoter loop (adjust flags/config as needed)
python -m agents.promoter.x_promoter_loop \
  --config agents/promoter/data/x_promoter_config.json \
  >> agents/promoter/data/x_promoter_cron.log 2>&1


