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

# Optionally run via Tor (default disabled). Ensure Tor is installed on the VPS.
USE_TOR="${USE_TOR:-0}"
if [[ "$USE_TOR" == "1" ]]; then
  printf "%s tor:start_if_needed\n" "$(date -Is)" >> agents/promoter/data/x_promoter_cron.log
  python - <<'PY' >> agents/promoter/data/x_promoter_cron.log 2>&1
import sys
import shutil
from agents.utils.tor.tor_manager import check_tor_connection, start_tor_if_needed, get_tor_ip

if check_tor_connection():
    ip = get_tor_ip()
    print(f"tor:already_running ip={ip}")
else:
    # Ensure tor binary exists before attempting inline start
    tor_path = shutil.which("tor")
    if not tor_path:
        print("ERROR: Tor binary not found in PATH. Install with: sudo apt-get update && sudo apt-get install -y tor", file=sys.stderr)
        sys.exit(1)
    started = start_tor_if_needed()
    if not check_tor_connection():
        print("ERROR: Tor not running and failed to start", file=sys.stderr)
        sys.exit(1)
    ip = get_tor_ip()
    print(f"tor:started_by_script={started} ip={ip}")
PY
  TOR_FLAG="--use_tor"
else
  TOR_FLAG="--no-use_tor"
fi

# Run the promoter loop (adjust flags/config as needed)
python -m agents.promoter.x_promoter_loop \
  --config agents/promoter/data/x_promoter_config.json \
  ${TOR_FLAG} \
  >> agents/promoter/data/x_promoter_cron.log 2>&1


