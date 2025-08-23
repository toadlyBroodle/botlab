#!/usr/bin/env bash

# Resolve Windows host IP for X server
WINIP=$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)
export DISPLAY=${DISPLAY:-"${WINIP}:0"}
export LIBGL_ALWAYS_INDIRECT=${LIBGL_ALWAYS_INDIRECT:-1}

# Default to Tor SOCKS proxy if not set
export PLAYWRIGHT_PROXY=${PLAYWRIGHT_PROXY:-"socks5://127.0.0.1:9050"}

# Launch in headed mode with safe flags for WSL2/XLaunch
exec /home/rob/Dev/botlab/.venv/bin/playwright-universal-mcp \
  --browser chromium \
  --headful \
  --debug \
  --browser-arg="--disable-gpu" \
  --browser-arg="--disable-dev-shm-usage" \
  "$@"
