#!/usr/bin/env bash

# Minimal flag parsing: support --no-proxy, pass all other args through
NO_PROXY=""
FORWARD_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--no-proxy" ]; then
    NO_PROXY=1
  else
    FORWARD_ARGS+=("$arg")
  fi
done

# Resolve Windows host IP for X server
WINIP=$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)
export DISPLAY=${DISPLAY:-"${WINIP}:0"}
export LIBGL_ALWAYS_INDIRECT=${LIBGL_ALWAYS_INDIRECT:-1}

# Default to Tor SOCKS proxy if not set (unless --no-proxy was provided)
if [ -z "$NO_PROXY" ]; then
  export PLAYWRIGHT_PROXY=${PLAYWRIGHT_PROXY:-"socks5://127.0.0.1:9050"}
else
  unset PLAYWRIGHT_PROXY PLAYWRIGHT_PROXY_USERNAME PLAYWRIGHT_PROXY_PASSWORD
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

# Launch in headed mode with safe flags for WSL2/XLaunch
if [ -n "$NO_PROXY" ]; then
  exec /home/rob/Dev/botlab/.venv/bin/playwright-universal-mcp \
    --browser chromium \
    --headful \
    --debug \
    --browser-arg="--disable-gpu" \
    --browser-arg="--disable-dev-shm-usage" \
    --browser-arg="--no-proxy-server" \
    --browser-arg="--proxy-bypass-list=*" \
    "${FORWARD_ARGS[@]}"
else
  exec /home/rob/Dev/botlab/.venv/bin/playwright-universal-mcp \
    --browser chromium \
    --headful \
    --debug \
    --browser-arg="--disable-gpu" \
    --browser-arg="--disable-dev-shm-usage" \
    "${FORWARD_ARGS[@]}"
fi
