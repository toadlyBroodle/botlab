import os
import json
import time
import subprocess
from typing import Optional

import requests
from smolagents import tool
from ..utils.logger_config import setup_logger


LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
logger = setup_logger('promoter_tools', log_dir=LOGS_DIR)


def _base_url() -> str:
    return os.getenv('PLAYWRIGHT_MCP_URL', 'http://127.0.0.1:3333')


def _wait_for_ready(timeout_sec: int = 30, interval_ms: int = 500) -> str:
    """Internal helper to poll the MCP server until it accepts HTTP connections.

    Tries a lightweight GET on /playwright/visible_text to confirm the server is up.
    Returns a status string.
    """
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            # Any response (even 4xx/5xx) means the TCP server is up
            requests.get(_base_url().rstrip('/') + '/playwright/visible_text?limit=1', timeout=interval_ms/1000)
            return "MCP server is ready"
        except Exception as e:
            last_err = e
            time.sleep(interval_ms / 1000.0)
    return f"MCP server not ready within {timeout_sec}s: {last_err}"


@tool
def mcp_playwright_launch(command: str, cwd: Optional[str] = None, wait_for_ready: bool = True, ready_timeout_sec: int = 30) -> str:
    """Launch the Playwright MCP server as a background process.

    Args:
        command: Shell command to start the MCP server (for example: `npx @modelcontextprotocol/server-playwright --port 3333`).
        cwd: Optional working directory to execute the command from. Uses current directory if not provided.
        wait_for_ready: If true, poll the MCP HTTP endpoint until it accepts connections.
        ready_timeout_sec: Max seconds to wait for server readiness when wait_for_ready is true.

    Returns:
        A status string containing the background process PID if successful, plus readiness result, or an error message.

    Usage:
        - Use when you need to start the Playwright MCP server before navigating or interacting with pages.
        - Ensure that the specified port in the command matches PLAYWRIGHT_MCP_URL.
        - If the server is already running, you can skip launching and only call `mcp_wait_for_ready`.
    """
    try:
        logger.info(f"Launching Playwright MCP: {command}")
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd or os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        status = f"MCP server launched with PID {proc.pid}"
        if wait_for_ready:
            ready = _wait_for_ready(timeout_sec=ready_timeout_sec)
            status = status + f"; {ready}"
        return status
    except Exception as e:
        logger.error(f"Failed to launch MCP server: {e}")
        return f"Error launching MCP server: {e}"


def _request(method: str, path: str, payload: Optional[dict] = None, timeout: int = 60000) -> str:
    """Low-level helper to call the Playwright MCP HTTP endpoint.

    Args:
        method: HTTP method (e.g., "GET", "POST").
        path: Endpoint path (e.g., "/navigate").
        payload: Optional JSON body to send for POST/PUT.
        timeout: Timeout in milliseconds for the HTTP request.

    Returns:
        Raw text or JSON string from the HTTP response, or an error string.
    """
    url = _base_url().rstrip('/') + '/' + path.lstrip('/')
    try:
        logger.info(f"MCP request {method} {url}")
        if method.upper() == 'GET':
            r = requests.get(url, timeout=timeout/1000)
        elif method.upper() == 'POST':
            r = requests.post(url, json=payload or {}, timeout=timeout/1000)
        else:
            r = requests.request(method.upper(), url, json=payload or {}, timeout=timeout/1000)
        r.raise_for_status()
        try:
            return json.dumps(r.json())
        except Exception:
            return r.text
    except Exception as e:
        logger.error(f"MCP request failed: {e}")
        return f"Error: {e}"


@tool
def mcp_wait_for_ready(timeout_sec: int = 30, interval_ms: int = 500) -> str:
    """Wait for the Playwright MCP HTTP server to accept connections.

    Args:
        timeout_sec: Maximum seconds to wait for readiness.
        interval_ms: Polling interval in milliseconds between attempts.

    Returns:
        A status string describing whether the server is ready.

    Usage:
        - Call after starting MCP (e.g., via `mcp_playwright_launch`) to ensure it's ready before sending commands.
        - Useful when MCP startup is slow and immediate calls would fail with connection refused.
    """
    return _wait_for_ready(timeout_sec=timeout_sec, interval_ms=interval_ms)


@tool
def pw_navigate(url: str, wait_until: str = "load", timeout_ms: int = 60000) -> str:
    """Navigate the browser to a URL via the Playwright MCP server.

    Args:
        url: The absolute URL to navigate to (for example: "https://old.reddit.com/login").
        wait_until: When to consider navigation finished. Common values: "load", "domcontentloaded", "networkidle".
        timeout_ms: Maximum time to wait for the navigation to complete, in milliseconds.

    Returns:
        A JSON string response from the MCP server, or an error string.

    Usage:
        - Call this after launching the MCP server to open a page.
        - Combine with `pw_click` / `pw_fill` to interact with the loaded page.
    """
    payload = {"url": url, "waitUntil": wait_until, "timeout": timeout_ms}
    return _request('POST', '/playwright/navigate', payload)


@tool
def pw_click(selector: str) -> str:
    """Click the first element matching a CSS selector on the current page.

    Args:
        selector: A CSS selector that matches the element to click (for example: "button[type='submit']").

    Returns:
        A JSON string response from the MCP server, or an error string.

    Usage:
        - Use after `pw_navigate` to click buttons/links.
        - If the element is inside an iframe or shadow DOM, ensure your MCP server supports deep selectors.
    """
    return _request('POST', '/playwright/click', {"selector": selector})


@tool
def pw_fill(selector: str, value: str) -> str:
    """Fill a text-like input element identified by a CSS selector.

    Args:
        selector: A CSS selector for the input/textarea to fill (for example: "input#loginUsername").
        value: The text value to type into the target element.

    Returns:
        A JSON string response from the MCP server, or an error string.

    Usage:
        - Use for login forms and search fields before submitting.
        - Combine with `pw_click` to submit forms after filling values.
    """
    return _request('POST', '/playwright/fill', {"selector": selector, "value": value})


@tool
def pw_get_visible_text(limit: int = 20000) -> str:
    """Retrieve the visible text content of the current page.

    Args:
        limit: Maximum number of characters to return from the page text.

    Returns:
        A string containing visible text (possibly truncated), or an error string.

    Usage:
        - Use to detect login prompts, 2FA screens, or to confirm page state.
    """
    return _request('GET', f'/playwright/visible_text?limit={limit}')


@tool
def pw_get_visible_html(selector: Optional[str] = None, max_length: int = 20000) -> str:
    """Retrieve sanitized visible HTML from the current page, optionally limited to a CSS selector.

    Args:
        selector: Optional CSS selector to scope the HTML extraction to a specific container.
        max_length: Maximum number of characters to return from the HTML.

    Returns:
        A string containing HTML (scripts/styles stripped depending on server settings), or an error string.

    Usage:
        - Helpful when element structure is needed to choose the correct selectors.
    """
    q = f"?maxLength={max_length}"
    if selector:
        q += f"&selector={selector}"
    return _request('GET', f'/playwright/visible_html{q}')


