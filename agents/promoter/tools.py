import os
import json
import atexit
import base64
import threading
from queue import Queue
from typing import Optional, Any, Dict, List, Callable, Tuple

from smolagents import tool
from ..utils.logger_config import setup_logger
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeoutError


LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
logger = setup_logger('promoter_tools', log_dir=LOGS_DIR)


_state: Dict[str, Any] = {
    'playwright': None,   # Playwright instance
    'browser': None,      # Browser
    'context': None,      # BrowserContext
    'page': None,         # Page
    'console_logs': [],   # Recent console logs
    'worker': None,       # Browser worker thread
}


def _headless() -> bool:
    value = os.getenv('PROMOTER_PLAYWRIGHT_HEADLESS', '1').strip().lower()
    return value not in ('0', 'false', 'no', 'off')


def _slow_mo_ms() -> int:
    try:
        return int(os.getenv('PROMOTER_PLAYWRIGHT_SLOWMO_MS', '0'))
    except Exception:
        return 0


def _tor_enabled() -> bool:
    value = os.getenv('PROMOTER_USE_TOR', '1').strip().lower()
    return value not in ('0', 'false', 'no', 'off')


class _BrowserWorker:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._task_queue: "Queue[Tuple[Callable[[Dict[str, Any]], Any], Queue]]" = Queue()
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="PlaywrightBrowserWorker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)

    def _run(self) -> None:
        try:
            logger.info("Starting Playwright browser (worker thread, direct mode)...")
            pw = sync_playwright().start()
            headless_flag = _headless()
            display_env = os.environ.get('DISPLAY', '')
            slow_mo_value = _slow_mo_ms()
            logger.info(f"Playwright launch params: headless={headless_flag}, slow_mo_ms={slow_mo_value}, DISPLAY='{display_env}'")

            # Configure Tor proxy (via tor_manager) if enabled
            proxy_settings = None
            if _tor_enabled():
                try:
                    # Import lazily to avoid hard dependency if Tor is not used
                    from agents.utils.tor.tor_manager import start_tor_if_needed, check_tor_connection, setup_tor_cleanup

                    started = start_tor_if_needed()
                    if not check_tor_connection():
                        raise RuntimeError("Tor not reachable on 127.0.0.1:9050 after startup attempt")

                    # Ensure Tor is cleaned up on exit if we started it
                    try:
                        setup_tor_cleanup()
                    except Exception:
                        pass

                    tor_server = os.getenv('PROMOTER_TOR_SOCKS', 'socks5://127.0.0.1:9050')
                    proxy_settings = {"server": tor_server}
                    logger.info(f"Using Tor proxy for Playwright: {tor_server} (started_by_us={started})")
                except Exception as e:
                    logger.warning(f"Tor requested but not available, continuing without Tor: {e}")

            launch_args = []
            if not headless_flag:
                launch_args.extend(["--disable-gpu", "--no-sandbox"])  # friendly flags for WSL/X11

            browser: Browser = pw.chromium.launch(
                headless=headless_flag,
                slow_mo=slow_mo_value,
                devtools=(not headless_flag),
                args=launch_args or None,
                proxy=proxy_settings,
            )
            context: BrowserContext = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                ignore_https_errors=True,
            )
            page: Page = context.new_page()

            # Attach console logger
            def _on_console(msg):
                try:
                    get = getattr
                    msg_type = get(msg, 'type', None)
                    msg_text = get(msg, 'text', None)
                    msg_loc = get(msg, 'location', None)
                    msg_type = msg_type() if callable(msg_type) else msg_type
                    msg_text = msg_text() if callable(msg_text) else msg_text
                    msg_loc = msg_loc() if callable(msg_loc) else msg_loc
                    _state['console_logs'].append({'type': msg_type, 'text': msg_text, 'location': msg_loc})
                    if len(_state['console_logs']) > 500:
                        _state['console_logs'] = _state['console_logs'][-500:]
                except Exception:
                    pass
            page.on("console", _on_console)

            # Publish state
            _state['playwright'] = pw
            _state['browser'] = browser
            _state['context'] = context
            _state['page'] = page

            self._ready.set()

            # Task loop
            while True:
                func, outq = self._task_queue.get()
                try:
                    result = func(_state)
                    outq.put((True, result))
                except Exception as e:
                    outq.put((False, str(e)))
        except Exception as e:
            logger.error(f"Browser worker fatal error: {e}")
            self._ready.set()

    def call(self, func: Callable[[Dict[str, Any]], Any]) -> Any:
        self.start()
        outq: "Queue[Tuple[bool, Any]]" = Queue()
        self._task_queue.put((func, outq))
        ok, payload = outq.get()
        if ok:
            return payload
        return f"Error: {payload}"


def _ensure_worker() -> _BrowserWorker:
    if _state['worker'] is None:
        _state['worker'] = _BrowserWorker()
        _state['worker'].start()
    return _state['worker']


def _shutdown() -> None:
    # Ensure teardown happens on the worker thread to avoid greenlet/thread switching errors
    worker = _state.get('worker')
    if worker is None:
        return
    def _teardown(state: Dict[str, Any]):
        try:
            if state.get('context') is not None:
                state['context'].close()
        except Exception as e:
            logger.warning(f"Error closing Playwright context: {e}")
        finally:
            state['context'] = None

        try:
            if state.get('browser') is not None:
                state['browser'].close()
        except Exception as e:
            logger.warning(f"Error closing Playwright browser: {e}")
        finally:
            state['browser'] = None

        try:
            if state.get('playwright') is not None:
                state['playwright'].stop()
        except Exception as e:
            logger.warning(f"Error stopping Playwright: {e}")
        finally:
            state['playwright'] = None
            state['page'] = None
        return "ok"
    try:
        worker.call(_teardown)
    except Exception as e:
        logger.warning(f"Worker teardown error: {e}")


atexit.register(_shutdown)


@tool
def pw_navigate(url: str, wait_until: str = "load", timeout_ms: int = 60000) -> str:
    """Navigate the browser to a URL using direct Playwright control.

    Args:
        url: The absolute URL to navigate to (for example: "https://old.reddit.com/login").
        wait_until: When to consider navigation finished. Common values: "load", "domcontentloaded", "networkidle".
        timeout_ms: Maximum time to wait for the navigation to complete, in milliseconds.

    Returns:
        A JSON string with navigation status or an error string.
    """
    worker = _ensure_worker()
    valid_waits = {"load", "domcontentloaded", "networkidle"}
    wu = wait_until if wait_until in valid_waits else "load"
    def _task(state: Dict[str, Any]):
        try:
            state['page'].goto(url, wait_until=wu, timeout=timeout_ms)
            return json.dumps({"ok": True, "url": state['page'].url})
        except PlaywrightTimeoutError:
            logger.error(f"pw_navigate timeout after {timeout_ms}ms to {url}")
            return json.dumps({"ok": False, "error": f"Timeout after {timeout_ms}ms", "url": url})
        except Exception as e:
            logger.error(f"pw_navigate error: {e}")
            return json.dumps({"ok": False, "error": str(e), "url": url})
    return worker.call(_task)


@tool
def pw_click(selector: str) -> str:
    """Click the first element matching a CSS selector on the current page.

    Args:
        selector: A CSS selector that matches the element to click (for example: "button[type='submit']").

    Returns:
        A JSON string indicating success or an error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            state['page'].locator(selector).first.click()
            return json.dumps({"ok": True, "action": "click", "selector": selector})
        except Exception as e:
            logger.error(f"pw_click error: {e}")
            return json.dumps({"ok": False, "error": str(e), "selector": selector})
    return worker.call(_task)


@tool
def pw_fill(selector: str, value: str) -> str:
    """Fill a text-like input element identified by a CSS selector.

    Args:
        selector: A CSS selector for the input/textarea to fill (for example: "input#loginUsername").
        value: The text value to type into the target element.

    Returns:
        A JSON string indicating success or an error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            # Handle shadow DOM path pattern: "<host> > shadowRoot > <inner>"
            if 'shadowRoot' in selector and '>' in selector:
                try:
                    parts = [s.strip() for s in selector.split('>')]
                    host_selector = parts[0]
                    if 'shadowRoot' in parts:
                        shadow_idx = parts.index('shadowRoot')
                        inner_selector = '>'.join(parts[shadow_idx+1:]).strip()
                    else:
                        inner_selector = parts[-1]
                    script = """
                        (hostSel, innerSel, val) => {
                          const host = document.querySelector(hostSel);
                          if (!host || !host.shadowRoot) { return { ok: false, error: 'Host or shadowRoot not found' }; }
                          const target = host.shadowRoot.querySelector(innerSel);
                          if (!target) { return { ok: false, error: 'Inner element not found' }; }
                          target.focus();
                          target.value = val;
                          target.dispatchEvent(new Event('input', { bubbles: true }));
                          target.dispatchEvent(new Event('change', { bubbles: true }));
                          return { ok: true };
                        }
                    """
                    res = state['page'].evaluate(script, host_selector, inner_selector, value)
                    if isinstance(res, dict) and res.get('ok'):
                        return json.dumps({"ok": True, "action": "fill", "selector": selector, "mode": "shadow"})
                except Exception:
                    pass
            state['page'].fill(selector, value)
            return json.dumps({"ok": True, "action": "fill", "selector": selector, "mode": "normal"})
        except Exception as e:
            logger.error(f"pw_fill error: {e}")
            return json.dumps({"ok": False, "error": str(e), "selector": selector})
    return worker.call(_task)


@tool
def pw_get_visible_text(limit: int = 20000) -> str:
    """Retrieve the visible text content of the current page.

    Args:
        limit: Maximum number of characters to return from the page text.

    Returns:
        A string containing visible text (possibly truncated), or an error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            text = state['page'].evaluate("() => document.body && document.body.innerText || ''")
            return text[: max(0, int(limit))]
        except Exception as e:
            logger.error(f"pw_get_visible_text error: {e}")
            return f"Error: {e}"
    return worker.call(_task)


@tool
def pw_get_visible_html(selector: Optional[str] = None, max_length: int = 20000) -> str:
    """Retrieve visible HTML from the current page, optionally limited to a CSS selector.

    Args:
        selector: Optional CSS selector to scope the HTML extraction to a specific container.
        max_length: Maximum number of characters to return from the HTML.

    Returns:
        A string containing HTML (possibly truncated), or an error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            if selector:
                html = state['page'].locator(selector).evaluate("el => el.outerHTML")
            else:
                html = state['page'].content()
            return html[: max(0, int(max_length))]
        except Exception as e:
            logger.error(f"pw_get_visible_html error: {e}")
            return f"Error: {e}"
    return worker.call(_task)


@tool
def pw_press_key(key: str, selector: Optional[str] = None) -> str:
    """Press a keyboard key, optionally targeting a selector first.

    Args:
        key: The key to press (e.g., 'Enter', 'Ctrl+Enter').
        selector: Optional CSS selector to focus before pressing the key.

    Returns:
        A JSON string indicating success or error.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            normalized = key.replace('Ctrl+', 'Control+').replace('cmd+', 'Meta+').replace('Cmd+', 'Meta+')
            if selector:
                state['page'].locator(selector).first.click()
            state['page'].keyboard.press(normalized)
            return json.dumps({"ok": True, "action": "press_key", "key": normalized, "selector": selector})
        except Exception as e:
            logger.error(f"pw_press_key error: {e}")
            return json.dumps({"ok": False, "error": str(e), "key": key, "selector": selector})
    return worker.call(_task)


@tool
def pw_screenshot(full_page: bool = False, selector: Optional[str] = None, max_bytes: int = 2000000) -> str:
    """Take a screenshot and return base64-encoded PNG (possibly truncated).

    Args:
        full_page: Capture the full page (ignored if selector is provided).
        selector: Optional CSS selector to capture a specific element.
        max_bytes: Maximum number of bytes to return (base64 output may be larger).

    Returns:
        Base64 string of the PNG image, possibly truncated, or an error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            if selector:
                locator = state['page'].locator(selector)
                binary: bytes = locator.screenshot()
            else:
                binary: bytes = state['page'].screenshot(full_page=full_page)
            if max_bytes and len(binary) > max_bytes:
                binary = binary[:max_bytes]
            return base64.b64encode(binary).decode('ascii')
        except Exception as e:
            logger.error(f"pw_screenshot error: {e}")
            return f"Error: {e}"
    return worker.call(_task)


@tool
def pw_evaluate(script: str, args: Optional[Any] = None) -> str:
    """Evaluate JavaScript in the page and return a JSON-serializable string.

    Args:
        script: JavaScript expression or function body string.
        args: Optional argument(s) to pass into the evaluation context. Can be any JSON-serializable value.

    Returns:
        JSON stringified result or error string.
    """
    worker = _ensure_worker()
    def _task(state: Dict[str, Any]):
        try:
            if args is None:
                result = state['page'].evaluate(script)
            else:
                result = state['page'].evaluate(script, args)
            try:
                return json.dumps(result)
            except Exception:
                return json.dumps({"repr": repr(result)})
        except Exception as e:
            logger.error(f"pw_evaluate error: {e}")
            return f"Error: {e}"
    return worker.call(_task)


@tool
def pw_console_logs(limit: int = 50, clear: bool = False) -> str:
    """Retrieve recent page console logs captured during this session.

    Args:
        limit: Maximum number of recent logs to return.
        clear: Whether to clear the stored logs after retrieval.

    Returns:
        JSON array string of log entries.
    """
    try:
        logs: List[Dict[str, Any]] = _state.get('console_logs', [])
        out = logs[-max(0, int(limit)):] if limit else logs[:]
        if clear:
            _state['console_logs'] = []
        return json.dumps(out)
    except Exception as e:
        logger.error(f"pw_console_logs error: {e}")
        return f"Error: {e}"


# --- MCP-style aliases for compatibility with prompts that expect these names ---

@tool
def mcp_playwright_playwright_navigate(url: str, wait_until: str = "load", timeout_ms: int = 60000) -> str:
    """Alias of pw_navigate. Navigate to a URL.

    Args:
        url: Absolute URL to open.
        wait_until: One of "load", "domcontentloaded", "networkidle".
        timeout_ms: Max wait time in milliseconds.

    Returns:
        JSON string with navigation status.
    """
    return pw_navigate(url=url, wait_until=wait_until, timeout_ms=timeout_ms)


@tool
def mcp_playwright_playwright_click(selector: str) -> str:
    """Alias of pw_click. Click element by CSS selector.

    Args:
        selector: CSS selector to click.

    Returns:
        JSON string indicating success or error.
    """
    return pw_click(selector=selector)


@tool
def mcp_playwright_playwright_fill(selector: str, value: str) -> str:
    """Alias of pw_fill. Fill text into an input/textarea.

    Args:
        selector: CSS selector for the input element.
        value: Text to fill.

    Returns:
        JSON string indicating success or error.
    """
    return pw_fill(selector=selector, value=value)


@tool
def mcp_playwright_playwright_press_key(key: str, selector: Optional[str] = None) -> str:
    """Alias of pw_press_key. Press a keyboard key.

    Args:
        key: Key to press (e.g., "Enter", "Control+Enter").
        selector: Optional CSS selector to focus first.

    Returns:
        JSON string indicating success or error.
    """
    return pw_press_key(key=key, selector=selector)


@tool
def mcp_playwright_playwright_screenshot(full_page: bool = False, selector: Optional[str] = None, max_bytes: int = 2000000) -> str:
    """Alias of pw_screenshot. Take a PNG screenshot and return base64.

    Args:
        full_page: Capture the whole page.
        selector: Optional CSS selector for element-only capture.
        max_bytes: Truncate binary before base64 if over this size.

    Returns:
        Base64-encoded PNG or error string.
    """
    return pw_screenshot(full_page=full_page, selector=selector, max_bytes=max_bytes)


@tool
def mcp_playwright_playwright_get_visible_text(limit: int = 20000) -> str:
    """Alias of pw_get_visible_text. Get visible page text.

    Args:
        limit: Max characters to return.

    Returns:
        Visible text string (possibly truncated) or error string.
    """
    return pw_get_visible_text(limit=limit)


@tool
def mcp_playwright_playwright_get_visible_html(selector: Optional[str] = None, max_length: int = 20000) -> str:
    """Alias of pw_get_visible_html. Get HTML content.

    Args:
        selector: Optional CSS selector to scope extraction.
        max_length: Max characters to return.

    Returns:
        HTML string (possibly truncated) or error string.
    """
    return pw_get_visible_html(selector=selector, max_length=max_length)


@tool
def mcp_playwright_playwright_evaluate(script: str, args: Optional[Any] = None) -> str:
    """Alias of pw_evaluate. Evaluate JavaScript in the page.

    Args:
        script: JavaScript to execute.
        args: Optional argument(s) to pass into the evaluation context.

    Returns:
        JSON-serialized result or error string.
    """
    return pw_evaluate(script=script, args=args)


@tool
def mcp_playwright_playwright_console_logs(limit: int = 50, clear: bool = False) -> str:
    """Alias of pw_console_logs. Return recent console logs.

    Args:
        limit: Max number of logs to return.
        clear: Clear stored logs after returning if True.

    Returns:
        JSON array string of console log entries.
    """
    return pw_console_logs(limit=limit, clear=clear)
