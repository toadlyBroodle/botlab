"""
Programmatic Reddit promoter loop (reusable for multiple products)

- Performs login, search, thread navigation, and posting via direct Playwright control
  using the local wrappers in this package.
- Delegates only reply text generation to a minimal AgentLoop promoter agent (text-only).
  This avoids brittle tool-use while still leveraging LLMs for content crafting.

Quick start (JSON config):

```bash
python -m agents.promoter.reddit_promoter_loop --config agents/promoter/data/promo_reddit_csvagent_config.json
```

Override options with flags (flags override config values):

```bash
python -m agents.promoter.reddit_promoter_loop \
  --config agents/promoter/data/promo_reddit_csvagent_config.json \
  --max_comments_total 3 \
  --dry_run
```

Headless and Tor:
- `--headless` / `--no-headless` control Playwright visibility
- `--use_tor` / `--no-use_tor` toggle Tor proxy via `agent/utils/tor/tor_manager.py`

Logs are written to `agents/promoter/data/promo-agent-log.csv` by default.

Run as module directly with a custom config path:
  python -m agents.promoter.reddit_promoter_loop --config path/to/config.json

Config JSON example:
{
  "reddit_user": "username",
  "reddit_pass": "password",
  "product_name": "CSV Agent",
  "product_url": "https://csvagent.com",
  "one_line_value": "Agentic AI CSV enrichment and missing-data infill with sources",
  "positioning_bullets": [
    "Fill missing fields with sources and timestamps",
    "Validate schema and flag bad rows pre-ingest",
    "Deduplicate records with fuzzy matching",
    "Keep enriched columns clearly prefixed",
    "Budget caps per job to prevent runaway costs"
  ],
  "keywords": ["csv enrichment", "fill missing", "imputation"],
  "subreddits": ["AITools", "data", "dataanalyst", "dataengineering", "excel"],
  "exclusion_subreddits": ["webscraping"],
  "time_window_days": 30,
  "max_comments_total": 5,
  "max_comments_per_sub": 2,
  "log_csv": "agents/promoter/data/promo-agent-log.csv",
  "model_id": "gemini/gemini-2.0-flash",
  "alt_tools": ["OpenRefine", "Great Expectations", "Custom Pandas scripts"],
  "post_wait_seconds": 30,
  "use_tor": true,
  "headless": true,
  "dry_run": false
}
"""

import os
import csv
import json
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv

from .tools import (
    pw_navigate,
    pw_click,
    pw_fill,
    pw_press_key,
    pw_get_visible_text,
    pw_get_visible_html,
    pw_evaluate,
)
from ..agent_loop import AgentLoop


def _json_loads_safe(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"ok": False, "error": f"Non-JSON response: {s[:200]}"}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RedditPromoterLoop:
    """
    Programmatic controller for Reddit promotion.

    - Logs in using direct Playwright control
    - Searches sitewide and per-subreddit for recent threads
    - For fitting threads, asks an LLM agent to craft a concise reply
    - Posts the reply and records a log CSV
    """

    def __init__(
        self,
        reddit_user: str,
        reddit_pass: str,
        product_name: str,
        product_url: str,
        one_line_value: str,
        positioning_bullets: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        exclusion_subreddits: Optional[List[str]] = None,
        time_window_days: int = 30,
        max_comments_total: int = 5,
        max_comments_per_sub: int = 2,
        log_csv_path: Optional[str] = None,
        model_id: Optional[str] = None,
        alt_tools: Optional[List[str]] = None,
        post_wait_seconds: Optional[int] = None,
        page_settle_seconds: float = 2.0,
        action_settle_seconds: float = 0.5,
        dry_run: bool = False,
    ) -> None:
        load_dotenv()
        self.reddit_user = reddit_user or os.getenv("REDDIT_USER", "")
        self.reddit_pass = reddit_pass or os.getenv("REDDIT_PASS", "")

        self.product_name = product_name
        self.product_url = product_url
        self.one_line_value = one_line_value
        self.positioning_bullets = positioning_bullets or []

        self.keywords = keywords or [
            "csv enrichment",
            "fill missing",
            "imputation",
            "lead enrichment",
            "web scraping to csv",
            "pipeline validation",
        ]
        self.subreddits = subreddits or [
            "AITools",
            "data",
            "dataanalyst",
            "dataengineering",
            "excel",
        ]
        self.exclusion_subreddits = set((exclusion_subreddits or ["webscraping"]))
        self.time_window_days = max(1, int(time_window_days))
        self.max_comments_total = max(0, int(max_comments_total))
        self.max_comments_per_sub = max(0, int(max_comments_per_sub))
        self.dry_run = dry_run

        default_log = os.path.join(os.path.dirname(__file__), "data", "promo-agent-log.csv")
        self.log_csv_path = log_csv_path or default_log
        _ensure_dir(self.log_csv_path)

        self.posted_total = 0
        self.posted_per_sub: Dict[str, int] = {}

        env_model = os.getenv("PROMOTER_MODEL", "gemini/gemini-2.0-flash")
        self.model_id = model_id or env_model

        self.reply_loop = self._build_reply_loop()
        self.alt_tools = alt_tools or []
        self.post_wait_seconds = int(post_wait_seconds) if post_wait_seconds is not None else None
        self.page_settle_seconds = float(page_settle_seconds)
        self.action_settle_seconds = float(action_settle_seconds)

    def run(self) -> Dict[str, Any]:
        if not self.reddit_user or not self.reddit_pass:
            raise ValueError("Missing reddit_user/reddit_pass")

        self._login()

        seen_threads: Set[str] = set()
        sitewide_queries = [self._search_url_site(q) for q in self.keywords]
        per_sub_queries = [
            self._search_url_sub(sub, q)
            for sub in self.subreddits
            if sub not in self.exclusion_subreddits
            for q in self.keywords
        ]

        for search_url in sitewide_queries + per_sub_queries:
            if self._quotas_reached():
                break
            # Prefer old.reddit.com but tolerate response code failures, retry if Tor/pages block load
            try:
                self._navigate(search_url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)
            except Exception:
                # Fallback: try equivalent www.reddit.com URL and then convert links later
                try:
                    self._navigate(search_url.replace("old.reddit.com", "www.reddit.com"), wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)
                except Exception:
                    continue
            thread_links = self._extract_comment_links()
            for link in thread_links:
                if self._quotas_reached():
                    break
                old_url = self._to_old_reddit(link)
                if old_url in seen_threads:
                    continue
                seen_threads.add(old_url)
                if self._already_logged_thread(old_url):
                    continue
                try:
                    self._navigate(old_url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)
                except Exception:
                    try:
                        self._navigate(old_url.replace("old.reddit.com", "www.reddit.com"), wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)
                    except Exception:
                        self._log_visit(old_url, reason="navigate_failed", posted=False)
                        continue
                if self._thread_contains_product_link():
                    self._log_visit(old_url, reason="contains_product_link", posted=False)
                    continue
                sub_name, title, context = self._scrape_thread_context()
                if not sub_name:
                    sub_name = self._infer_subreddit_from_url(old_url)
                if sub_name in self.exclusion_subreddits:
                    self._log_visit(old_url, reason="excluded_subreddit", posted=False, subreddit=sub_name, title=title)
                    continue
                if self.posted_per_sub.get(sub_name, 0) >= self.max_comments_per_sub:
                    self._log_visit(old_url, reason="subreddit_quota_met", posted=False, subreddit=sub_name, title=title)
                    continue

                reply_text = self._generate_reply(sub_name, title, context)
                if not reply_text or self.product_url not in reply_text:
                    reply_text = self._fallback_reply_text(title)

                posted, permalink = self._post_reply(reply_text)
                if posted:
                    self._increment_counters(sub_name)
                    self._log_visit(old_url, reason="", posted=True, subreddit=sub_name, title=title, permalink=permalink)
                    if not self.dry_run and self.posted_total < self.max_comments_total:
                        if self.post_wait_seconds is not None:
                            time.sleep(max(0, int(self.post_wait_seconds)))
                        else:
                            time.sleep(random.randint(20, 60))
                else:
                    self._log_visit(old_url, reason="post_failed", posted=False, subreddit=sub_name, title=title)

        return {
            "status": "completed" if self.posted_total > 0 else "no_posts",
            "posted_total": int(self.posted_total),
            "posted_per_sub": {k: int(v) for k, v in self.posted_per_sub.items()},
            "log_csv": self.log_csv_path,
        }

    # --- Internal helpers ---

    def _build_reply_loop(self) -> AgentLoop:
        promoter_description = (
            "Craft a concise, value-first Reddit reply for CSV cleaning/enrichment (or analogous for the product). "
            "Do not browse. Output only the final reply text, 1–2 short sentences, include the product URL exactly once."
        )
        agent_configs = {
            "promoter_description": promoter_description,
            "promoter_prompt": promoter_description,
        }
        agent_contexts = {
            "promoter": {
                "additional_tools": [],
                "use_rate_limiting": False,
            }
        }
        return AgentLoop(
            agent_sequence=["promoter"],
            max_iterations=1,
            max_steps_per_agent=8,
            model_id=self.model_id,
            use_custom_prompts=True,
            agent_configs=agent_configs,
            agent_contexts=agent_contexts,
        )

    def _quotas_reached(self) -> bool:
        return self.posted_total >= self.max_comments_total

    def _increment_counters(self, subreddit: str) -> None:
        self.posted_total += 1
        self.posted_per_sub[subreddit] = self.posted_per_sub.get(subreddit, 0) + 1

    def _navigate(self, url: str, wait_until: Optional[str] = None, tolerate_http_errors: bool = False, retries: int = 0) -> None:
        attempt = 0
        last_err = None
        while True:
            attempt += 1
            kwargs = {}
            if wait_until:
                kwargs["wait_until"] = wait_until
            res = _json_loads_safe(pw_navigate(url, **kwargs))
            if res.get("ok"):
                if self.page_settle_seconds > 0:
                    time.sleep(self.page_settle_seconds)
                return
            last_err = res
            # Tolerate HTTP response code failures if requested
            err_str = str(res)
            if tolerate_http_errors and ("ERR_HTTP_RESPONSE_CODE_FAILURE" in err_str or "HTTP" in err_str):
                if self.page_settle_seconds > 0:
                    time.sleep(self.page_settle_seconds)
                return
            if attempt > max(0, int(retries)):
                break
            time.sleep(1.0)
        raise RuntimeError(f"Navigate failed: {last_err}")

    # Action helpers with brief settle delays
    def _fill(self, selector: str, value: str) -> bool:
        res = _json_loads_safe(pw_fill(selector, value))
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)
        return bool(res.get("ok"))

    def _click(self, selector: str) -> bool:
        res = _json_loads_safe(pw_click(selector))
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)
        return bool(res.get("ok"))

    def _press_key(self, key: str, selector: Optional[str] = None) -> None:
        _ = pw_press_key(key, selector=selector)
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)

    def _login(self) -> None:
        self._navigate("https://www.reddit.com/login", wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)

        for sel in ["input[name='username']", "input#loginUsername", "faceplate-text-input#login-username > shadowRoot > input"]:
            if self._fill(sel, self.reddit_user):
                break
        else:
            raise RuntimeError("Failed to fill username")

        for sel in ["input[name='password']", "input#loginPassword", "faceplate-text-input#login-password > shadowRoot > input"]:
            if self._fill(sel, self.reddit_pass):
                break
        else:
            raise RuntimeError("Failed to fill password")

        submit_selectors = [
            "button.login",
            "form[action*='login'] button[type='submit']",
            "button.AnimatedForm__submitButton",
        ]
        clicked = False
        for sel in submit_selectors:
            if self._click(sel):
                clicked = True
                break
        if not clicked:
            self._press_key("Enter", selector="input[name='password']")

        # Switch to old Reddit with relaxed waiting and tolerant HTTP handling (Tor may trigger non-2xx)
        try:
            self._navigate("https://old.reddit.com/", wait_until="domcontentloaded", tolerate_http_errors=True, retries=2)
        except Exception:
            # As fallback, stay on www.reddit.com; subsequent navigations will retry old.reddit.com as needed
            pass

    def _search_url_site(self, query: str) -> str:
        from urllib.parse import quote
        q = quote(query)
        return f"https://old.reddit.com/search?q={q}&sort=new&t=month"

    def _search_url_sub(self, subreddit: str, query: str) -> str:
        from urllib.parse import quote
        q = quote(query)
        sub = subreddit.strip().lstrip("r/")
        return f"https://old.reddit.com/r/{sub}/search?q={q}&restrict_sr=1&sort=new&t=month"

    def _extract_comment_links(self) -> List[str]:
        script = (
            "() => Array.from(document.querySelectorAll('a'))\n"
            "  .map(a => a.getAttribute('href') || '')\n"
            "  .filter(h => /\\/comments\\//.test(h))\n"
            "  .map(h => (h.startsWith('http') ? h : (new URL(h, location.href)).href))\n"
            "  .filter((v, i, a) => a.indexOf(v) === i)\n"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, list):
            return res
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            try:
                return json.loads(res["repr"])  # best-effort
            except Exception:
                return []
        return []

    def _to_old_reddit(self, url: str) -> str:
        if "old.reddit.com" in url:
            return url
        return url.replace("https://www.reddit.com", "https://old.reddit.com").replace("http://www.reddit.com", "https://old.reddit.com")

    def _thread_contains_product_link(self) -> bool:
        html = pw_get_visible_html(max_length=200000) or ""
        return self.product_url in html

    def _scrape_thread_context(self) -> Tuple[str, str, str]:
        sub_script = "() => (document.querySelector('a.subreddit')?.textContent || '').replace('/r/', '').trim()"
        sub = _json_loads_safe(pw_evaluate(sub_script))
        subreddit = sub if isinstance(sub, str) else (sub.get("repr") if isinstance(sub, dict) else "")

        title_script = "() => document.querySelector('a.title, a.title.may-blank, h1, .title a')?.textContent || ''"
        t = _json_loads_safe(pw_evaluate(title_script))
        title = t if isinstance(t, str) else (t.get("repr") if isinstance(t, dict) else "")

        text = pw_get_visible_text(limit=20000) or ""
        # If login flow redirected to new Reddit after success, attempt to get username or logout indicator for verification
        # Non-fatal if not present (Reddit UI varies and Tor may degrade signals)
        return (subreddit or "").strip(), (title or "").strip(), text.strip()

    def _infer_subreddit_from_url(self, url: str) -> str:
        try:
            parts = url.split("/")
            idx = parts.index("r") if "r" in parts else -1
            if idx != -1 and idx + 1 < len(parts):
                return parts[idx + 1]
            return ""
        except Exception:
            return ""

    def _already_logged_thread(self, thread_url: str) -> bool:
        if not os.path.exists(self.log_csv_path):
            return False
        try:
            with open(self.log_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("thread_url") == thread_url and row.get("comment_posted", "").lower() == "true":
                        return True
        except Exception:
            return False
        return False

    def _generate_reply(self, subreddit: str, title: str, context_text: str) -> str:
        rules = (
            "Craft a concise, value-first reply (1–2 short sentences) tailored to the OP. "
            f"Include the product link exactly once: {self.product_url}. "
            "Mention capabilities like filling missing fields, validating/flagging bad rows, and dedupe. "
            "Avoid hype; neutral tone; offer one neutral alternative if useful. Output only the reply text."
        )
        query = (
            f"Subreddit: r/{subreddit}\n"
            f"Thread title: {title}\n"
            f"Context excerpt (truncated):\n{context_text[:1200]}\n\n"
            f"PRODUCT_NAME: {self.product_name}\n"
            f"ONE_LINE_VALUE: {self.one_line_value}\n"
            f"POSITIONING_BULLETS: {json.dumps(self.positioning_bullets) if self.positioning_bullets else '[]'}\n"
            f"ALT_TOOLS: {json.dumps(self.alt_tools) if self.alt_tools else '[]'}\n"
            f"RULES: {rules}"
        )
        try:
            result = self.reply_loop.run(query)
            reply = (
                result.get("results", {}).get("promoter")
                if isinstance(result, dict)
                else None
            )
            if isinstance(reply, str):
                occurrences = reply.count(self.product_url)
                if occurrences == 0:
                    reply = reply.strip()
                    suffix = (" " if reply and not reply.endswith(('.', '!', '?')) else "")
                    reply = f"{reply}{suffix}{self.product_url}"
                elif occurrences > 1:
                    first_idx = reply.find(self.product_url)
                    reply = reply.replace(self.product_url, "")[0:first_idx] + self.product_url + reply[first_idx + len(self.product_url):].replace(self.product_url, "")
                return reply.strip()
        except Exception:
            pass
        return ""

    def _fallback_reply_text(self, title: str) -> str:
        base = (
            "Quick tip: add a light schema/missing-data check so columns don’t misalign, and keep enriched fields clearly prefixed. "
            f"If you want a simple CSV enrichment/cleanup pass (fill missing, validate/flag, dedupe), {self.product_url}"
        )
        return base

    def _post_reply(self, reply_text: str) -> Tuple[bool, str]:
        if self.dry_run:
            return True, ""
        ok = False
        for sel in ["textarea[name='text']"]:
            if self._fill(sel, reply_text):
                ok = True
                break
        if not ok:
            return False, ""

        if not self._click("form.usertext button[type='submit']"):
            self._press_key("Control+Enter")

        time.sleep(2)
        script = (
            "() => {\n"
            "  const anchors = Array.from(document.querySelectorAll('a.bylink, a[data-event-action=\\'permalink\\']'));\n"
            "  const href = anchors.length ? anchors[0].href : '';\n"
            "  return href;\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, str):
            return True, res
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            return True, res.get("repr", "")
        return True, ""

    def _log_visit(
        self,
        thread_url: str,
        reason: str,
        posted: bool,
        subreddit: str = "",
        title: str = "",
        permalink: str = "",
    ) -> None:
        exists = os.path.exists(self.log_csv_path)
        with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [
                "timestamp",
                "subreddit",
                "thread_title",
                "thread_url",
                "reason_if_skipped",
                "comment_posted",
                "comment_permalink",
                "contains_product_link",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            from datetime import datetime
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "subreddit": subreddit or "",
                "thread_title": title or "",
                "thread_url": thread_url,
                "reason_if_skipped": reason,
                "comment_posted": str(bool(posted)).lower(),
                "comment_permalink": permalink or "",
                "contains_product_link": str(self.product_url in (permalink or "")).lower(),
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Programmatic Reddit promoter loop (reusable)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--reddit_user", type=str, default=None, help="Reddit username (or set REDDIT_USER)")
    parser.add_argument("--reddit_pass", type=str, default=None, help="Reddit password (or set REDDIT_PASS)")
    parser.add_argument("--product_name", type=str, default=None, help="Product name")
    parser.add_argument("--product_url", type=str, default=None, help="Product URL")
    parser.add_argument("--one_line_value", type=str, default=None, help="One-line value proposition")
    parser.add_argument("--positioning_bullets", type=str, nargs="*", default=None, help="Bulleted differentiators/capabilities")
    parser.add_argument("--keywords", type=str, nargs="*", default=None, help="Override keyword list")
    parser.add_argument("--subreddits", type=str, nargs="*", default=None, help="Override subreddit list")
    parser.add_argument("--exclude_subs", type=str, nargs="*", default=None, help="Exclude subreddits")
    parser.add_argument("--time_window_days", type=int, default=None, help="Time window for search (days)")
    parser.add_argument("--max_comments_total", type=int, default=None, help="Maximum comments to post overall")
    parser.add_argument("--max_comments_per_sub", type=int, default=None, help="Maximum comments per subreddit")
    parser.add_argument("--log_csv", type=str, default=None, help="Path to CSV log file")
    parser.add_argument("--model_id", type=str, default=None, help="Model id for reply generation (e.g., gemini/gemini-2.0-flash)")
    parser.add_argument("--alt_tools", type=str, nargs="*", default=None, help="Optional neutral alternatives to mention")
    parser.add_argument("--post_wait_seconds", type=int, default=None, help="Fixed seconds to wait between posts (overrides 20–60s random)")
    parser.add_argument("--page_settle_seconds", type=float, default=None, help="Seconds to wait after page navigation to allow dynamic content to settle")
    parser.add_argument("--action_settle_seconds", type=float, default=None, help="Seconds to wait after click/fill/press actions")
    parser.add_argument("--use_tor", action="store_true", help="Use Tor via tor_manager (proxied Playwright)")
    parser.add_argument("--no-use_tor", dest="use_tor", action="store_false", help="Disable Tor explicitly")
    parser.set_defaults(use_tor=None)
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run Playwright with a visible browser")
    parser.set_defaults(headless=None)
    parser.add_argument("--dry_run", action="store_true", help="Do everything except actually posting the comment")
    return parser.parse_args()


def main(args: argparse.Namespace):
    cfg = _load_config(args.config)

    def pick(name: str, default: Any = None):
        val = getattr(args, name, None)
        return cfg.get(name, default) if (val is None) else val

    # Configure Tor and headless via environment before first Playwright call
    use_tor_val = pick("use_tor", None)
    if use_tor_val is not None:
        os.environ["PROMOTER_USE_TOR"] = "1" if bool(use_tor_val) else "0"

    headless_val = pick("headless", None)
    if headless_val is not None:
        os.environ["PROMOTER_PLAYWRIGHT_HEADLESS"] = "1" if bool(headless_val) else "0"

    loop = RedditPromoterLoop(
        reddit_user=pick("reddit_user", ""),
        reddit_pass=pick("reddit_pass", ""),
        product_name=pick("product_name", "Product"),
        product_url=pick("product_url", "https://example.com"),
        one_line_value=pick("one_line_value", "Value proposition"),
        positioning_bullets=pick("positioning_bullets"),
        keywords=pick("keywords"),
        subreddits=pick("subreddits"),
        exclusion_subreddits=pick("exclude_subs"),
        time_window_days=int(pick("time_window_days", 30)),
        max_comments_total=int(pick("max_comments_total", 5)),
        max_comments_per_sub=int(pick("max_comments_per_sub", 2)),
        log_csv_path=pick("log_csv"),
        model_id=pick("model_id"),
        alt_tools=pick("alt_tools"),
        post_wait_seconds=pick("post_wait_seconds"),
        page_settle_seconds=float(pick("page_settle_seconds", 2.0)) if pick("page_settle_seconds", None) is not None else 2.0,
        action_settle_seconds=float(pick("action_settle_seconds", 0.5)) if pick("action_settle_seconds", None) is not None else 0.5,
        dry_run=bool(args.dry_run or cfg.get("dry_run", False)),
    )
    result = loop.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    """Main function to parse arguments."""
    main(parse_args())


