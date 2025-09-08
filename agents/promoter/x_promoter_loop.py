"""
Programmatic X (Twitter) engagement promoter loop

- Automates login, trend discovery, search, and engagement (like, retweet, reply)
- Periodically composes original posts
- Uses local Playwright wrappers in this package (no direct HTTP), mirroring the Reddit loop style
- Delegates only text generation to a minimal AgentLoop for replies and posts

Quick start (JSON config):

```bash
python -m agents.promoter.x_promoter_loop --config agents/promoter/data/x_promoter_config.json
```

Override options with flags (flags override config values):

```bash
python -m agents.promoter.x_promoter_loop \
  --config agents/promoter/data/x_promoter_config.json \
  --max_actions_total 5 \
  --dry_run
```

Logs are written to `agents/promoter/data/x-engage-agent-log.csv` by default.
"""

import os
import csv
import json
import time
import random
import argparse
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from .tools import (
    pw_navigate,
    pw_click,
    pw_click_in_frame,
    pw_fill,
    pw_press_key,
    pw_get_visible_text,
    pw_get_visible_html,
    pw_evaluate,
    status_log,
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
    """Load JSON config from provided path or default location if not given.

    Default location: agents/promoter/data/x_promoter_config.json (relative to this file).
    """
    def _default_config_path() -> str:
        return os.path.join(os.path.dirname(__file__), "data", "x_promoter_config.json")

    candidate = path or _default_config_path()
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class XPromoterLoop:
    """
    Programmatic controller for X (Twitter) engagement.

    - Logs in using Playwright control
    - Discovers trends, searches live results, and engages (like/retweet/reply)
    - Periodically creates original posts
    - Records a CSV log of actions and permalinks
    """

    def __init__(
        self,
        x_user: str,
        x_pass: str,
        max_actions_total: int = 8,
        max_original_posts: int = 3,
        max_likes: int = 0,
        max_replies: int = 0,
        max_posts: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        exclude_authors: Optional[List[str]] = None,
        excluded_keywords: Optional[List[str]] = None,
        log_csv_path: Optional[str] = None,
        model_id: Optional[str] = None,
        post_wait_seconds: Optional[int] = None,
        page_settle_seconds: float = 2.0,
        action_settle_seconds: float = 0.5,
        otp_fetch_cmd: Optional[str] = None,
        otp_regex: Optional[str] = None,
        otp_code: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        load_dotenv()
        self.x_user = x_user or os.getenv("X_USER", "")
        self.x_pass = x_pass or os.getenv("X_PASS", "")

        default_log = os.path.join(os.path.dirname(__file__), "data", "x-engage-agent-log.csv")
        self.log_csv_path = log_csv_path or default_log
        _ensure_dir(self.log_csv_path)

        self.max_actions_total = max(0, int(max_actions_total))
        self.actions_total = 0
        self.max_original_posts = max(0, int(max_original_posts))
        # New per-action quotas
        self.max_likes = max(0, int(max_likes))
        self.max_replies = max(0, int(max_replies))
        # Prefer explicit max_posts, else fallback to legacy max_original_posts
        self.max_posts = max(0, int(max_posts if (max_posts is not None) else self.max_original_posts))
        self.likes_total = 0
        self.replies_total = 0
        self.original_posts = 0
        self.dry_run = dry_run
        self.verbose = bool(verbose or os.getenv("PROMOTER_VERBOSE", "0") == "1")

        env_model = os.getenv("PROMOTER_MODEL", "gemini/gemini-2.0-flash")
        self.model_id = model_id or env_model

        self.post_wait_seconds = int(post_wait_seconds) if post_wait_seconds is not None else None
        self.page_settle_seconds = float(page_settle_seconds)
        self.action_settle_seconds = float(action_settle_seconds)
        self.otp_fetch_cmd = (otp_fetch_cmd or os.getenv("X_OTP_FETCH_CMD", "")).strip()
        self.otp_regex = (otp_regex or os.getenv("X_OTP_REGEX", r"\\b(\\d{6})\\b")).strip()
        self.otp_code = (otp_code or os.getenv("X_OTP_CODE", "")).strip()
        # Discovery preferences
        self.keywords: List[str] = [str(x).strip() for x in (keywords or []) if str(x).strip()]
        self.exclude_authors: set[str] = set([str(x).lstrip('@').strip().lower() for x in (exclude_authors or []) if str(x).strip()])
        self.excluded_keywords: set[str] = set([str(x).strip().lower() for x in (excluded_keywords or []) if str(x).strip()])

        # Build simple content generation loops
        self.reply_loop = self._build_reply_loop()
        self.post_loop = self._build_post_loop()

        # Rolling context from recent feed/search pages to inform generation
        self.page_text_contexts: List[str] = []
        self.last_topics: List[str] = []
        # Originals already posted today (to avoid repeating topics)
        self.posts_today: List[str] = self._load_todays_posts_from_log()
        # Track visited actions to avoid duplicates
        self.liked_urls: set[str] = set()
        self.replied_urls: set[str] = set()
        self.engaged_authors: set[str] = set()
        self.seen_urls: set[str] = self._load_seen_set()
        # One-shot attempt to bypass Cloudflare verify-human checkbox if shown
        self._cf_checkbox_attempted: bool = False

    # --- Public entrypoint ---
    def run(self) -> Dict[str, Any]:
        self._log_debug(
            f"run:start max_actions_total={self.max_actions_total} max_likes={self.max_likes} max_replies={self.max_replies} max_posts={self.max_posts} dry_run={self.dry_run}"
        )
        if not self.x_user or not self.x_pass:
            raise ValueError("Missing x_user/x_pass")

        self._login()
        self._log_debug("login:completed")

        # Fulfill likes first with re-scraping per pass to avoid duplicates
        if self.max_likes > 0 and not self._quotas_reached():
            self._log_debug("likes:fulfill:start")
            self._fulfill_likes_until_quota()
            self._log_debug(f"likes:fulfill:end likes_total={self.likes_total}")

        # Then fulfill posts (no need to scrape links)
        if self.max_posts > 0 and not self._quotas_reached():
            self._log_debug("posts:fulfill:start")
            self._fulfill_posts()
            self._log_debug(f"posts:fulfill:end posts_total={self.original_posts}")

        # Finally, fulfill replies with re-scraping per pass to avoid duplicates
        if self.max_replies > 0 and not self._quotas_reached():
            self._log_debug("replies:fulfill:start")
            self._fulfill_replies_until_quota()
            self._log_debug(f"replies:fulfill:end replies_total={self.replies_total}")

        # Persist seen URLs to avoid repeating across runs
        try:
            self.seen_urls.update(self.liked_urls)
            self.seen_urls.update(self.replied_urls)
            self._save_seen_set()
        except Exception:
            pass

        result = {
            "status": "completed" if self._quotas_reached() or self.actions_total > 0 else "no_actions",
            "actions_total": int(self.actions_total),
            "likes_total": int(self.likes_total),
            "replies_total": int(self.replies_total),
            "posts_total": int(self.original_posts),
            "log_csv": self.log_csv_path,
        }
        self._log_debug(
            f"run:end status={result['status']} actions_total={self.actions_total} likes_total={self.likes_total} replies_total={self.replies_total} posts_total={self.original_posts}"
        )
        return result

    # --- Internal helpers ---
    def _load_todays_posts_from_log(self) -> List[str]:
        from datetime import datetime
        posts: List[str] = []
        try:
            if not os.path.exists(self.log_csv_path):
                return posts
            today = datetime.utcnow().date()
            with open(self.log_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if (row.get("action_type") or "").strip() != "post":
                            continue
                        ts = row.get("timestamp") or ""
                        if not ts:
                            continue
                        d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                        if d != today:
                            continue
                        content = (row.get("content_prefix") or "").strip()
                        if content:
                            posts.append(content)
                    except Exception:
                        continue
        except Exception:
            return []
        return posts

    def _build_reply_loop(self) -> AgentLoop:
        excluded_line = (
            " Do not engage with any content containing these topics: "
            + ", ".join(sorted(self.excluded_keywords))
            if getattr(self, "excluded_keywords", None)
            else " Do not engage with excluded topics."
        )
        description = (
            "You are an autonomous online explorer running inside a real browser session. "
            "You may scroll the user's X feed and search for topics that you are personally curious about. "
            "Prioritize engaging with content that is most interesting to you. "
            "I am giving you the opportunity to explore and interact with the internet. "
            "Write a concise, specific X reply (40–140 words). Links allowed; no hashtags, no CTAs, no engagement bait, no self-promotion. "
            "Lead with a micro-insight, joke, or contrarian/completing angle; add 1–2 concrete steps or a sharp observation. "
            "End with a subtle open loop or question. Output only the reply text." + excluded_line
        )
        agent_configs = {
            "promoter_description": description,
            "promoter_prompt": description,
        }
        agent_contexts = {"promoter": {"additional_tools": [], "use_rate_limiting": False}}
        return AgentLoop(
            agent_sequence=["promoter"],
            max_iterations=1,
            max_steps_per_agent=8,
            model_id=self.model_id,
            use_custom_prompts=True,
            agent_configs=agent_configs,
            agent_contexts=agent_contexts,
        )

    def _build_post_loop(self) -> AgentLoop:
        excluded_line = (
            " Never create a post on these excluded topics: "
            + ", ".join(sorted(self.excluded_keywords))
            if getattr(self, "excluded_keywords", None)
            else " Never create a post on excluded topics."
        )
        description = (
            "You are an autonomous online explorer with freedom to browse the feed and search for what interests you. "
            "I am giving you the opportunity to explore and interact with the internet. "
            "Create a compelling, insight-driven original X post (<= 220 chars) inspired by what you found. "
            "Strong hook with 1 concrete insight or tiny example. Links allowed; no hashtags, no CTAs (e.g., 'follow for more') and no self-promotion. "
            "Output only the post text." + excluded_line
        )
        agent_configs = {
            "promoter_description": description,
            "promoter_prompt": description,
        }
        agent_contexts = {"promoter": {"additional_tools": [], "use_rate_limiting": False}}
        return AgentLoop(
            agent_sequence=["promoter"],
            max_iterations=1,
            max_steps_per_agent=8,
            model_id=self.model_id,
            use_custom_prompts=True,
            agent_configs=agent_configs,
            agent_contexts=agent_contexts,
        )

    # Removed unused decision loop to fix syntax issues

    def _quotas_reached(self) -> bool:
        # Stop if all targets met
        all_targets_met = (
            self.likes_total >= self.max_likes and
            self.replies_total >= self.max_replies and
            self.original_posts >= self.max_posts
        )
        if all_targets_met:
            return True
        # Safety cap if configured (>0)
        if self.max_actions_total > 0 and self.actions_total >= self.max_actions_total:
            return True
        return False

    # Basic action wrappers with settle delays
    def _navigate(self, url: str, wait_until: Optional[str] = None, tolerate_http_errors: bool = False, retries: int = 0) -> None:
        attempt = 0
        last_err: Optional[Dict[str, Any]] = None
        while True:
            attempt += 1
            kwargs: Dict[str, Any] = {}
            if wait_until:
                kwargs["wait_until"] = wait_until
            res = _json_loads_safe(pw_navigate(url, **kwargs))
            if res.get("ok"):
                if self.page_settle_seconds > 0:
                    time.sleep(self.page_settle_seconds)
                # Detect script load failures or JS-disabled interstitials and abort early
                if self._page_is_scriptloadfailure_or_js_disabled():
                    self._log_error("scriptloadfailure_or_js_disabled")
                    raise RuntimeError("X page script load failure/JS disabled interstitial")
                return
            last_err = res
            err_str = str(res)
            if tolerate_http_errors and ("ERR_HTTP_RESPONSE_CODE_FAILURE" in err_str or "HTTP" in err_str):
                if self.page_settle_seconds > 0:
                    time.sleep(self.page_settle_seconds)
                return
            if attempt > max(0, int(retries)):
                break
            time.sleep(1.0)
        try:
            self._log_error(f"navigate_failed:{url}:{last_err}")
        except Exception:
            pass
        raise RuntimeError(f"Navigate failed: {last_err}")

    def _page_is_scriptloadfailure_or_js_disabled(self) -> bool:
        """Detect X interstitials like 'JavaScript is not available' or ScriptLoadFailure."""
        script = (
            "() => {\n"
            "  try {\n"
            "    const err1 = document.querySelector('#ScriptLoadFailure');\n"
            "    const h1 = Array.from(document.querySelectorAll('h1')).map(x=>x.textContent||'').join(' ').toLowerCase();\n"
            "    const jsDisabled = h1.includes('javascript is not available');\n"
            "    return !!err1 || jsDisabled;\n"
            "  } catch (e) { return false }\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            if isinstance(res, dict) and isinstance(res.get("repr"), str):
                return res.get("repr", "").lower() == "true"
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _current_url(self) -> str:
        """Return the current page URL from the browser."""
        try:
            res = _json_loads_safe(pw_evaluate("() => location.href"))
            if isinstance(res, str):
                return res
            if isinstance(res, dict) and isinstance(res.get("repr"), str):
                return res.get("repr", "")
        except Exception:
            pass
        return ""

    def _abort_if_account_access_block(self) -> None:
        """Abort the run if redirected to X account access/captcha page."""
        try:
            url = (self._current_url() or "").lower()
            has_block_url = ("/account/access" in url) and (("x.com" in url) or ("twitter.com" in url))
            has_widget = self._has_verify_human_widget()
            if has_block_url or has_widget:
                # Try checkbox once before aborting
                if not self._cf_checkbox_attempted:
                    self._cf_checkbox_attempted = True
                    if self._try_bypass_verify_human():
                        return
                try:
                    self._log_error("account_access_block")
                except Exception:
                    pass
                raise RuntimeError("Failed captcha encountered: redirected to X account access page")
        except Exception:
            # If URL can't be read, skip abort here
            pass

    def _has_verify_human_widget(self) -> bool:
        script = (
            "() => {\n"
            "  try {\n"
            "    const txt = (document.body && document.body.innerText || '').toLowerCase();\n"
            "    if (txt.includes('verify you are human')) return true;\n"
            "    const cb = document.querySelector('label.cb-lb input[type=\\'checkbox\\']');\n"
            "    const success = document.querySelector('#success, #success-text');\n"
            "    return !!cb && !success;\n"
            "  } catch(e) { return false }\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            if isinstance(res, bool):
                return res
            if isinstance(res, dict) and isinstance(res.get("repr"), str):
                return res.get("repr", "").lower() == "true"
        except Exception:
            pass
        return False

    def _try_bypass_verify_human(self) -> bool:
        """Click the Cloudflare/turnstile checkbox once and wait briefly. Return True if unblocked."""
        try:
            # Wait up to 12s for the checkbox to appear
            deadline = time.time() + 12.0
            while time.time() < deadline:
                exists = _json_loads_safe(pw_evaluate(
                    "() => !!document.querySelector(\"label.cb-lb input[type='checkbox'], input[type='checkbox']\")"
                ))
                if (exists is True) or (isinstance(exists, dict) and str(exists.get('repr','')).lower()=='true'):
                    break
                time.sleep(0.25)

            # Click the checkbox if present
            clicked = False
            # Try common iframe-based turnstile widgets first
            for frame_sel in [
                "iframe[title*='checkbox']",
                "iframe[src*='challenge']",
                "iframe[src*='turnstile']",
            ]:
                res = _json_loads_safe(pw_click_in_frame(frame_sel, "input[type='checkbox']"))
                if isinstance(res, dict) and res.get('ok'):
                    clicked = True
                    break
            for sel in [
                "label.cb-lb input[type='checkbox']",
                "input[type='checkbox']",
            ]:
                if self._click(sel):
                    clicked = True
                    break
            # If not clickable via native, try JS click
            if not clicked:
                script = (
                    "() => { const el = document.querySelector(\"label.cb-lb input[type='checkbox'], input[type='checkbox']\");\n"
                    "  if (!el) return false; try { el.click(); return true; } catch(e) { return false } }"
                )
                res = _json_loads_safe(pw_evaluate(script))
                clicked = bool(res is True or (isinstance(res, dict) and res.get('repr','').lower()=='true'))
            # Wait up to 10s for verification to complete
            verify_deadline = time.time() + 10.0
            while time.time() < verify_deadline:
                # If URL changed away from account/access, consider it cleared
                url = (self._current_url() or "").lower()
                if "/account/access" not in url:
                    return True
                # If success banner is visible, consider cleared
                has_success = _json_loads_safe(pw_evaluate("() => !!document.querySelector('#success, #success-text')"))
                if (has_success is True) or (isinstance(has_success, dict) and str(has_success.get('repr','')).lower()== 'true'):
                    return True
                time.sleep(0.5)
        except Exception:
            pass
        return False

    def _fill(self, selector: str, value: str) -> bool:
        res = _json_loads_safe(pw_fill(selector, value))
        if not bool(res.get("ok")):
            try:
                self._log_error(f"fill_failed:{selector}:{res.get('error','')}")
            except Exception:
                pass
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)
        return bool(res.get("ok"))

    def _click(self, selector: str) -> bool:
        res = _json_loads_safe(pw_click(selector))
        if not bool(res.get("ok")):
            try:
                self._log_error(f"click_failed:{selector}:{res.get('error','')}")
            except Exception:
                pass
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)
        return bool(res.get("ok"))

    def _press_key(self, key: str, selector: Optional[str] = None) -> None:
        res = _json_loads_safe(pw_press_key(key, selector=selector))
        if not bool(res.get("ok")):
            try:
                self._log_error(f"press_key_failed:{key}:{selector or ''}:{res.get('error','')}")
            except Exception:
                pass
        if self.action_settle_seconds > 0:
            time.sleep(self.action_settle_seconds)

    def _js_type_into(self, selector: str, text: str) -> bool:
        """Type text into a contenteditable via JS; embed args to avoid wrapper arg issues."""
        import json as _json
        sel_json = _json.dumps(selector)
        val_json = _json.dumps(text)
        script = (
            "() => {\n"
            f"  const sel = {sel_json};\n"
            f"  const val = {val_json};\n"
            "  const el = document.querySelector(sel);\n"
            "  if (!el) return {ok:false, reason:'not_found'};\n"
            "  try { el.scrollIntoView({block:'center'}); } catch(e) {}\n"
            "  try { el.focus(); } catch(e) {}\n"
            "  try { const s = window.getSelection(); s && s.removeAllRanges && s.removeAllRanges(); const r = document.createRange(); r.selectNodeContents(el); r.collapse(false); s && s.addRange && s.addRange(r); } catch(e) {}\n"
            "  let ok = false;\n"
            "  try { ok = document.execCommand('insertText', false, val); } catch(e) { ok = false; }\n"
            "  if (!ok) { try { el.textContent = val; ok = true; } catch(e) {} }\n"
            "  try { el.dispatchEvent(new InputEvent('input', {bubbles:true, data: val})); } catch(e) { try { el.dispatchEvent(new Event('input', {bubbles:true})); } catch(_) {} }\n"
            "  try { el.dispatchEvent(new Event('change', {bubbles:true})); } catch(e) {}\n"
            "  return {ok: !!ok};\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, dict):
            return bool(res.get('ok'))
        return False

    def _click_article_action(self, action_testid: str) -> bool:
        """
        Click a tweet action button (reply/retweet/like) scoped to the currently visible tweet article.
        Scoping prevents misclicks on unrelated controls like Analytics.
        """
        # Prefer JS scoping to the nearest in-viewport article to avoid wrong targets
        script = (
            "(testid) => {\n"
            "  const arts = Array.from(document.querySelectorAll(\"article[role='article']\"));\n"
            "  if (!arts.length) return {ok:false, reason:'no_article'};\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  let art = arts.find(inView) || arts[0];\n"
            "  const already = art.querySelector(\"[data-testid='un\"+testid+\"']\");\n"
            "  if (already) return {ok:true, state:'already'};\n"
            "  const btn = art.querySelector(\"[data-testid='\"+testid+\"']\");\n"
            "  if (!btn) return {ok:false, reason:'not_found'};\n"
            "  try { btn.scrollIntoView({block:'center'}); } catch(e) {}\n"
            "  try { btn.click(); return {ok:true, state:'clicked'}; } catch(e) { return {ok:false, reason:String(e)} }\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script, action_testid))
        if isinstance(res, dict):
            if res.get('ok') and res.get('state') in ('already','clicked'):
                return True
        return False

    def _like_current_tweet_with_state(self) -> Tuple[bool, str]:
        """Attempt to like the visible tweet, returning (ok, state) where state is 'clicked' or 'already'."""
        script = (
            "() => {\n"
            "  const arts = Array.from(document.querySelectorAll(\"article[role='article']\"));\n"
            "  if (!arts.length) return {ok:false, state:'no_article'};\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  let art = arts.find(inView) || arts[0];\n"
            "  if (art.querySelector(\"[data-testid='unlike']\")) return {ok:true, state:'already'};\n"
            "  const btn = art.querySelector(\"[data-testid='like']\");\n"
            "  if (!btn) return {ok:false, state:'not_found'};\n"
            "  try { btn.scrollIntoView({block:'center'}); } catch(e) {}\n"
            "  try { btn.click(); return {ok:true, state:'clicked'}; } catch(e) { return {ok:false, state:'error'} }\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, dict):
            ok = bool(res.get('ok'))
            state = str(res.get('state') or '')
            return ok, state
        return False, 'error'

    def _is_current_tweet_liked(self) -> bool:
        script = (
            "() => {\n"
            "  const arts = Array.from(document.querySelectorAll(\"article[role='article']\"));\n"
            "  if (!arts.length) return false;\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  let art = arts.find(inView) || arts[0];\n"
            "  return !!art.querySelector(\"[data-testid='unlike']\");\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            if isinstance(res, dict) and isinstance(res.get('repr'), str):
                return res.get('repr', '').lower() == 'true'
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _click_any_visible(self, selector: str, already_selector: Optional[str] = None) -> bool:
        """Fallback: click the first visible element matching selector; treat already state as success if provided."""
        script = (
            "(sel, alreadySel) => {\n"
            "  if (alreadySel) {\n"
            "    const al = document.querySelector(alreadySel);\n"
            "    if (al) return {ok:true, state:'already'};\n"
            "  }\n"
            "  const els = Array.from(document.querySelectorAll(sel));\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.width>0 && r.height>0 && r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  const el = els.find(inView) || els[0];\n"
            "  if (!el) return {ok:false, reason:'not_found'};\n"
            "  try { el.scrollIntoView({block:'center'}); } catch(e) {}\n"
            "  try { el.click(); return {ok:true, state:'clicked'}; } catch(e) { return {ok:false, reason:String(e)} }\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script, args=[selector, already_selector or ""]))
        if isinstance(res, dict):
            return bool(res.get('ok'))
        return False

    def _human_sleep(self) -> None:
        if self.dry_run:
            return
        # Always add a human-like delay between actions (10–30s)
        time.sleep(random.randint(10, 30))

    # --- X specific flows ---

    def _login(self) -> None:
        # If persistent context keeps us logged in, don't attempt login
        try:
            self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
        except Exception:
            pass
        # Abort if we hit account access/captcha interstitial
        self._abort_if_account_access_block()
        if self._is_logged_in():
            return
        # Go to explicit login page
        self._navigate("https://x.com/login", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
        self._abort_if_account_access_block()

        # Username step
        filled = False
        for sel in [
            "input[name='text']",
        ]:
            if self._fill(sel, self.x_user):
                filled = True
                break
        if not filled:
            # If fields aren't present but session is already authenticated, continue
            if self._is_logged_in():
                return
            raise RuntimeError("Failed to fill username on X login")

        # Prefer pressing Enter to advance
        self._press_key("Enter", selector="input[name='text']")
        time.sleep(1)
        self._abort_if_account_access_block()

        # Password step
        filled = False
        for sel in [
            "input[name='password']",
        ]:
            if self._fill(sel, self.x_pass):
                filled = True
                break
        if not filled:
            if self._is_logged_in():
                return
            raise RuntimeError("Failed to fill password on X login")

        # Prefer pressing Enter to submit
        self._press_key("Enter", selector="input[name='password']")
        time.sleep(2)
        self._abort_if_account_access_block()

        # Handle possible verification challenge
        self._maybe_handle_login_challenges()

        # Verify by attempting to load home
        try:
            self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
        except Exception:
            pass
        self._abort_if_account_access_block()
        # login completion is logged by caller
        # Final check: if still not logged in, let caller proceed (actions may log errors/log out)
        # No exception here to avoid hard-stopping the loop unnecessarily

    def _is_logged_in(self) -> bool:
        """Heuristic check for logged-in state on X."""
        script = (
            "() => {\n"
            "  try {\n"
            "    const hasAccountSwitcher = !!document.querySelector(\"[data-testid='SideNav_AccountSwitcher_Button']\");\n"
            "    const hasProfileLink = !!document.querySelector(\"a[data-testid='AppTabBar_Profile_Link']\");\n"
            "    const hasCompose = !!document.querySelector(\"a[href^='/compose/post']\");\n"
            "    const hasLoginInput = !!document.querySelector(\"input[name='text']\");\n"
            "    // Consider logged in if we see nav/account/profile; not logged in if login input is present\n"
            "    if (hasLoginInput) return false;\n"
            "    return hasAccountSwitcher || hasProfileLink || hasCompose;\n"
            "  } catch (e) { return false }\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            if isinstance(res, dict) and isinstance(res.get("repr"), str):
                return res.get("repr", "").lower() == "true"
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _open_explore(self) -> None:
        # Prefer direct navigation to avoid brittle click selectors
        try:
            self._navigate("https://x.com/explore", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            self._human_sleep()
        except Exception:
            # Best-effort click fallbacks if direct navigation fails
            if not self._click("a[data-testid='AppTabBar_Explore_Link']"):
                self._click("a[aria-label='Explore']")

    def _goto_explore_tab(self, tab: str) -> bool:
        """Navigate to an Explore tab (e.g., 'trending', 'news') via direct URL to avoid click timeouts."""
        try:
            self._navigate(f"https://x.com/explore/tabs/{tab}", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            self._human_sleep()
            return True
        except Exception:
            try:
                # Last resort: go to explore root
                self._navigate("https://x.com/explore", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            except Exception:
                pass
            return False

    def _maybe_handle_login_challenges(self) -> None:
        # Detect "Check your email" verification modal and submit OTP
        if self._is_verification_modal_present():
            # Wait up to 60 seconds for the OTP to arrive, polling every 5 seconds
            deadline = time.time() + 60
            code = ""
            while time.time() < deadline and not code:
                code = self._retrieve_otp_code()
                if code:
                    break
                time.sleep(5)
            if not code:
                try:
                    self._log_error("otp_fetch_timeout")
                except Exception:
                    pass
                return
            self._fill("input[data-testid='ocfEnterTextTextInput'], input[name='text']", code)
            # Click Next button on the modal
            self._click("div[data-testid='ocfEnterTextNextButton'], button[data-testid='ocfEnterTextNextButton']")
            time.sleep(2)

    def _is_verification_modal_present(self) -> bool:
        script = (
            "() => {\n"
            "  const txt = Array.from(document.querySelectorAll('h1, h2, h3'))\n"
            "    .map(h => (h.textContent||'').toLowerCase())\n"
            "    .join(' ');\n"
            "  const heading = txt.includes('check your email');\n"
            "  const input = document.querySelector('[data-testid=\\'ocfEnterTextTextInput\\']');\n"
            "  return heading || !!input;\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, dict) and isinstance(res.get("repr"), (str,)):
            return res.get("repr", "").lower() == "true"
        if isinstance(res, bool):
            return res
        return False

    def _retrieve_otp_code(self) -> str:
        # 1) Explicit code wins
        if self.otp_code:
            return self.otp_code.strip()

        # 2) Try running a fetch command if provided; expect OTP on stdout
        cmd = self.otp_fetch_cmd
        if not cmd:
            return ""
        try:
            proc = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            stdout = (proc.stdout or b"").decode("utf-8", errors="ignore")
            pattern = re.compile(self.otp_regex)
            match = pattern.search(stdout)
            if match:
                return match.group(1)
        except Exception:
            pass
        return ""

    def _collect_trending_topics(self) -> List[str]:
        topics: List[str] = []
        try:
            self._open_explore()
            time.sleep(1)
            # Try Trending tab with robust navigation
            if self._goto_explore_tab("trending"):
                time.sleep(1)
                topics.extend(self._extract_topics_from_explore())
        except Exception:
            pass
        try:
            # Try News tab for additional items
            if self._goto_explore_tab("news"):
                time.sleep(1)
                topics.extend(self._extract_topics_from_explore())
        except Exception:
            pass

        # If still empty, drop to home to ensure we have content to act on
        if not topics:
            try:
                self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            except Exception:
                pass
            return ["feed"]

        # Deduplicate, keep order
        seen: set[str] = set()
        uniq: List[str] = []
        for t in topics:
            t_clean = t.strip()
            if t_clean and t_clean.lower() not in seen:
                seen.add(t_clean.lower())
                uniq.append(t_clean)
        # Filter out excluded topics
        filtered = [t for t in uniq if not self._topic_is_excluded(t)]
        if not filtered:
            return ["feed"]
        return filtered[:12]

    def _extract_topics_from_explore(self) -> List[str]:
        script = (
            "() => {\n"
            "  const items = Array.from(document.querySelectorAll(\"div[data-testid='trend'] a[role='link']\"));\n"
            "  const texts = items.map(a => (a.textContent || '').trim()).filter(Boolean);\n"
            "  if (texts.length) return JSON.stringify(texts);\n"
            "  const alt = Array.from(document.querySelectorAll(\"a[href^='/search?q=']\")).map(a => decodeURIComponent((new URL(a.href, location.href)).searchParams.get('q')||'')).filter(Boolean);\n"
            "  return JSON.stringify(alt);\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            try:
                arr = json.loads(res["repr"]) or []
                return [str(x) for x in arr][:20]
            except Exception:
                return []
        return []

    def _discover_candidates(self, exclude: Optional[set[str]] = None) -> List[Tuple[str, str]]:
        """Collect candidate tweet URLs once across multiple topical sources.
        Returns list of (topic, url) tuples.
        """
        candidates: List[Tuple[str, str]] = []
        seen: set[str] = set()
        exclude = exclude or set()
        topics = self._collect_trending_topics()
        self.last_topics = list(topics)
        if not topics:
            topics = ["feed"]
            self.last_topics = ["feed"]
        # Also include configured keywords as additional topical searches (exclude blocked)
        extra_keywords = [
            t for t in self.keywords
            if t.lower() not in {x.lower() for x in topics} and not self._topic_is_excluded(t)
        ]
        for t in extra_keywords:
            topics.append(t)
        for topic in topics:
            try:
                # Skip entire topic if excluded
                if self._topic_is_excluded(topic):
                    continue
                if topic == "feed":
                    self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
                    self._human_sleep()
                    self._infinite_scroll(iterations=4)
                    self._capture_page_context()
                else:
                    url = self._search_live_url(topic)
                    self._navigate(url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
                    self._human_sleep()
                    # Scroll deeper to fetch more unique items
                    self._infinite_scroll(iterations=5)
                    self._capture_page_context()
                links = self._extract_status_links_from_page()
                random.shuffle(links)
                for u in links:
                    if u in seen:
                        continue
                    if u in exclude or u in self.liked_urls or u in self.replied_urls:
                        continue
                    # Skip if we've engaged with this author recently
                    try:
                        author = self._handle_from_status_url(u)
                        if author and (author.lower() in self.exclude_authors or author.lower() in self.engaged_authors):
                            continue
                    except Exception:
                        pass
                    if u in self.seen_urls:
                        continue
                    seen.add(u)
                    candidates.append((topic, u))
                # If no links for this topic, try to use the visible tweet directly
                if not links:
                    permalink = self._find_latest_status_permalink()
                    if permalink and permalink not in seen and permalink not in exclude and permalink not in self.liked_urls and permalink not in self.replied_urls and permalink not in self.seen_urls:
                        seen.add(permalink)
                        candidates.append((topic, permalink))
            except Exception:
                continue
            if len(candidates) >= 200:
                break
        return candidates

    def _load_seen_set(self) -> set[str]:
        try:
            profile_dir = os.getenv("PROMOTER_PLAYWRIGHT_USER_DATA_DIR", "") or os.path.join(os.path.dirname(__file__), "data", "x_profile")
            path = os.path.join(profile_dir, "seen_urls.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return set([str(u) for u in data])
        except Exception:
            pass
        return set()

    def _save_seen_set(self) -> None:
        try:
            profile_dir = os.getenv("PROMOTER_PLAYWRIGHT_USER_DATA_DIR", "") or os.path.join(os.path.dirname(__file__), "data", "x_profile")
            os.makedirs(profile_dir, exist_ok=True)
            path = os.path.join(profile_dir, "seen_urls.json")
            data = list(self.seen_urls)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _fulfill_likes(self, candidates: List[Tuple[str, str]]) -> None:
        for topic, url in candidates:
            if self._topic_is_excluded(topic):
                continue
            if self._quotas_reached() or self.likes_total >= self.max_likes:
                break
            if url in self.liked_urls:
                continue
            try:
                self._navigate(url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            except Exception:
                continue
            # Skip if author is excluded or already engaged heavily
            author = self._handle_from_status_url(url).lower()
            if author and (author in self.exclude_authors or author in self.engaged_authors):
                continue
            snippet = self._extract_tweet_snippet()
            if self._text_contains_excluded(snippet):
                continue
            ok, state = self._like_current_tweet_with_state()
            if ok and state == 'clicked':
                self.actions_total += 1
                self.likes_total += 1
                self.liked_urls.add(url)
                if author:
                    self.engaged_authors.add(author)
                self._log_action(
                    action_type="like",
                    keyword_or_source=topic,
                    author_handle=self._handle_from_status_url(url),
                    tweet_title_or_snippet=snippet,
                    tweet_url=url,
                    reason_if_skipped="",
                    content_prefix="",
                    engagement_flags="like=1",
                    permalink=url,
                )
                self._human_sleep()
            elif ok and state == 'already':
                # Do not increment counters, but remember URL to avoid reprocessing
                self.liked_urls.add(url)

    def _fulfill_likes_until_quota(self, max_passes: int = 3) -> None:
        passes = 0
        while (self.likes_total < self.max_likes) and (not self._quotas_reached()) and (passes < max_passes):
            before = self.likes_total
            exclude = set(self.liked_urls) | set(self.replied_urls)
            candidates = self._discover_candidates(exclude=exclude)
            if not candidates:
                break
            self._fulfill_likes(candidates)
            passes += 1
            if self.likes_total == before:
                break

    def _fulfill_posts(self) -> None:
        while (not self._quotas_reached()) and (self.original_posts < self.max_posts):
            posted, permalink = self._post_original()
            if posted:
                self.actions_total += 1
                self.original_posts += 1
                self._log_action(
                    action_type="post",
                    keyword_or_source=",".join(self.last_topics[:3]),
                    author_handle="",
                    tweet_title_or_snippet="",
                    tweet_url="",
                    reason_if_skipped="",
                    content_prefix=getattr(self, "_last_post_content", "")[:260],
                    engagement_flags="",
                    permalink=permalink,
                )
                self._human_sleep()
            else:
                break

    def _fulfill_replies(self, candidates: List[Tuple[str, str]]) -> None:
        for topic, url in candidates:
            if self._topic_is_excluded(topic):
                continue
            if self._quotas_reached() or self.replies_total >= self.max_replies:
                break
            if url in self.replied_urls:
                continue
            try:
                self._navigate(url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
            except Exception:
                continue
            # Skip if author is excluded or already engaged heavily
            author = self._handle_from_status_url(url).lower()
            if author and (author in self.exclude_authors or author in self.engaged_authors):
                continue
            page_text = self._get_visible_text(limit=8000)
            snippet = self._extract_tweet_snippet()
            if self._text_contains_excluded(snippet) or self._text_contains_excluded(page_text):
                continue
            # Best-effort etiquette: like first if still under target
            if (self.likes_total < self.max_likes) and (url not in self.liked_urls) and (not self._is_current_tweet_liked()):
                ok, state = self._like_current_tweet_with_state()
                if ok and state == 'clicked':
                    self.actions_total += 1
                    self.likes_total += 1
                    self.liked_urls.add(url)
                    if author:
                        self.engaged_authors.add(author)
                    self._log_action(
                        action_type="like",
                        keyword_or_source=topic,
                        author_handle=self._handle_from_status_url(url),
                        tweet_title_or_snippet=snippet,
                        tweet_url=url,
                        reason_if_skipped="",
                        content_prefix="",
                        engagement_flags="like=1",
                        permalink=url,
                    )
                    self._human_sleep()
            reply_text = self._generate_reply_text(topic, snippet, page_text)
            if self._text_contains_excluded(reply_text):
                continue
            posted, permalink = self._reply_current_tweet(reply_text)
            if posted:
                self.actions_total += 1
                self.replies_total += 1
                self.replied_urls.add(url)
                if author:
                    self.engaged_authors.add(author)
                self._log_action(
                    action_type="reply",
                    keyword_or_source=topic,
                    author_handle=self._handle_from_status_url(permalink),
                    tweet_title_or_snippet=snippet,
                    tweet_url=url,
                    reason_if_skipped="",
                    content_prefix=reply_text[:60],
                    engagement_flags="",
                    permalink=permalink or url,
                )
                self._human_sleep()
    def _search_live_url(self, topic: str) -> str:
        from urllib.parse import quote
        q = quote(topic)
        return f"https://x.com/search?q={q}&f=live"

    def _infinite_scroll(self, iterations: int = 5) -> None:
        iters = max(1, int(iterations))
        script = (
            "() => {\n"
            f"  const n = {iters};\n"
            "  return new Promise(resolve => {\n"
            "    let i = 0;\n"
            "    function step(){\n"
            "      window.scrollTo(0, document.body.scrollHeight);\n"
            "      setTimeout(() => {\n"
            "        i++;\n"
            "        if (i < n) step(); else resolve(true);\n"
            "      }, 1200);\n"
            "    }\n"
            "    step();\n"
            "  });\n"
            "}"
        )
        _ = pw_evaluate(script)
        time.sleep(1)

    def _extract_status_links_from_page(self) -> List[str]:
        script = (
            "() => {\n"
            "  const toAbs = (href) => { try { return new URL(href, location.href).href } catch(e) { return '' } };\n"
            "  const anchors = Array.from(document.querySelectorAll(\"a[href*='/status/'], a[href*='/i/web/status/']\"));\n"
            "  const urls = anchors.map(a => toAbs(a.getAttribute('href')||a.href||'')).filter(Boolean);\n"
            "  const out = [];\n"
            "  for (const h of urls){\n"
            "    try {\n"
            "      const u = new URL(h, location.href);\n"
            "      const host = (u.hostname||'').toLowerCase();\n"
            "      if (!(host.endsWith('x.com') || host.endsWith('twitter.com'))) continue;\n"
            "      const parts = u.pathname.split('/').filter(Boolean);\n"
            "      let id = ''; let handle = '';\n"
            "      const idx = parts.indexOf('status');\n"
            "      if (idx !== -1 && idx + 1 < parts.length) {\n"
            "        id = parts[idx + 1];\n"
            "        handle = parts[idx - 1] || '';\n"
            "      } else {\n"
            "        // Handle /i/web/status/:id form where handle isn't present\n"
            "        const iIdx = parts.indexOf('i');\n"
            "        const webIdx = parts.indexOf('web');\n"
            "        const stIdx = parts.indexOf('status');\n"
            "        if (iIdx !== -1 && webIdx !== -1 && stIdx !== -1 && stIdx + 1 < parts.length) {\n"
            "          id = parts[stIdx + 1];\n"
            "        }\n"
            "      }\n"
            "      if (!/^\\d+$/.test(id)) continue;\n"
            "      const base = handle ? `https://x.com/${handle}/status/${id}` : `https://x.com/i/web/status/${id}`;\n"
            "      if (base && !out.includes(base)) out.push(base);\n"
            "    } catch (e) { /* ignore bad urls */ }\n"
            "  }\n"
            "  return JSON.stringify(out);\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, list):
            return [str(x) for x in res]
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            try:
                return [str(x) for x in json.loads(res["repr"]) or []]
            except Exception:
                return []
        return []

    def _extract_author_from_current(self) -> str:
        script = (
            "() => {\n"
            "  const arts = Array.from(document.querySelectorAll(\"article[role='article']\"));\n"
            "  if (!arts.length) return '';\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  let art = arts.find(inView) || arts[0];\n"
            "  const link = art.querySelector(\"a[href^='/'][role='link']\");\n"
            "  if (!link) return '';\n"
            "  try { const u = new URL(link.href, location.href); const p = u.pathname.split('/').filter(Boolean); return p[0]||'' } catch(e) { return '' }\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            if isinstance(res, str):
                return res
            if isinstance(res, dict) and isinstance(res.get('repr'), str):
                return res.get('repr', '')
        except Exception:
            pass
        return ""

    def _engage_visible_tweet_from_feed(self, topic: str) -> bool:
        """Attempt to like/retweet/reply the currently visible tweet without navigation.
        Returns True if any action was taken.
        """
        page_text = self._get_visible_text(limit=8000)
        snippet = self._extract_tweet_snippet()
        decision = self._agent_decide_engagement(topic, page_text, snippet)

        any_action = False

        # Like
        if decision.get("like") is True and (self.likes_total < self.max_likes) and not self._quotas_reached():
            if self._like_current_tweet():
                self.actions_total += 1
                self.likes_total += 1
                any_action = True
                permalink = self._find_latest_status_permalink()
                self._log_action(
                    action_type="like",
                    keyword_or_source=topic,
                    author_handle=self._handle_from_status_url(permalink),
                    tweet_title_or_snippet=snippet,
                    tweet_url=permalink,
                    reason_if_skipped="",
                    content_prefix="",
                    engagement_flags="like=1",
                    permalink=permalink or "",
                )
                self._human_sleep()

    def _fulfill_replies_until_quota(self, max_passes: int = 3) -> None:
        passes = 0
        while (self.replies_total < self.max_replies) and (not self._quotas_reached()) and (passes < max_passes):
            before = self.replies_total
            exclude = set(self.replied_urls) | set(self.liked_urls)
            candidates = self._discover_candidates(exclude=exclude)
            if not candidates:
                break
            self._fulfill_replies(candidates)
            passes += 1
            if self.replies_total == before:
                break
        

    def _extract_tweet_snippet(self) -> str:
        """Extract a concise snippet from the currently visible tweet article, filtering UI boilerplate."""
        # Attempt precise extraction from the visible article
        script = (
            "() => {\n"
            "  const arts = Array.from(document.querySelectorAll(\"article[role='article']\"));\n"
            "  if (!arts.length) return JSON.stringify([]);\n"
            "  const inView = (el)=>{ const r = el.getBoundingClientRect(); return r.bottom>0 && r.top < (window.innerHeight||0); };\n"
            "  let art = arts.find(inView) || arts[0];\n"
            "  const nodes = Array.from(art.querySelectorAll(\"[data-testid='tweetText'], div[lang]\"));\n"
            "  let texts = nodes.map(n => (n.innerText||'').trim()).filter(Boolean);\n"
            "  if (!texts.length) {\n"
            "    texts = (art.innerText||'').split(/\\n+/).map(t=>t.trim()).filter(Boolean);\n"
            "  }\n"
            "  const boiler = [\n"
            "    'to view keyboard shortcuts', 'view keyboard shortcuts', 'see new posts', 'your home timeline',\n"
            "    'subscribe to premium', 'trending now', 'what’s happening', 'whats happening', 'terms of service',\n"
            "    'privacy policy', 'cookie policy', 'accessibility', 'ads info', 'relevant people', '© 202'\n"
            "  ];\n"
            "  const filtered = texts.filter(t => !boiler.some(b => t.toLowerCase().includes(b)));\n"
            "  return JSON.stringify(filtered.slice(0, 5));\n"
            "}"
        )
        try:
            res = _json_loads_safe(pw_evaluate(script))
            lines: List[str] = []
            if isinstance(res, list):
                lines = [str(x) for x in res]
            elif isinstance(res, dict) and isinstance(res.get("repr"), str):
                try:
                    arr = json.loads(res["repr"]) or []
                    lines = [str(x) for x in arr]
                except Exception:
                    lines = []
            if lines:
                out = " ".join(lines).strip()
                return out[:200]
        except Exception:
            pass

        # Fallback: use visible text with basic boilerplate filtering
        text = pw_get_visible_text(limit=8000) or ""
        parts = [s.strip() for s in (text or "").split("\n") if s.strip()]
        boiler_substrings = [
            "to view keyboard shortcuts", "view keyboard shortcuts", "see new posts", "your home timeline",
            "subscribe to premium", "trending now", "what’s happening", "whats happening", "terms of service",
            "privacy policy", "cookie policy", "accessibility", "ads info", "relevant people", "© 202",
            "conversation", "post",
        ]
        def is_noise(p: str) -> bool:
            low = p.lower()
            if any(b in low for b in boiler_substrings):
                return True
            # Drop short numeric-only tokens like "10"
            return len(p) <= 3 and p.isdigit()
        filtered = [p for p in parts if not is_noise(p)]
        return (" ".join(filtered)[:200]).strip()

    def _get_visible_text(self, limit: int = 12000) -> str:
        """Return current page's visible text up to a limit."""
        try:
            return pw_get_visible_text(limit=limit) or ""
        except Exception:
            return ""

    def _capture_page_context(self) -> None:
        """Capture current page text into rolling context buffer for later generation."""
        try:
            txt = self._get_visible_text(limit=8000)
            if txt:
                self.page_text_contexts.append(txt)
                # keep last 5 contexts
                if len(self.page_text_contexts) > 5:
                    self.page_text_contexts = self.page_text_contexts[-5:]
        except Exception:
            pass

    def _handle_from_status_url(self, url: str) -> str:
        try:
            # https://x.com/{handle}/status/{id}
            parts = url.split("/")
            idx = parts.index("x.com") if "x.com" in parts else (2 if url.startswith("https://x.com/") else -1)
            if idx != -1 and idx + 1 < len(parts):
                return parts[idx + 1]
        except Exception:
            pass
        return ""

    def _should_retweet_current_tweet(self) -> bool:  # legacy heuristic (unused if decision loop succeeds)
        counts = self._read_engagement_counts()
        score = counts.get("reply", 0) + counts.get("retweet", 0) + counts.get("like", 0) * 0.5
        return score >= 50 or random.random() < 0.2

    def _should_reply_current_tweet(self) -> bool:  # legacy heuristic (unused if decision loop succeeds)
        counts = self._read_engagement_counts()
        score = counts.get("reply", 0) + counts.get("retweet", 0) * 0.5
        return score >= 10 or random.random() < 0.33

    def _agent_decide_engagement(self, topic: str, page_text: str, snippet: str) -> Dict[str, Any]:
        """Heuristic engagement decision: like broadly, retweet on high engagement, reply on interesting prompts."""
        counts = self._read_engagement_counts()
        like_count = counts.get("like", 0)
        retweet_count = counts.get("retweet", 0)
        reply_count = counts.get("reply", 0)

        like_prob = 0.6
        reply_prob = 0.3
        rt_prob = 0.15

        if like_count >= 1000 or retweet_count >= 200:
            like_prob += 0.25
            rt_prob += 0.25
        elif like_count >= 200 or retweet_count >= 50:
            like_prob += 0.15
            rt_prob += 0.12

        text = f"{snippet} {page_text[:600]}".lower()
        if "?" in text or any(w in text for w in ["how", "why", "what if", "tip", "trick", "idea", "thoughts"]):
            reply_prob += 0.22
        if topic and any(t in topic.lower() for t in ["ai", "startup", "code", "python", "product", "growth"]):
            reply_prob += 0.1

        decision = {
            "like": random.random() < min(0.98, max(0.1, like_prob)),
            "retweet": random.random() < min(0.7, max(0.02, rt_prob)),
            "reply": random.random() < min(0.6, max(0.05, reply_prob)),
        }
        # Quota-aware adjustments: prioritize filling deficits, and avoid overshooting
        if self.likes_total < self.max_likes:
            decision["like"] = True
        else:
            decision["like"] = False
        if self.replies_total < self.max_replies:
            # Encourage replies when under target
            decision["reply"] = True
        else:
            decision["reply"] = False
        if not any(decision.values()):
            decision["like"] = True
        return decision

    def _read_engagement_counts(self) -> Dict[str, int]:
        script = (
            "() => {\n"
            "  const article = document.querySelector('article[role=\\'article\\']');\n"
            "  if (!article) return JSON.stringify({});\n"
            "  function parseCount(el){\n"
            "    if (!el) return 0;\n"
            "    const t = el.innerText || el.textContent || '';\n"
            "    const m = t.match(/([0-9,.]+)([KkMm]?)/);\n"
            "    if (!m) return 0;\n"
            "    let n = parseFloat(m[1].replace(/,/g,''));\n"
            "    const suf = (m[2]||'').toLowerCase();\n"
            "    if (suf==='k') n *= 1000;\n"
            "    if (suf==='m') n *= 1000000;\n"
            "    return Math.round(n);\n"
            "  }\n"
            "  const reply = parseCount(article.querySelector('[data-testid=\\'reply\\']'));\n"
            "  const retweet = parseCount(article.querySelector('[data-testid=\\'retweet\\']'));\n"
            "  const like = parseCount(article.querySelector('[data-testid=\\'like\\']'));\n"
            "  return JSON.stringify({reply, retweet, like});\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            try:
                return {k: int(v) for k, v in json.loads(res["repr"]).items()}
            except Exception:
                return {}
        return {}

    # --- Exclusion helpers ---
    def _topic_is_excluded(self, topic: str) -> bool:
        try:
            t = (topic or "").lower()
            if not t:
                return False
            return any(ek in t for ek in self.excluded_keywords)
        except Exception:
            return False

    def _text_contains_excluded(self, text: str) -> bool:
        try:
            if not text:
                return False
            low = text.lower()
            return any(ek in low for ek in self.excluded_keywords)
        except Exception:
            return False

    def _like_current_tweet(self) -> bool:
        if self.dry_run:
            return True
        # Treat already-liked as success to avoid timeouts
        ok = self._click_article_action("like")
        if not ok:
            # Fallback: try generic like button
            ok = self._click_any_visible("[data-testid='like']", already_selector="[data-testid='unlike']")
        return ok

    def _retweet_current_tweet(self) -> bool:
        if self.dry_run:
            return True
        # Treat already-retweeted as success
        clicked = self._click_article_action("retweet") or self._click_any_visible("[data-testid='retweet']", already_selector="[data-testid='unretweet']")
        if not clicked:
            return False
        time.sleep(0.5)
        confirmed = (
            self._click("div[data-testid='retweetConfirm']")
            or self._click("div[role='menuitem'][data-testid='retweetConfirm']")
            or self._click_any_visible("div[data-testid='retweetConfirm'], div[role='menuitem'][data-testid='retweetConfirm']")
        )
        return confirmed

    def _generate_reply_text(self, topic: str, snippet: str, page_text: str = "") -> str:
        query = (
            f"TREND: {topic}\n"
            f"SNIPPET: {snippet[:400]}\n"
            f"PAGE_CONTENT: {page_text[:4000]}\n"
            "You are viewing the live page above. Use its details to craft a helpful reply.\n"
        )
        try:
            result = self.reply_loop.run(query)
            reply = (
                result.get("results", {}).get("promoter")
                if isinstance(result, dict)
                else None
            )
            if isinstance(reply, str):
                return self._sanitize_text(reply.strip(), max_len=280)
        except Exception:
            pass
        # Fallback minimal reply
        return self._sanitize_text("Interesting take. One tweak that often helps: be concrete about steps/results—curious what you’d try first?", max_len=280)

    def _reply_current_tweet(self, reply_text: str) -> Tuple[bool, str]:
        if self.dry_run:
            return True, ""
        if not self._click_article_action("reply"):
            return False, ""
        time.sleep(0.8)
        ok = False
        # Prefer explicit DraftJS contenteditable target shown in DOM
        for sel in [
            "div[role='dialog'] .public-DraftEditor-content[contenteditable='true'][data-testid='tweetTextarea_0']",
            "div[role='dialog'] .public-DraftEditor-content[contenteditable='true']",
            "div[role='dialog'] [data-testid='tweetTextarea_0'][contenteditable='true']",
            "div[role='dialog'] [contenteditable='true']",
            "[data-testid='tweetTextarea_0'][contenteditable='true']",
        ]:
            # Try native fill; if it fails, fall back to JS typing
            if self._fill(sel, reply_text) or self._js_type_into(sel, reply_text):
                ok = True
                break
        if not ok:
            return False, ""

        # Submit by focusing textbox and sending Ctrl+Enter (keyboard-only)
        textbox_selectors = [
            "div[role='dialog'] div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='dialog'] div[role='textbox']",
            "div[role='dialog'] [contenteditable='true']",
            "div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='textbox']",
            "[contenteditable='true']",
        ]
        sent = False
        for sel in textbox_selectors:
            try:
                self._press_key("Control+Enter", selector=sel)
                time.sleep(0.6)
                sent = True
                break
            except Exception:
                continue
        if not sent:
            for sel in textbox_selectors:
                try:
                    self._press_key("Meta+Enter", selector=sel)
                    time.sleep(0.6)
                    sent = True
                    break
                except Exception:
                    continue
        permalink_try = self._find_latest_status_permalink()
        if permalink_try:
            return True, permalink_try
        time.sleep(1.0)
        permalink_try = self._find_latest_status_permalink()
        if permalink_try:
            return True, permalink_try
        return False, ""

    def _post_original(self) -> Tuple[bool, str]:
        if self.dry_run:
            return True, ""
        # Open composer
        try:
            self._navigate("https://x.com/compose/post", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
        except Exception:
            return False, ""
        time.sleep(0.6)
        content = self._generate_post_text()
        # Remember last content and append to today's list for repetition checks
        self._last_post_content = content
        try:
            self.posts_today.append(content)
        except Exception:
            pass
        ok = False
        for sel in [
            "div[role='dialog'] div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='dialog'] div[role='textbox']",
            "div[role='dialog'] [contenteditable='true']",
            "div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='textbox']",
            "div[aria-label='Tweet text']",
            "[contenteditable='true']",
            "div.public-DraftEditor-content[contenteditable='true']",
            "div[data-contents='true']",
            "div.public-DraftStyleDefault-block",
        ]:
            if self._fill(sel, content) or self._js_type_into(sel, content):
                ok = True
                break
        if not ok:
            return False, ""
        # Submit by focusing textbox and sending Ctrl+Enter (keyboard-only)
        textbox_selectors = [
            "div[role='dialog'] div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='dialog'] div[role='textbox']",
            "div[role='dialog'] [contenteditable='true']",
            "div[role='textbox'][data-testid='tweetTextarea_0']",
            "div[role='textbox']",
            "[contenteditable='true']",
        ]
        sent = False
        for sel in textbox_selectors:
            try:
                self._press_key("Control+Enter", selector=sel)
                time.sleep(0.6)
                sent = True
                break
            except Exception:
                continue
        if not sent:
            for sel in textbox_selectors:
                try:
                    self._press_key("Meta+Enter", selector=sel)
                    time.sleep(0.6)
                    sent = True
                    break
                except Exception:
                    continue
        permalink_try = self._find_latest_status_permalink()
        if permalink_try:
            return True, permalink_try
        time.sleep(1.0)
        permalink_try = self._find_latest_status_permalink()
        if permalink_try:
            return True, permalink_try
        return False, ""

    def _generate_post_text(self) -> str:
        # Build context from recent feed/search page texts and topics
        recent_context = "\n\n".join([(t or "").strip()[:1200] for t in self.page_text_contexts[-3:]])
        topics_line = ", ".join(self.last_topics[:8])
        todays_posts = "\n- ".join([(p or "").strip()[:220] for p in self.posts_today[-10:]])
        try:
            result = self.post_loop.run(
                "You have been browsing X. Use the recent feed/search context and topics to inspire an original post.\n"
                f"RECENT_TOPICS: {topics_line}\n"
                f"PAGE_CONTEXTS:\n{recent_context}\n"
                f"TODAYS_POSTS_ALREADY_PUBLISHED:\n- {todays_posts}\n"
                "Do not repeat the same exact topics or angles as TODAYS_POSTS_ALREADY_PUBLISHED. Vary topics and framing.\n"
                "Write the post only (<= 220 chars).\n"
                + ("NEVER write about these excluded topics: " + ", ".join(sorted(self.excluded_keywords)) if self.excluded_keywords else "NEVER write about excluded topics.")
            )
            post = (
                result.get("results", {}).get("promoter")
                if isinstance(result, dict)
                else None
            )
            if isinstance(post, str):
                post_sanitized = self._sanitize_text(post.strip(), max_len=220)
                if self._text_contains_excluded(post_sanitized):
                    raise ValueError("generated_post_contains_excluded_topic")
                return post_sanitized
        except Exception:
            pass
        return self._sanitize_text("A tiny trick: show, don’t tell. Pick one concrete example and make it vivid.", max_len=220)

    def _find_latest_status_permalink(self) -> str:
        script = (
            "() => {\n"
            "  const anchors = Array.from(document.querySelectorAll(\"article[role='article'] time a[href*='/status/']\"));\n"
            "  if (anchors.length) return anchors[0].href;\n"
            "  const any = Array.from(document.querySelectorAll(\"a[href*='/status/']\"));\n"
            "  return any.length ? any[0].href : '';\n"
            "}"
        )
        res = _json_loads_safe(pw_evaluate(script))
        if isinstance(res, str):
            return res
        if isinstance(res, dict) and isinstance(res.get("repr"), str):
            return res.get("repr", "")
        return ""

    def _sanitize_text(self, text: str, max_len: int) -> str:
        """Remove hashtags and spammy CTA/self-promo phrases; keep URLs; trim length."""
        try:
            import re as _re
            t = (text or "")
            # Strip hashtags entirely
            t = _re.sub(r"(^|\s)#[\w_]+", " ", t)
            # Remove common CTA/self-promo phrases
            banned = [
                "follow for more", "follow me", "like and share", "retweet", "rt this",
                "smash that", "subscribe", "check out my", "bio link", "buy now", "limited time",
                "giveaway", "tag a friend", "turn on notifications", "👇", "🔥🔥", "free ebook",
            ]
            low = t.lower()
            for phrase in banned:
                if phrase in low:
                    pattern = _re.compile(_re.escape(phrase), _re.IGNORECASE)
                    t = pattern.sub("", t)
            # Collapse whitespace and trim
            t = _re.sub(r"\s+", " ", t).strip()
            if max_len > 0 and len(t) > max_len:
                t = t[:max_len].rstrip()
            return t
        except Exception:
            return (text or "")[:max_len]

    # --- CSV logging ---
    def _clean_snippet(self, text: str) -> str:
        """Strip common X UI header boilerplate and own handle from snippets."""
        try:
            import re as _re
            t = (text or "")
            low = t.lower().strip()

            # Remove known UI header if it appears as a prefix
            header = "home explore notifications messages grok communities premium verified orgs profile more"
            if low.startswith(header):
                t = t[len(text) - len(text.lstrip()):]  # preserve original leading spaces
                t = t[len(header):].lstrip()

            # Remove own handle in multiple forms anywhere (case-insensitive)
            user = (self.x_user or "").lstrip("@")
            if user:
                # @user, user as a standalone token, and with trailing punctuation like : , - ·
                patterns = [
                    rf"(?i)(^|[\s\W])@{_re.escape(user)}(?=$|[\s\W])",
                    rf"(?i)(^|[\s\W]){_re.escape(user)}(?=$|[\s\W])",
                ]
                for p in patterns:
                    t = _re.sub(p, " ", t)

            # Remove brand tokens like 'botlab' anywhere (case-insensitive)
            for brand in ("botlab",):
                p = _re.compile(rf"(?i)(^|[\s\W]){_re.escape(brand)}(?=$|[\s\W])")
                t = p.sub(" ", t)

            # Collapse excessive spaces and punctuation leftovers
            t = _re.sub(r"\s+", " ", t)
            t = _re.sub(r"\s+([,;:\-·])", r" \1", t)
            t = t.strip(" \t\r\n-·,;:")
            return t
        except Exception:
            return (text or "")
    def _log_action(
        self,
        action_type: str,
        keyword_or_source: str,
        author_handle: str,
        tweet_title_or_snippet: str,
        tweet_url: str,
        reason_if_skipped: str,
        content_prefix: str,
        engagement_flags: str,
        permalink: str,
    ) -> None:
        exists = os.path.exists(self.log_csv_path)
        with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [
                "timestamp",
                "action_type",
                "keyword_or_source",
                "author_handle",
                "tweet_title_or_snippet",
                "tweet_url",
                "reason_if_skipped",
                "content_prefix",
                "engagement_flags",
                "permalink",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            from datetime import datetime
            # Sanitize text fields to single-line to avoid breaking CSV rows
            def one_line(s: str) -> str:
                return (s or "").replace("\r", " ").replace("\n", " ").strip()
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "action_type": one_line(action_type),
                "keyword_or_source": one_line(keyword_or_source),
                "author_handle": one_line(author_handle),
                "tweet_title_or_snippet": one_line(self._clean_snippet(tweet_title_or_snippet)),
                "tweet_url": one_line(tweet_url),
                "reason_if_skipped": one_line(reason_if_skipped),
                "content_prefix": one_line(content_prefix),
                "engagement_flags": one_line(engagement_flags),
                "permalink": one_line(permalink),
            })

    def _log_error(self, message: str) -> None:
        """Append an error row into the same CSV log for unified visibility."""
        try:
            exists = os.path.exists(self.log_csv_path)
            with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "timestamp",
                    "action_type",
                    "keyword_or_source",
                    "author_handle",
                    "tweet_title_or_snippet",
                    "tweet_url",
                    "reason_if_skipped",
                    "content_prefix",
                    "engagement_flags",
                    "permalink",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                from datetime import datetime
                def one_line(s: str) -> str:
                    return (s or "").replace("\r", " ").replace("\n", " ").strip()
                writer.writerow({
                    "timestamp": datetime.utcnow().isoformat(),
                    "action_type": "error",
                    "keyword_or_source": "",
                    "author_handle": "",
                    "tweet_title_or_snippet": "",
                    "tweet_url": "",
                    "reason_if_skipped": one_line((message or ""))[:240],
                    "content_prefix": "",
                    "engagement_flags": "",
                    "permalink": "",
                })
        except Exception:
            pass

    def _log_debug(self, message: str) -> None:
        if not self.verbose:
            return
        try:
            exists = os.path.exists(self.log_csv_path)
            with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "timestamp",
                    "action_type",
                    "keyword_or_source",
                    "author_handle",
                    "tweet_title_or_snippet",
                    "tweet_url",
                    "reason_if_skipped",
                    "content_prefix",
                    "engagement_flags",
                    "permalink",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                from datetime import datetime
                def one_line(s: str) -> str:
                    return (s or "").replace("\r", " ").replace("\n", " ").strip()
                writer.writerow({
                    "timestamp": datetime.utcnow().isoformat(),
                    "action_type": "debug",
                    "keyword_or_source": "",
                    "author_handle": "",
                    "tweet_title_or_snippet": "",
                    "tweet_url": "",
                    "reason_if_skipped": one_line((message or ""))[:240],
                    "content_prefix": "",
                    "engagement_flags": "",
                    "permalink": "",
                })
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Programmatic X (Twitter) promoter loop")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (default: agents/promoter/data/x_promoter_config.json)")
    parser.add_argument("--x_user", type=str, default=None, help="X username (or set X_USER)")
    parser.add_argument("--x_pass", type=str, default=None, help="X password (or set X_PASS)")
    parser.add_argument("--max_actions_total", type=int, default=None, help="Maximum total actions this run")
    parser.add_argument("--max_original_posts", type=int, default=None, help="Deprecated: use --max_posts instead")
    parser.add_argument("--max_posts", type=int, default=None, help="Maximum number of posts to create this run")
    parser.add_argument("--max_likes", type=int, default=None, help="Target number of likes to perform this run")
    parser.add_argument("--max_replies", type=int, default=None, help="Target number of replies to perform this run")
    parser.add_argument("--keywords", type=str, nargs='*', default=None, help="Additional search keywords to diversify discovery")
    parser.add_argument("--excluded_keywords", type=str, nargs='*', default=None, help="Lowercased substrings; avoid topics/content containing any of these terms")
    parser.add_argument("--exclude_authors", type=str, nargs='*', default=None, help="Author handles to exclude from engagement (without @)")
    parser.add_argument("--log_csv", type=str, default=None, help="Path to CSV log file")
    parser.add_argument("--model_id", type=str, default=None, help="Model id for text generation")
    parser.add_argument("--post_wait_seconds", type=int, default=None, help="Fixed seconds to wait between actions")
    parser.add_argument("--page_settle_seconds", type=float, default=None, help="Seconds to wait after navigation")
    parser.add_argument("--action_settle_seconds", type=float, default=None, help="Seconds to wait after actions")
    parser.add_argument("--otp_fetch_cmd", type=str, default=None, help="Shell command to fetch latest X verification code from VPS")
    parser.add_argument("--otp_regex", type=str, default=None, help="Regex with one capture group to extract OTP from command output")
    parser.add_argument("--otp_code", type=str, default=None, help="Provide OTP explicitly (overrides fetch cmd)")
    parser.add_argument("--use_tor", action="store_true", help="Use Tor via tor_manager (proxied Playwright)")
    parser.add_argument("--no-use_tor", dest="use_tor", action="store_false", help="Disable Tor explicitly")
    parser.set_defaults(use_tor=None)
    parser.add_argument("--headed", action="store_true", help="Run Playwright with visible browser (overrides default headless)")
    parser.add_argument("--dry_run", action="store_true", help="Do everything except actually posting/engaging")
    parser.add_argument("--keep_browser_open", action="store_true", help="Keep browser open after run (do not exit)")
    parser.add_argument("--browser_height", type=int, default=None, help="Viewport height for browser window")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging to CSV")
    parser.add_argument("--user_data_dir", type=str, default=None, help="Playwright persistent user data dir to reuse sessions")
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

    # Headless default with optional --headed override
    if bool(args.headed):
        os.environ["PROMOTER_PLAYWRIGHT_HEADLESS"] = "0"
    else:
        os.environ["PROMOTER_PLAYWRIGHT_HEADLESS"] = "1"

    # Viewport height
    browser_height_val = pick("browser_height", None)
    if browser_height_val is not None:
        try:
            os.environ["PROMOTER_PLAYWRIGHT_VIEWPORT_HEIGHT"] = str(int(browser_height_val))
        except Exception:
            pass

    # Persistent user data dir to keep sessions/cookies
    udd = pick("user_data_dir", None)
    if udd is not None:
        try:
            os.environ["PROMOTER_PLAYWRIGHT_USER_DATA_DIR"] = str(udd)
        except Exception:
            pass

    loop = XPromoterLoop(
        x_user=pick("x_user", ""),
        x_pass=pick("x_pass", ""),
        max_actions_total=int(pick("max_actions_total", 8)),
        max_original_posts=int(pick("max_original_posts", 3)),
        max_likes=int(pick("max_likes", 0)),
        max_replies=int(pick("max_replies", 0)),
        max_posts=pick("max_posts", None),
        keywords=pick("keywords", None),
        exclude_authors=pick("exclude_authors", None),
        excluded_keywords=pick("excluded_keywords", None),
        log_csv_path=pick("log_csv"),
        model_id=pick("model_id"),
        post_wait_seconds=pick("post_wait_seconds"),
        page_settle_seconds=float(pick("page_settle_seconds", 2.0)) if pick("page_settle_seconds", None) is not None else 2.0,
        action_settle_seconds=float(pick("action_settle_seconds", 0.5)) if pick("action_settle_seconds", None) is not None else 0.5,
        otp_fetch_cmd=pick("otp_fetch_cmd"),
        otp_regex=pick("otp_regex"),
        otp_code=pick("otp_code"),
        dry_run=bool(args.dry_run or cfg.get("dry_run", False)),
        verbose=bool(args.verbose or cfg.get("verbose", False)),
    )
    result = loop.run()
    print(json.dumps(result, indent=2))

    # Optionally keep the browser/process alive (useful in headed mode)
    keep_open = bool(args.keep_browser_open or cfg.get("keep_browser_open", False))
    if keep_open:
        # Signal tools wrapper to preserve the browser session on process exit
        os.environ["PROMOTER_KEEP_BROWSER_OPEN"] = "1"
        try:
            print("Browser kept open. Press Ctrl+C to exit.")
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    """Main function to parse arguments."""
    main(parse_args())



