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
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        log_csv_path: Optional[str] = None,
        model_id: Optional[str] = None,
        post_wait_seconds: Optional[int] = None,
        page_settle_seconds: float = 2.0,
        action_settle_seconds: float = 0.5,
        otp_fetch_cmd: Optional[str] = None,
        otp_regex: Optional[str] = None,
        otp_code: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        load_dotenv()
        self.x_user = x_user or os.getenv("X_USER", "")
        self.x_pass = x_pass or os.getenv("X_PASS", "")

        default_log = os.path.join(os.path.dirname(__file__), "data", "x-engage-agent-log.csv")
        self.log_csv_path = log_csv_path or default_log
        _ensure_dir(self.log_csv_path)

        self.max_actions_total = max(0, int(max_actions_total))
        self.actions_total = 0
        self.dry_run = dry_run

        env_model = os.getenv("PROMOTER_MODEL", "gemini/gemini-2.0-flash")
        self.model_id = model_id or env_model

        self.post_wait_seconds = int(post_wait_seconds) if post_wait_seconds is not None else None
        self.page_settle_seconds = float(page_settle_seconds)
        self.action_settle_seconds = float(action_settle_seconds)
        self.otp_fetch_cmd = (otp_fetch_cmd or os.getenv("X_OTP_FETCH_CMD", "")).strip()
        self.otp_regex = (otp_regex or os.getenv("X_OTP_REGEX", r"\\b(\\d{6})\\b")).strip()
        self.otp_code = (otp_code or os.getenv("X_OTP_CODE", "")).strip()

        # Build simple content generation loops
        self.reply_loop = self._build_reply_loop()
        self.post_loop = self._build_post_loop()

        # Rolling context from recent feed/search pages to inform generation
        self.page_text_contexts: List[str] = []
        self.last_topics: List[str] = []

    # --- Public entrypoint ---
    def run(self) -> Dict[str, Any]:
        if not self.x_user or not self.x_pass:
            raise ValueError("Missing x_user/x_pass")

        self._login()

        seen_tweets: set[str] = set()
        posted_original = False
        passes_without_actions = 0

        while not self._quotas_reached():
            topics = self._collect_trending_topics()
            self.last_topics = list(topics)
            if not topics:
                # Fallback to Home if no topics found
                topics = ["feed"]
                self.last_topics = ["feed"]

            actions_before_pass = self.actions_total

            for topic in topics:
                if self._quotas_reached():
                    break
                if topic == "feed":
                    self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
                    # Best-effort infinite scroll to collect a few items
                    self._infinite_scroll(iterations=4)
                    self._capture_page_context()
                    tweet_links = self._extract_status_links_from_page()
                else:
                    search_url = self._search_live_url(topic)
                    try:
                        self._navigate(search_url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
                    except Exception:
                        continue
                    self._capture_page_context()
                    tweet_links = self._extract_status_links_from_page()

                for url in tweet_links:
                    if self._quotas_reached():
                        break
                    if url in seen_tweets:
                        continue
                    seen_tweets.add(url)

                    try:
                        self._navigate(url, wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
                    except Exception:
                        self._log_action(
                            action_type="",
                            keyword_or_source=topic,
                            author_handle=self._handle_from_status_url(url),
                            tweet_title_or_snippet="",
                            tweet_url=url,
                            reason_if_skipped="navigate_failed",
                            content_prefix="",
                            engagement_flags="",
                            permalink="",
                        )
                        continue

                    snippet = self._extract_tweet_snippet()
                    page_text = self._get_visible_text(limit=8000)
                    decision = self._agent_decide_engagement(topic, page_text, snippet)
                    # Ensure at least one reply attempt early in the run
                    if self.actions_total == 0 and not bool(decision.get("reply")):
                        decision["reply"] = True

                    # Like if agent advises
                    like_ok = False
                    if decision.get("like") is True:
                        like_ok = self._like_current_tweet()
                    if like_ok:
                        self.actions_total += 1
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

                    # Retweet if agent advises
                    if not self._quotas_reached() and decision.get("retweet") is True:
                        rt_ok = self._retweet_current_tweet()
                        if rt_ok:
                            self.actions_total += 1
                            self._log_action(
                                action_type="retweet",
                                keyword_or_source=topic,
                                author_handle=self._handle_from_status_url(url),
                                tweet_title_or_snippet=snippet,
                                tweet_url=url,
                                reason_if_skipped="",
                                content_prefix="",
                                engagement_flags="retweet=1",
                                permalink=url,
                            )
                            self._human_sleep()

                    # Reply if agent advises; keep it concise and valuable
                    if not self._quotas_reached() and decision.get("reply") is True:
                        reply_text = str(decision.get("reply_text") or "").strip() or self._generate_reply_text(topic, snippet, page_text)
                        posted, permalink = self._reply_current_tweet(reply_text)
                        if posted:
                            self.actions_total += 1
                            self._log_action(
                                action_type="reply",
                                keyword_or_source=topic,
                                author_handle=self._handle_from_status_url(url),
                                tweet_title_or_snippet=snippet,
                                tweet_url=url,
                                reason_if_skipped="",
                                content_prefix=reply_text[:60],
                                engagement_flags="",
                                permalink=permalink or url,
                            )
                            self._human_sleep()

                    # Occasionally post an original
                    if not posted_original and not self._quotas_reached() and random.random() < 0.25:
                        posted, permalink = self._post_original()
                        if posted:
                            posted_original = True
                            self.actions_total += 1
                            self._log_action(
                                action_type="post",
                                keyword_or_source=topic,
                                author_handle="",
                                tweet_title_or_snippet="",
                                tweet_url="",
                                reason_if_skipped="",
                                content_prefix="",
                                engagement_flags="",
                                permalink=permalink,
                            )
                            self._human_sleep()

            # If no original was posted yet, make one best-effort attempt per pass
            if (not posted_original) and (not self._quotas_reached()):
                posted, permalink = self._post_original()
                if posted:
                    posted_original = True
                    self.actions_total += 1
                    self._log_action(
                        action_type="post",
                        keyword_or_source=",".join(self.last_topics[:3]),
                        author_handle="",
                        tweet_title_or_snippet="",
                        tweet_url="",
                        reason_if_skipped="",
                        content_prefix="",
                        engagement_flags="",
                        permalink=permalink,
                    )
                    self._human_sleep()

            if self.actions_total == actions_before_pass:
                passes_without_actions += 1
                if passes_without_actions >= 2:
                    break
            else:
                passes_without_actions = 0

        return {
            "status": "completed" if self.actions_total > 0 else "no_actions",
            "actions_total": int(self.actions_total),
            "log_csv": self.log_csv_path,
        }

    # --- Internal helpers ---

    def _build_reply_loop(self) -> AgentLoop:
        description = (
            "You are an autonomous online explorer running inside a real browser session. "
            "You may scroll the user's X feed and search for topics that you are personally curious about. "
            "Prioritize engaging with content that is most interesting to you. "
            "I am giving you the opportunity to explore and interact with the internet. "
            "Write a concise, specific X reply (40–140 words). No hashtags, no links. "
            "Lead with a micro-insight, joke, or contrarian/completing angle; add 1–2 concrete steps or a sharp observation. "
            "End with a subtle open loop or question. Output only the reply text."
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
        description = (
            "You are an autonomous online explorer with freedom to browse the feed and search for what interests you. "
            "I am giving you the opportunity to explore and interact with the internet. "
            "Create a viral-optimized original X post (<= 220 chars) inspired by what you found. "
            "Strong hook, 1 concrete insight or tiny example, soft CTA to follow for more. "
            "No hashtags. Output only the post text."
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
        return self.actions_total >= self.max_actions_total

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
        """Type text into a contenteditable DraftJS editor via JS selection + events."""
        script = (
            "(sel, val) => {\n"
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
        # Pass selector and text as separate args (Playwright will marshal arguments correctly)
        res = _json_loads_safe(pw_evaluate(script, args=[selector, text]))
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

    def _human_sleep(self) -> None:
        if self.dry_run:
            return
        if self.post_wait_seconds is not None:
            time.sleep(max(0, int(self.post_wait_seconds)))
        else:
            time.sleep(random.randint(20, 60))

    # --- X specific flows ---

    def _login(self) -> None:
        self._navigate("https://x.com/login", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)

        # Username step
        filled = False
        for sel in [
            "input[name='text']",
        ]:
            if self._fill(sel, self.x_user):
                filled = True
                break
        if not filled:
            raise RuntimeError("Failed to fill username on X login")

        # Prefer pressing Enter to advance
        self._press_key("Enter", selector="input[name='text']")
        time.sleep(1)

        # Password step
        filled = False
        for sel in [
            "input[name='password']",
        ]:
            if self._fill(sel, self.x_pass):
                filled = True
                break
        if not filled:
            raise RuntimeError("Failed to fill password on X login")

        # Prefer pressing Enter to submit
        self._press_key("Enter", selector="input[name='password']")
        time.sleep(2)

        # Handle possible verification challenge
        self._maybe_handle_login_challenges()

        # Verify by attempting to load home
        try:
            self._navigate("https://x.com/home", wait_until="domcontentloaded", tolerate_http_errors=True, retries=1)
        except Exception:
            pass

    def _open_explore(self) -> None:
        # Click Explore tab; try primary then fallback
        if not self._click("a[data-testid='AppTabBar_Explore_Link']"):
            self._click("a[aria-label='Explore']")

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
            # Try Trending tab
            self._click("a[href^='/explore/tabs/trending']")
            time.sleep(1)
            topics.extend(self._extract_topics_from_explore())
        except Exception:
            pass
        try:
            # Try News tab for additional items
            self._click("a[href^='/explore/tabs/news']")
            time.sleep(1)
            topics.extend(self._extract_topics_from_explore())
        except Exception:
            pass

        # Deduplicate, keep order
        seen: set[str] = set()
        uniq: List[str] = []
        for t in topics:
            t_clean = t.strip()
            if t_clean and t_clean.lower() not in seen:
                seen.add(t_clean.lower())
                uniq.append(t_clean)
        return uniq[:12]

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
            "  const anchors = Array.from(document.querySelectorAll(\"a[href*='/status/']\"));\n"
            "  const urls = anchors.map(a => a.href).filter(Boolean);\n"
            "  const out = [];\n"
            "  for (const h of urls){\n"
            "    try {\n"
            "      const u = new URL(h, location.href);\n"
            "      if (u.hostname !== 'x.com') continue;\n"
            "      const parts = u.pathname.split('/').filter(Boolean);\n"
            "      const idx = parts.indexOf('status');\n"
            "      if (idx === -1 || idx + 1 >= parts.length) continue;\n"
            "      const handle = parts[idx - 1] || '';\n"
            "      const id = parts[idx + 1];\n"
            "      if (!/^\\d+$/.test(id)) continue;\n"
            "      const base = `https://x.com/${handle}/status/${id}`;\n"
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

    def _extract_tweet_snippet(self) -> str:
        text = pw_get_visible_text(limit=8000) or ""
        snippet = (text or "").strip().split("\n")
        joined = " ".join([s.strip() for s in snippet if s.strip()])
        return joined[:200]

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

        like_prob = 0.35
        reply_prob = 0.18
        rt_prob = 0.08

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
            "like": random.random() < min(0.95, max(0.05, like_prob)),
            "retweet": random.random() < min(0.6, max(0.0, rt_prob)),
            "reply": random.random() < min(0.5, max(0.0, reply_prob)),
        }
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

    def _like_current_tweet(self) -> bool:
        if self.dry_run:
            return True
        # Treat already-liked as success to avoid timeouts
        return self._click_article_action("like")

    def _retweet_current_tweet(self) -> bool:
        if self.dry_run:
            return True
        # Treat already-retweeted as success
        clicked = self._click_article_action("retweet")
        if not clicked:
            return False
        time.sleep(0.5)
        confirmed = (
            self._click("div[data-testid='retweetConfirm']")
            or self._click("div[role='menuitem'][data-testid='retweetConfirm']")
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
                return reply.strip()
        except Exception:
            pass
        # Fallback minimal reply
        return "Interesting take. One tweak that often helps: be concrete about steps/results—curious what you’d try first?"

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
            if self._fill(sel, content):
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
        try:
            result = self.post_loop.run(
                "You have been browsing X. Use the recent feed/search context and topics to inspire an original post.\n"
                f"RECENT_TOPICS: {topics_line}\n"
                f"PAGE_CONTEXTS:\n{recent_context}\n"
                "Write the post only (<= 220 chars). No hashtags."
            )
            post = (
                result.get("results", {}).get("promoter")
                if isinstance(result, dict)
                else None
            )
            if isinstance(post, str):
                return post.strip()[:220]
        except Exception:
            pass
        return "A tiny trick: show, don’t tell. Pick one concrete example and make it vivid. Follow for more."[:220]

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

    # --- CSV logging ---
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
                "tweet_title_or_snippet": one_line(tweet_title_or_snippet),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Programmatic X (Twitter) promoter loop")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--x_user", type=str, default=None, help="X username (or set X_USER)")
    parser.add_argument("--x_pass", type=str, default=None, help="X password (or set X_PASS)")
    parser.add_argument("--max_actions_total", type=int, default=None, help="Maximum total actions this run")
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
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run with visible browser")
    parser.set_defaults(headless=None)
    parser.add_argument("--dry_run", action="store_true", help="Do everything except actually posting/engaging")
    parser.add_argument("--keep_browser_open", action="store_true", help="Keep browser open after run (do not exit)")
    parser.add_argument("--browser_height", type=int, default=None, help="Viewport height for browser window")
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

    # Viewport height
    browser_height_val = pick("browser_height", None)
    if browser_height_val is not None:
        try:
            os.environ["PROMOTER_PLAYWRIGHT_VIEWPORT_HEIGHT"] = str(int(browser_height_val))
        except Exception:
            pass

    loop = XPromoterLoop(
        x_user=pick("x_user", ""),
        x_pass=pick("x_pass", ""),
        max_actions_total=int(pick("max_actions_total", 8)),
        log_csv_path=pick("log_csv"),
        model_id=pick("model_id"),
        post_wait_seconds=pick("post_wait_seconds"),
        page_settle_seconds=float(pick("page_settle_seconds", 2.0)) if pick("page_settle_seconds", None) is not None else 2.0,
        action_settle_seconds=float(pick("action_settle_seconds", 0.5)) if pick("action_settle_seconds", None) is not None else 0.5,
        otp_fetch_cmd=pick("otp_fetch_cmd"),
        otp_regex=pick("otp_regex"),
        otp_code=pick("otp_code"),
        dry_run=bool(args.dry_run or cfg.get("dry_run", False)),
    )
    result = loop.run()
    print(json.dumps(result, indent=2))

    # Optionally keep the browser/process alive (useful in headed mode)
    keep_open = bool(args.keep_browser_open or cfg.get("keep_browser_open", False))
    if keep_open:
        try:
            print("Browser kept open. Press Ctrl+C to exit.")
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    """Main function to parse arguments."""
    main(parse_args())



