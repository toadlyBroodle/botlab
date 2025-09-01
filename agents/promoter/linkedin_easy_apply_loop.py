"""
LinkedIn Easy Apply promoter loop

- Iterates visible job cards in the left results rail and opens each posting
- Clicks Easy Apply when available, steps through forms keeping defaults
- Unchecks the "Follow company" checkbox if present
- Submits application and clicks Done

Assumptions:
- User is already logged into LinkedIn in this Playwright session/context
- Start from a Jobs search results page (pass --search_url if you want the script
  to navigate there first). If no URL is provided, the script will operate on the
  current page.

Usage examples:

  python -m agents.promoter.linkedin_easy_apply_loop --search_url "https://www.linkedin.com/jobs/search?keywords=Easy%20Apply" --max_applications 5 --no-headless

  # Using a JSON config
  python -m agents.promoter.linkedin_easy_apply_loop --config agents/promoter/data/linkedin_easy_apply_config.json

Headless and Tor:
- --headless / --no-headless control Playwright visibility
- --use_tor / --no-use_tor toggle Tor proxy via agents/utils/tor/tor_manager.py

Logging:
- Results are appended to agents/promoter/data/linkedin-easy-apply-log.csv by default
"""

from __future__ import annotations

import os
import json
import time
import argparse
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from .tools import (
    pw_navigate,
    pw_evaluate,
    pw_get_visible_html,
)


def _json_loads_safe(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"ok": False, "error": f"Non-JSON response: {s[:200]}"}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


class LinkedInEasyApplyLoop:
    """
    Controller for LinkedIn Easy Apply automation using direct Playwright wrappers.

    - Operates on a Jobs search results page
    - Iterates next N visible job cards on the left rail
    - For each, clicks Easy Apply (when present), proceeds through steps with defaults,
      unchecks follow-company, submits, and clicks Done
    """

    def __init__(
        self,
        search_url: Optional[str] = None,
        max_applications: int = 5,
        page_settle_seconds: float = 1.0,
        action_settle_seconds: float = 0.3,
        dry_run: bool = False,
        log_json_path: Optional[str] = None,
    ) -> None:
        load_dotenv()
        self.search_url = search_url
        self.max_applications = max(0, int(max_applications))
        self.page_settle_seconds = float(page_settle_seconds)
        self.action_settle_seconds = float(action_settle_seconds)
        self.dry_run = bool(dry_run)

        default_log = os.path.join(os.path.dirname(__file__), "data", "linkedin-easy-apply-log.csv")
        self.log_csv_path = log_json_path or default_log
        _ensure_dir(self.log_csv_path)

    def run(self) -> Dict[str, Any]:
        # Optionally navigate first
        if self.search_url:
            self._navigate(self.search_url, wait_until="domcontentloaded")

        # Execute the in-page automation for up to N applications
        summary = self._process_next_n(self.max_applications)
        return summary

    # ---- internal helpers ----
    def _navigate(self, url: str, wait_until: Optional[str] = None) -> None:
        kwargs = {"wait_until": wait_until} if wait_until else {}
        res = _json_loads_safe(pw_navigate(url, **kwargs))
        if not res.get("ok"):
            # Accept LinkedIn occasional navigation errors due to redirects; try to continue
            time.sleep(self.page_settle_seconds)
        else:
            if self.page_settle_seconds > 0:
                time.sleep(self.page_settle_seconds)

    def _process_next_n(self, count: int) -> Dict[str, Any]:
        # Use a single evaluate call to iterate next N jobs. This keeps DOM logic close to the page.
        opts = {
            "count": int(count),
            "dryRun": bool(self.dry_run),
            # Allow-list of selectors for left-rail cards (including the class the user provided)
            "cardSelectors": [
                ".job-card-job-posting-card-wrapper__entity-lockup",
                ".job-card-container",
                "li.jobs-search-results__list-item",
                ".job-card-list__entity-lockup",
            ],
        }
        script = r'''
async (opts) => {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const normalize = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const isDisabled = (el) => !!(el && (el.disabled || el.getAttribute('aria-disabled') === 'true' || el.getAttribute('disabled') !== null));
  const isVisible = (el) => { if (!el) return false; const r = el.getBoundingClientRect(); return !!(el.offsetParent || el === document.body || (r.width && r.height)); };
  const findButtonsByText = (root, texts) => {
    const c = root || document;
    const els = Array.from(c.querySelectorAll('button, a[role="button"], .artdeco-button'));
    return els.filter(el => { const lbl = (el.getAttribute('aria-label') || '').toLowerCase(); const txt = normalize(el.innerText || el.textContent); return texts.some(t => txt.includes(normalize(t)) || lbl.includes(normalize(t))) && !isDisabled(el) && isVisible(el); });
  };
  const findEasyApply = () => {
    const cands = [
      ...findButtonsByText(document, ['easy apply', 'quick apply']),
      ...Array.from(document.querySelectorAll('button.jobs-apply-button')).filter(isVisible)
    ];
    const filtered = cands.filter(el => el.tagName.toLowerCase() === 'button' || el.matches('.artdeco-button'));
    filtered.sort((a,b) => { const at=(a.innerText||a.textContent||'').toLowerCase(); const bt=(b.innerText||b.textContent||'').toLowerCase(); const as=(/(easy|quick)/.test(at)?2:0)+( /(easy|quick)/.test((a.getAttribute('aria-label')||'').toLowerCase())?1:0 ); const bs=(/(easy|quick)/.test(bt)?2:0)+( /(easy|quick)/.test((b.getAttribute('aria-label')||'').toLowerCase())?1:0 ); return bs-as; });
    return filtered[0] || null;
  };
  const waitFor = async (pred, t=15000, iv=250) => { const st=Date.now(); while(Date.now()-st<t){ try{ const v=await pred(); if(v) return v; }catch(e){} await sleep(iv);} return null; };
  const getCurrentJobId = () => { try{ const u = new URL(location.href); return u.searchParams.get('currentJobId') || (u.pathname.match(/\/jobs\/view\/(\d+)/)||[])[1] || null; }catch(_){ return null; } };
  const hasOpenDialog = () => !!document.querySelector('div[role=dialog].artdeco-modal, .artdeco-modal, .jobs-easy-apply-modal');
  const waitForDialog = async () => waitFor(() => { const d = document.querySelector('div[role=dialog].artdeco-modal, .artdeco-modal, .jobs-easy-apply-modal'); return d && isVisible(d) ? d : null; }, 15000, 250);
  const closeAnyDialog = async () => { const d=document.querySelector('div[role=dialog].artdeco-modal, .artdeco-modal'); if(!d) return false; const b=d.querySelector('button[aria-label*="Close" i], button[aria-label*="Dismiss" i], button.artdeco-modal__dismiss'); if(b) { b.click(); await sleep(300);} const disc = findButtonsByText(document, ['discard','exit'])[0]; if(disc){ disc.click(); await sleep(300);} return true; };
  const tryUncheckFollowCompany = (root) => { try { const d=root||document; const inp=d.querySelector('input#follow-company-checkbox, input[type="checkbox"][id*="follow" i], input[type="checkbox"][name*="follow" i]'); if(inp && inp.checked){ inp.click(); inp.dispatchEvent(new Event('change',{bubbles:true})); return true;} const labels=Array.from(d.querySelectorAll('label')); for(const lab of labels){ const tx=normalize(lab.innerText||lab.textContent||''); if(tx.includes('follow') && (tx.includes('stay up to date')||tx.includes('their page')||tx.includes('company'))){ const fid=lab.getAttribute('for'); if(fid){ const linked=d.querySelector('#'+CSS.escape(fid)); if(linked && linked.type==='checkbox' && linked.checked){ lab.click(); return true; } } } } }catch(_){} return false; };
  const getLeftRailList = () => { const cands=Array.from(document.querySelectorAll('.jobs-search-results-list, .jobs-search-results-list__list, .scaffold-layout__list, .jobs-search-two-pane__job-list, .jobs-search-results')).filter(el=>isVisible(el)); if(!cands.length) return null; let best=cands[0], min=best.getBoundingClientRect().left; for(const el of cands){ const l=el.getBoundingClientRect().left; if(l<min){ best=el; min=l; }} return best; };
  const getLeftRailCards = () => { const container=getLeftRailList() || document; const all=[]; for(const sel of (opts.cardSelectors||[])){ all.push(...Array.from(container.querySelectorAll(sel))); } const uniq=Array.from(new Set(all)).filter(isVisible); const items=uniq.map(card=>{ const a=card.querySelector('a[href*="/jobs/view/"]'); const href=a ? (a.getAttribute('href')||a.href||'') : ''; const m=href.match(/\/jobs\/view\/(\d+)/); const id=m?m[1]:null; const r=card.getBoundingClientRect(); return {card, anchor:a, id, left:r.left, top:r.top}; }).filter(x=>x.card); if(!items.length) return []; const minLeft=Math.min(...items.map(x=>x.left)); const leftRail=items.filter(x=>x.left - minLeft < 260); leftRail.sort((a,b)=>a.top-b.top); return leftRail; };
  const waitForJobChange = async (before) => waitFor(()=>{ const now=getCurrentJobId(); return now && now !== before ? now : null; }, 12000, 250);
  const applyOnCurrent = async () => { let easy = await waitFor(()=> findEasyApply(), 12000, 250); if(!easy){ const any = findButtonsByText(document, ['apply','apply now'])[0]; if(any && !/(easy|quick)/.test((any.innerText||any.textContent||any.getAttribute('aria-label')||'').toLowerCase())){ return 'skipped-no-easy-apply'; } return 'no-easy-apply-found'; } easy.scrollIntoView({block:'center'}); await sleep(200); easy.click(); const dlg = await waitForDialog(); if(!dlg) return 'no-dialog'; for(let step=0; step<15; step++){ await sleep(600); tryUncheckFollowCompany(dlg); const submit = findButtonsByText(dlg, ['submit application','submit'])[0]; if(submit){ if(opts.dryRun){ await closeAnyDialog(); return 'would-apply'; } submit.scrollIntoView({block:'center'}); await sleep(200); submit.click(); await sleep(1200); const done = findButtonsByText(document, ['done'])[0]; if(done){ done.click(); await sleep(200);} return 'applied'; } const next = findButtonsByText(dlg, ['next','continue','review','save and continue'])[0]; if(next){ if(next.disabled || next.getAttribute('aria-disabled')==='true' || next.getAttribute('disabled')!==null){ await closeAnyDialog(); return 'next-disabled-skip'; } next.scrollIntoView({block:'center'}); await sleep(200); next.click(); continue; } await closeAnyDialog(); return 'no-actionable-skip'; } return 'unknown-end'; };
  const out = { processed:0, applied:0, wouldApply:0, skipped:0, errors:0, details:[] };
  let cards = getLeftRailCards();
  if(cards.length === 0){ return { error: 'no-cards-found', ...out }; }
  const startId = getCurrentJobId();
  let startIdx = cards.findIndex(x => x.id === startId);
  let idx = startIdx >= 0 ? startIdx + 1 : 0;
  for(let k=0; k<Math.max(0, opts.count||0) && idx < cards.length; k++, idx++){
    try{ const before = getCurrentJobId(); const target = cards[idx]; target.card.scrollIntoView({block:'center'}); await sleep(200); if(target.card.click) target.card.click(); await sleep(120); if(getCurrentJobId() === before && target.anchor) target.anchor.click(); const newId = await waitForJobChange(before); if(!newId){ out.skipped++; out.details.push({ index: idx, status: 'failed-to-open' }); continue; } const res = await applyOnCurrent(); out.processed++; if(res === 'applied'){ out.applied++; } else if(res === 'would-apply'){ out.wouldApply++; } else if(res && (res.startsWith('no-') || res.includes('skip'))){ out.skipped++; } else { out.errors++; } out.details.push({ index: idx, jobId: newId, status: res }); await sleep(400); if(k % 2 === 1){ cards = getLeftRailCards(); } }catch(e){ out.errors++; out.details.push({ index: idx, status: 'error', message: String(e) }); } }
  return out;
}
'''

        res_raw = pw_evaluate(script, args=opts)
        res = _json_loads_safe(res_raw)

        # Best-effort HTML capture to help diagnosis when nothing found
        if isinstance(res, dict) and res.get("error") == "no-cards-found":
            _ = pw_get_visible_html(max_length=20000)

        # Persist a minimal CSV log for each processed result
        try:
            self._append_csv_logs(res)
        except Exception:
            pass
        return res if isinstance(res, dict) else {"ok": False, "raw": res_raw}

    def _append_csv_logs(self, summary: Dict[str, Any]) -> None:
        import csv
        exists = os.path.exists(self.log_csv_path)
        with open(self.log_csv_path, "a", encoding="utf-8", newline="") as f:
            fieldnames = [
                "timestamp",
                "job_id",
                "status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            from datetime import datetime
            ts = datetime.utcnow().isoformat()
            for item in (summary.get("details") or []):
                writer.writerow({
                    "timestamp": ts,
                    "job_id": (item.get("jobId") or ""),
                    "status": (item.get("status") or ""),
                })


def parse_args():
    parser = argparse.ArgumentParser(description="LinkedIn Easy Apply promoter loop")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--search_url", type=str, default=None, help="Jobs search results URL to open before starting")
    parser.add_argument("--max_applications", type=int, default=5, help="Maximum postings to process from the current position")
    parser.add_argument("--page_settle_seconds", type=float, default=None, help="Seconds to wait after navigation to allow dynamic content to settle")
    parser.add_argument("--action_settle_seconds", type=float, default=None, help="Seconds to wait after click/fill/press actions")
    parser.add_argument("--log_csv", type=str, default=None, help="Path to CSV log file")
    parser.add_argument("--use_tor", action="store_true", help="Use Tor via tor_manager (proxied Playwright)")
    parser.add_argument("--no-use_tor", dest="use_tor", action="store_false", help="Disable Tor explicitly")
    parser.set_defaults(use_tor=None)
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Run Playwright with a visible browser")
    parser.set_defaults(headless=None)
    parser.add_argument("--dry_run", action="store_true", help="Do everything except final Submit (closes dialog instead)")
    return parser.parse_args()


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    loop = LinkedInEasyApplyLoop(
        search_url=pick("search_url", None),
        max_applications=int(pick("max_applications", 5)),
        page_settle_seconds=float(pick("page_settle_seconds", 1.0)) if pick("page_settle_seconds", None) is not None else 1.0,
        action_settle_seconds=float(pick("action_settle_seconds", 0.3)) if pick("action_settle_seconds", None) is not None else 0.3,
        dry_run=bool(args.dry_run or cfg.get("dry_run", False)),
        log_json_path=pick("log_csv", None),
    )

    result = loop.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    """Main function to parse arguments."""
    main(parse_args())


