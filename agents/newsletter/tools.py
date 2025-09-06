"""Newsletter helpers and tool overrides.

Includes:
- Gemini-only web_search to avoid DuckDuckGo `ddgs` dependency
- URL resolution helpers to dereference redirects and find canonical URLs
"""

from smolagents import tool
from ..utils.agents.tools import _perform_gemini_search
import re
import requests
import urllib.parse


@tool
def web_search(query: str, max_results: int = 10, rate_limit_seconds: float = 5.0, max_retries: int = 3, disable_duckduckgo: bool = True) -> str:
    """Gemini-only web search (DuckDuckGo disabled).

    Args:
        query: Search query
        max_results: Maximum number of results
        rate_limit_seconds: Ignored here; kept for signature compatibility
        max_retries: Ignored here; kept for signature compatibility
        disable_duckduckgo: Unused; always True behavior

    Returns:
        Markdown-formatted search results using Gemini with grounding when available.
    """
    # Always use Gemini search to avoid ddgs dependency
    raw = _perform_gemini_search(query, max_results)
    # Post-process: resolve any URLs to canonical/article URLs
    urls = re.findall(r"https?://[^\s)]+", raw)
    for u in urls:
        resolved = resolve_url(u)
        if resolved != u:
            raw = raw.replace(u, resolved)
    return raw


def _extract_canonical_from_html(html: str) -> str | None:
    """Try to extract canonical URL from HTML using simple regex heuristics."""
    # rel=canonical
    m = re.search(r'<link[^>]*rel=["\']?canonical["\']?[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        return m.group(1)
    # OpenGraph og:url
    m = re.search(r'<meta[^>]*property=["\']og:url["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def resolve_url(url: str, timeout: float = 5.0) -> str:
    """Resolve redirects and return a canonical, article-level URL when possible.

    - Follows HTTP redirects (GET with allow_redirects)
    - Parses canonical or og:url from HTML when present
    - Returns original URL on failures
    """
    try:
        # Follow redirects
        resp = requests.get(url, timeout=timeout, allow_redirects=True, headers={
            "User-Agent": "botlab-newsletter/1.0"
        })
        final_url = resp.url or url
        # Try to extract canonical from final page
        if resp.ok and 'text/html' in resp.headers.get('Content-Type', '') and resp.text:
            canonical = _extract_canonical_from_html(resp.text)
            if canonical and canonical.startswith('http'):
                final_url = canonical
        return final_url
    except Exception:
        return url


def _duckduckgo_html_search(query: str, timeout: float = 6.0) -> list[str]:
    """Very lightweight HTML search against DuckDuckGo HTML endpoint. No ddgs dep.
    Returns a list of result URLs (best-first) or empty list.
    """
    try:
        params = {"q": query}
        headers = {"User-Agent": "botlab-newsletter/1.0"}
        resp = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=timeout)
        if not resp.ok:
            return []
        html = resp.text
        # Extract result links
        # Common pattern: <a class="result__a" href="/l/?kh=-1&uddg=<ENCODED>">
        urls = []
        for m in re.finditer(r'<a[^>]*class=["\']result__a["\'][^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = m.group(1)
            # Unwrap /l/?uddg=
            parsed = urllib.parse.urlparse(href)
            if parsed.path.startswith('/l/'):
                qs = urllib.parse.parse_qs(parsed.query)
                uddg = qs.get('uddg', [None])[0]
                if uddg:
                    href = urllib.parse.unquote(uddg)
            if href.startswith('http'):
                urls.append(href)
        return urls
    except Exception:
        return []


def upgrade_article_url(url: str, anchor_text: str | None = None, timeout: float = 6.0) -> str:
    """Upgrade a generic/top-level or redirect URL to a likely article URL.

    Strategy:
    1) Resolve redirects/canonical
    2) If still non-article (root/homepage or vertex redirect), run site-limited search
       with anchor_text to pick best article link.
    """
    u = resolve_url(url, timeout=timeout)
    try:
        pu = urllib.parse.urlparse(u)
        is_vertex = 'vertexaisearch.cloud.google.com' in pu.netloc
        is_rootish = (pu.path in ('', '/') or len(pu.path.strip('/')) <= 1)
        if is_vertex or is_rootish:
            domain = pu.netloc.replace('www.', '')
            q = f"site:{domain} {anchor_text}" if anchor_text else f"site:{domain}"
            results = _duckduckgo_html_search(q, timeout=timeout)
            # Pick first result that is on the same domain and has a deeper path
            for candidate in results:
                pc = urllib.parse.urlparse(candidate)
                if domain in pc.netloc and pc.path and pc.path not in ('/', '') and len(pc.path.strip('/')) > 1:
                    return candidate
        return u
    except Exception:
        return u


def upgrade_markdown_links(content: str) -> str:
    """Upgrade all markdown links in the content to article URLs using anchor text.
    Only modifies links that appear to be non-article (homepages/vertex redirects).
    """
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    def _repl(m: re.Match) -> str:
        anchor = m.group(1)
        url = m.group(2)
        better = upgrade_article_url(url, anchor_text=anchor)
        return f"[{anchor}]({better})"
    return pattern.sub(_repl, content)


