from typing import Optional, List, Dict, Any
import re

from ..researcher.agents import ResearcherAgent
from ..writer_critic.agents import WriterAgent
from ..editor.agents import EditorAgent
from ..utils.file_manager.file_manager import FileManager
from ..utils.agents.simple_llm import SimpleLiteLLMModel
from .tools import web_search as web_search_gemini_only, resolve_url, upgrade_markdown_links
from smolagents.utils import AgentGenerationError


class NewsletterPipeline:
    """Pipeline that produces a weekly newsletter from a topic.

    Flow: ResearcherAgent → WriterAgent (with Critic) → EditorAgent (with FactChecker)
    Saves artifacts to `agents/newsletter/data/` using FileManager.
    """

    def __init__(
        self,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
        editor: Optional[EditorAgent] = None,
        llm_only: bool = True,
    ) -> None:
        # Shared simple model (preferred over rate-limited model)
        shared_model = SimpleLiteLLMModel(
            model_id="gemini/gemini-2.0-flash",
            model_info_path="agents/utils/gemini/gem_llm_info.json",
        )
        self.model = shared_model
        self.llm_only = llm_only

        # Allow dependency injection; otherwise create defaults with shared model
        # Researcher: disable default web search; provide Gemini-only override
        self.researcher = researcher or ResearcherAgent(
            model=shared_model,
            max_steps=12,
            disable_default_web_search=True,
            additional_tools=[web_search_gemini_only],
        )
        self.writer = writer or WriterAgent(model=shared_model, max_steps=6)
        self.editor = editor or EditorAgent(model=shared_model, max_steps=30)

        # File manager for saving outputs
        self.file_manager = FileManager()

    def run(
        self,
        topic: str,
        audience: Optional[str] = None,
        style: Optional[str] = None,
        tweet: bool = False,
    ) -> str:
        """Generate a newsletter.

        Args:
            topic: The focus/topic for this week's newsletter.
            audience: Optional target audience hint (e.g., "technical PMs").
            style: Optional style hint (e.g., "concise, punchy").

        Returns:
            Final edited newsletter markdown.
        """
        # If LLM-only mode, bypass agent tool runs entirely
        if self.llm_only:
            # Lightweight search to collect source URLs (Gemini-only tool)
            queries = [
                f"{topic} site:news site:blog",
                f"{topic} this week site:news",
                f"{topic} latest analysis site:blog",
                f"{topic} research paper site:arxiv.org",
                f"{topic} press release site:press",
            ]
            collected_urls: List[str] = []
            for q in queries:
                try:
                    result = web_search_gemini_only(query=q, max_results=8, rate_limit_seconds=2, max_retries=2, disable_duckduckgo=True)
                    raw_urls = self._extract_urls(result)
                    # Resolve to canonical/article URLs
                    for u in raw_urls:
                        resolved = resolve_url(u)
                        collected_urls.append(resolved)
                except Exception:
                    continue
            # Dedupe while preserving order
            seen = set()
            sources_list: List[str] = []
            for u in collected_urls:
                if u not in seen:
                    seen.add(u)
                    sources_list.append(u)

            research_markdown = self._llm_only_research(topic, audience)
            self.file_manager.save_file(
                content=research_markdown,
                agent_name="newsletter_agent",
                file_type="research",
                title=f"Research - {topic}",
                metadata={"topic": topic, "audience": audience or "", "stage": "research"},
            )

            writer_prompt_parts = [
                "Write a weekly newsletter using the research below.",
                "Structure: intro (2-3 sentences), 3-6 major highlights with context,",
                "then Quick Bites (bullets), and What's Next (bullets).",
                "Citations: Turn the most relevant phrase directly into a Markdown link like [phrase](URL).",
                "Do NOT append '(Source: URL)' or use the word 'source' as the link text. No bottom 'Sources' section.",
                "Do NOT invent URLs.",
                "Keep it cohesive and readable; avoid generic filler.",
                "Research material follows delimited by <research> tags.",
                "<research>",
                research_markdown,
                "</research>",
                "\nRequired Sources (use inline only; no bottom list):",
                "<sources>",
                "\n".join(sources_list) if sources_list else "",
                "</sources>",
            ]
            if style:
                writer_prompt_parts.insert(1, f"Style preference: {style}.")
            if audience:
                writer_prompt_parts.insert(1, f"Audience: {audience}.")

            writer_prompt = " \n".join(writer_prompt_parts)
            draft_markdown = self._llm_only_write(writer_prompt)
            self.file_manager.save_file(
                content=draft_markdown,
                agent_name="newsletter_agent",
                file_type="draft",
                title=f"Draft - {topic}",
                metadata={"topic": topic, "audience": audience or "", "stage": "draft"},
            )

            final_markdown = self._llm_only_edit(draft_markdown)
            # Enforce inline link style (convert trailing (Source: URL) to inline [phrase](URL))
            final_markdown = self._inline_urls_in_text(final_markdown)
            # Upgrade all embedded links to article URLs where possible
            final_markdown = upgrade_markdown_links(final_markdown)
            # Save to daily master for history
            self.file_manager.save_file(
                content=final_markdown,
                agent_name="newsletter_agent",
                file_type="newsletter",
                title=f"Newsletter - {topic}",
                metadata={"topic": topic, "audience": audience or "", "stage": "final"},
            )
            # Also save a standalone markdown file
            self.file_manager.save_file(
                content=final_markdown,
                agent_name="newsletter_agent",
                file_type="newsletter",
                title=f"Newsletter - {topic}",
                metadata=None,
                use_daily_master=False,
                extension=".md",
            )
            if tweet:
                tweet_text = self._summarize_as_tweet(final_markdown)
                # Save tweet to daily master
                self.file_manager.save_file(
                    content=tweet_text,
                    agent_name="newsletter_agent",
                    file_type="tweet",
                    title=f"Tweet - {topic}",
                    metadata={"stage": "tweet"},
                )
                # Save standalone .txt
                self.file_manager.save_file(
                    content=tweet_text,
                    agent_name="newsletter_agent",
                    file_type="tweet",
                    title=f"Tweet - {topic}",
                    metadata=None,
                    use_daily_master=False,
                    extension=".txt",
                )
            return final_markdown

        # 1) Research (agent mode)
        research_query_parts = [
            f"Weekly roundup on: {topic}",
            "Summarize the most important developments from roughly the last 7 days.",
            "Prefer authoritative sources; include clear source URLs.",
            "Group findings into themes with short summaries.",
        ]
        if audience:
            research_query_parts.append(f"Tune for the needs of: {audience}.")
        research_query = " \n".join(research_query_parts)

        try:
            research_markdown = self.researcher.run_query(research_query)
        except Exception:
            # Fallback: LLM-only research (no tools)
            research_markdown = self._llm_only_research(topic, audience)

        # Save research artifact
        self.file_manager.save_file(
            content=research_markdown,
            agent_name="newsletter_agent",
            file_type="research",
            title=f"Research - {topic}",
            metadata={"topic": topic, "audience": audience or "", "stage": "research"},
        )

        # 2) Writing draft (with critic feedback inside WriterAgent)
        writer_prompt_parts = [
            "Write a weekly newsletter using the research below.",
            "Structure: intro (2-3 sentences), 3-6 major highlights with context,",
            "then Quick Bites (bullets), and What's Next (bullets).",
            "Include relevant sources inline where natural.",
            "Keep it cohesive and readable; avoid generic filler.",
            "Research material follows delimited by <research> tags.",
            "<research>",
            research_markdown,
            "</research>",
        ]
        if style:
            writer_prompt_parts.insert(1, f"Style preference: {style}.")
        if audience:
            writer_prompt_parts.insert(1, f"Audience: {audience}.")

        writer_prompt = " \n".join(writer_prompt_parts)
        try:
            draft_markdown = self.writer.write_draft(writer_prompt)
        except Exception:
            # Fallback: LLM-only drafting
            draft_markdown = self._llm_only_write(writer_prompt)

        # Save draft artifact
        self.file_manager.save_file(
            content=draft_markdown,
            agent_name="newsletter_agent",
            file_type="draft",
            title=f"Draft - {topic}",
            metadata={"topic": topic, "audience": audience or "", "stage": "draft"},
        )

        # 3) Editing + fact checking
        try:
            final_markdown = self.editor.edit_content(draft_markdown)
        except Exception:
            # Fallback: LLM-only editing pass
            final_markdown = self._llm_only_edit(draft_markdown)

        # Save final newsletter
        self.file_manager.save_file(
            content=final_markdown,
            agent_name="newsletter_agent",
            file_type="newsletter",
            title=f"Newsletter - {topic}",
            metadata={"topic": topic, "audience": audience or "", "stage": "final"},
        )

        return final_markdown

    # -------- Internal fallbacks (LLM-only, no tools) --------
    def _generate(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Call the underlying SimpleLiteLLMModel and extract plain text."""
        response = self.model.generate(messages=messages, temperature=temperature)
        # Try extract text from various response shapes
        if hasattr(response, 'content'):
            content = response.content
            # smolagents ChatMessage may use list of parts
            if isinstance(content, list):
                # Join any text parts
                texts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        texts.append(str(part['text']))
                    elif isinstance(part, str):
                        texts.append(part)
                if texts:
                    return "\n".join(texts)
            elif isinstance(content, str):
                return content
        # Fallback to string cast
        return str(response)

    def _llm_only_research(self, topic: str, audience: Optional[str]) -> str:
        user = (
            f"Produce a weekly research roundup on: {topic}.\n"
            "Summarize key developments from the last 7 days.\n"
            "Structure as: Themes with short summaries; include any well-known sources if you recall them, but do not invent URLs.\n"
        )
        if audience:
            user += f"Audience: {audience}.\n"
        system = (
            "You are a precise research summarizer. If unsure of a source URL, omit it."
        )
        edited = self._generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.2)
        edited = self._inline_urls_in_text(edited)
        return upgrade_markdown_links(edited)

    def _summarize_as_tweet(self, newsletter_md: str) -> str:
        system = (
            "You write tight, high-signal tweets summarizing a newsletter in <= 280 chars. "
            "Use an engaging hook, 2-4 crisp points separated by • or |, and 1-2 inline links if space allows. "
            "No hashtags, no emojis, no line breaks, avoid fluff."
        )
        user = (
            "Summarize this newsletter as a single tweet <= 280 chars (no line breaks):\n\n" + newsletter_md
        )
        text = self._generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.4)
        # Collapse whitespace and enforce length
        tweet = " ".join(text.strip().split())
        return tweet[:280]

    def _llm_only_write(self, prompt: str) -> str:
        system = "You are a concise newsletter writer."
        return self._generate([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], temperature=0.5)

    def _llm_only_edit(self, draft: str) -> str:
        system = (
            "You are an expert editor. Improve clarity, structure, and cohesion. Keep length similar. "
            "Ensure all external factual claims include inline URLs anchored on the most relevant phrase. "
            "Use the actual article URL (not homepages or aggregators). Prefer canonical article pages, press releases, or paper pages. "
            "If a 'Sources' section exists, remove it after moving all citations inline. Do not invent URLs."
        )
        user = ("Edit the following newsletter. Return only the revised markdown.\n\n" + draft)
        return self._generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=0.2)

    def _extract_urls(self, text: str) -> List[str]:
        """Extract HTTP/HTTPS URLs from text."""
        urls = re.findall(r"https?://[^\s)]+", text)
        # Strip trailing punctuation commonly attached
        cleaned: List[str] = []
        for u in urls:
            cleaned.append(u.rstrip(',.);]"\''))
        return cleaned

    def _inline_urls_in_text(self, content: str, max_anchor_words: int = 5) -> str:
        """Convert trailing sources to inline links on preceding phrase.

        Handles both '(Source: URL)' and '[source](URL)' patterns by linking the
        last few words in the preceding clause: '... phrase [source](URL)' -> '[phrase](URL) ...'.
        """
        patterns = [
            re.compile(r"\(Source:\s*(https?://[^\s)]+)\)", re.IGNORECASE),
            re.compile(r"\[source\]\((https?://[^\s)]+)\)", re.IGNORECASE),
        ]

        def transform_once(text: str) -> tuple[str, bool]:
            for pat in patterns:
                m = pat.search(text)
                if not m:
                    continue
                url = m.group(1)
                prefix = text[:m.start()]
                suffix = text[m.end():]
                # Soft boundary: sentence or line break
                boundary_idx = max(prefix.rfind("\n"), prefix.rfind(". "), prefix.rfind("! "), prefix.rfind("? "))
                segment = prefix[boundary_idx + 1:] if boundary_idx != -1 else prefix
                # Extract words (avoid already linked markdown or urls)
                words = re.findall(r"[A-Za-z0-9][^\s]*", segment)
                if not words:
                    anchor = "link"
                else:
                    anchor_tokens = words[-max_anchor_words:]
                    anchor = " ".join(anchor_tokens).strip().rstrip(",;:()[]{}")
                    if len(anchor) > 80:
                        anchor = " ".join(anchor.split()[-3:])
                # Replace tail of prefix with inline link
                anchor_escaped = re.escape(anchor)
                new_prefix = re.sub(anchor_escaped + r"$", f"[{anchor}]({url})", prefix)
                if new_prefix == prefix:
                    new_prefix = prefix + f" [{anchor}]({url})"
                return new_prefix + suffix, True
            return text, False

        changed = True
        while changed:
            content, changed = transform_once(content)
        return content


