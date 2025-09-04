from typing import Optional, List, Union
from smolagents import CodeAgent
from ..utils.agents.base_agent import BaseToolCallingAgent
from ..utils.agents.rate_lim_llm import RateLimitedLiteLLMModel
from ..utils.agents.simple_llm import SimpleLiteLLMModel
from ..utils.agents.tools import visit_webpage, apply_custom_agent_prompts, save_final_answer


class PromoterAgent(BaseToolCallingAgent):
    """A general-purpose promoter agent that can browse, log in, search for relevant posts, and post tailored replies.

    This agent is designed to work with a Playwright MCP server made available to the runtime
    as HTTP endpoints or SDK tools. It will call tools to navigate, authenticate, search,
    and submit replies according to its prompt instructions.
    """

    def __init__(
        self,
        model: Optional[Union[RateLimitedLiteLLMModel, SimpleLiteLLMModel]] = None,
        max_steps: int = 30,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        additional_tools: Optional[List] = None,
        enable_daily_quota_fallback: bool = False,
        use_rate_limiting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            max_steps=max_steps,
            agent_description=description,
            system_prompt=system_prompt,
            model_id=model_id,
            model_info_path=model_info_path,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            additional_tools=additional_tools,
            agent_name="promoter_agent",
            enable_daily_quota_fallback=enable_daily_quota_fallback,
            use_rate_limiting=use_rate_limiting,
            **kwargs,
        )

    def get_tools(self) -> List:
        # Return only core generic tools (exclude web_search and visit_webpage). Any extra tools are injected via BaseToolCallingAgent using additional_tools.
        return []

    def get_base_description(self) -> str:
        return (
            "Agent that promotes a target across sites: it navigates, logs in, searches for relevant "
            "threads, and posts short tailored replies following platform rules."
        )

    def get_default_system_prompt(self) -> str:
        return (
            "You are a promotion agent. You will: \n"
            "- Connect to the already running Playwright MCP server and ensure browsing tools are available.\n"
            "- Navigate to the target domain, handle cookie banners, and log in using provided creds.\n"
            "- Run targeted searches for recent posts matching the campaign intent.\n"
            "- Open promising threads, craft concise, non-spammy replies tailored to OP context, and submit.\n"
            "- Prefer posts within the last month. Avoid low-signal threads or off-topic contexts.\n"
            "- Keep replies brief, actionable, and disclose affiliation if required by subreddit rules.\n"
        )

    def get_agent_type_name(self) -> str:
        return "promoter"

    # Allow AgentLoop to call the agent instance directly
    def __call__(self, prompt: str) -> str:
        """Execute the promoter agent with a high-level instruction.

        Args:
            prompt: High-level instruction describing the navigation, login, search, and reply workflow.

        Returns:
            The final result string produced by the agent.
        """
        return self.run(prompt)


class RedditCsvAgentPromoter(PromoterAgent):
    """Specialized promoter configured to log into Reddit and promote csvagent.com for CSV enrichment."""

    def get_default_system_prompt(self) -> str:
        return (
            super().get_default_system_prompt()
            + "\nSpecific instructions for Reddit + CSV enrichment:\n"
            + "- Use the MCP tools to navigate and fill forms; MCP server is already running (do not attempt to launch it).\n"
            + "- Log in to Reddit with provided username/password.\n"
            + "- Search sitewide and in r/AITools, r/data, r/dataanalyst, r/dataengineering, r/excel for posts in the last month about CSV cleaning, enrichment, missing fields, dedupe, merging, or pre‑ingest validation.\n"
            + "- Post brief tailored replies recommending csvagent.com, focusing on: ‘If you want something simple for CSV data enrichment/cleaning (filling missing fields, validating/flagging bad rows, dedupe), try https://csvagent.com’.\n"
            + "- Tailor to the thread (e.g., e‑commerce CSVs → categories/brand/GTIN; pipelines → pre‑ingest validation split good/bad rows).\n"
            + "- Keep comments non‑spammy, under 2 sentences. Respect subreddit self‑promo rules.\n"
        )


