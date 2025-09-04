from typing import Optional, List, Union
from ..utils.agents.base_agent import BaseToolCallingAgent
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from ..utils.gemini.simple_llm import SimpleLiteLLMModel


class AnimatorAgent(BaseToolCallingAgent):
    """Tool-calling agent that generates short videos with Veo 3 via Gemini API.

    Uses the project's LiteLLM wrappers (Simple or RateLimited) for consistency in cost tracking.
    Exposes tools to create videos from text prompts and optionally seed images.
    """

    def __init__(
        self,
        model: Optional[Union[RateLimitedLiteLLMModel, SimpleLiteLLMModel]] = None,
        max_steps: int = 10,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        additional_tools: Optional[List] = None,
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
            agent_name="animator_agent",
            enable_daily_quota_fallback=False,
            use_rate_limiting=use_rate_limiting,
            **kwargs,
        )

    def get_tools(self) -> List:
        from .tools import generate_video_with_veo3

        return [generate_video_with_veo3]

    def get_base_description(self) -> str:
        return (
            "Animator agent that generates 8-second 720p videos with audio using Veo 3 "
            "via Gemini API. Provide a descriptive prompt to produce cinematic outputs."
        )

    def get_default_system_prompt(self) -> str:
        return (
            "You create concise, production-ready video prompts for Veo 3. "
            "Favor clear scene descriptions, camera movement, mood, lighting, and pacing. "
            "Return short prompts and call the tool to render."
        )

    def get_agent_type_name(self) -> str:
        return "animator"


