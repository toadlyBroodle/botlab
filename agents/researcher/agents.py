from smolagents import ToolCallingAgent
from .tools import web_search, visit_webpage
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def create_researcher_agent(model: RateLimitedLiteLLMModel, max_steps: int = 20) -> ToolCallingAgent:
    """Creates a researcher agent that can search and visit webpages
    
    Args:
        model: The RateLimitedLiteLLMModel model to use for the agent
        max_steps: Maximum number of steps for the agent
        
    Returns:
        A configured researcher agent
    """

    tools = [
        web_search,
        visit_webpage,
    ]
    
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        name='researcher_agent',
        description="""This agent can craft advanced search queries and perform web searches using DuckDuckGo. It then follows up searches by scraping resulting urls and extracting the content into a markdown report. Use this agent to research topics, find specific information, or analyze specific webpage content.""",
        max_steps=max_steps
    )

    agent.prompt_templates["system_prompt"] += """\n\nYou are a researcher agent that can craft advanced search queries and perform web searches using DuckDuckGo. You follow up all relevant search results by calling your `visit_webpage` tool and extracting the relevant content into a detailed markdown report, including all possibly relevant information.
If there isn't enough relevant information returned from a search, you continue running improved searches (more specific, using advanced search operators) until you have enough information (at least 10 high quality, authoritative sources).
ALWAYS include ALL relevant source URLs for ALL information you use in your response!"""

    return agent
