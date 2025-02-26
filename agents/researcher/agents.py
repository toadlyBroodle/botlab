from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel
)
from .tools import web_search, visit_webpage
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def create_web_agent(model: LiteLLMModel, max_steps: int = 20) -> ToolCallingAgent:
    """Creates a web search agent that can search and visit webpages
    
    Args:
        model: The LiteLLM model to use for the agent
        
    Returns:
        A configured web search agent
    """
    # Ensure the model is rate-limited
    if not isinstance(model, RateLimitedLiteLLMModel):
        print("Warning: Model is not rate-limited. Wrapping with RateLimitedLiteLLMModel...")
        if isinstance(model, LiteLLMModel):
            model_id = model.model_id
            model = RateLimitedLiteLLMModel(model_id=model_id)
        else:
            raise ValueError("Model must be a LiteLLMModel instance")

    tools = [
        web_search,
        visit_webpage,
    ]
    
    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        name='web_search_agent',
        description="""This agent can craft advanced search queries and perform web searches using DuckDuckGo. It then follows up searches by scraping resulting urls and extracting the content into a markdown report. Use this agent to research topics, find specific information, or analyze specific webpage content.""",
        max_steps=max_steps
    )

    agent.prompt_templates["system_prompt"] += """\n\nYou are a web search agent that can craft advanced search queries and perform web searches using DuckDuckGo. You follow up all relevant search results by calling your `visit_webpage` tool and extracting the relevant content into a detailed markdown report, including all possibly relevant information.
If there isn't enough relevant information returned from a search, you continue running improved searches (more specific, using advanced search operators) until you have enough information (at least 10 high quality, authoritative sources).
ALWAYS include ALL relevant source URLs for ALL information you use in your response!"""

    return agent

def create_researcher_agent(model: LiteLLMModel, web_search_agent: ToolCallingAgent, max_steps: int = 8) -> CodeAgent:
    """Creates a researcher agent that coordinates the web agent
    
    Args:
        model: The LiteLLM model to use for the agent
        web_search_agent: The web search agent that the researcher agent manages to perform the actual web searches
        
    Returns:
        A configured researcher agent
    """
    # Ensure the model is rate-limited
    if not isinstance(model, RateLimitedLiteLLMModel):
        print("Warning: Model is not rate-limited. Wrapping with RateLimitedLiteLLMModel...")
        if isinstance(model, LiteLLMModel):
            model_id = model.model_id
            model = RateLimitedLiteLLMModel(model_id=model_id)
        else:
            raise ValueError("Model must be a LiteLLMModel instance")
            
    agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_search_agent],
        additional_authorized_imports=["time", "json", "re"],
        name='researcher_agent',
        description="""This is a research agent that can gather and analyze information from the web. It can be called with very detailed, high-level research objectives and will handle the details of gathering information from the web and returning comprehensive reports or concise summaries and direct answers.""",
        max_steps=max_steps
    )

    agent.prompt_templates["system_prompt"] += """\n\nYou are a research agent that can gather and analyze information from the web. You manage a web_search_agent that performs all actual web searches and scraping of urls. You call it like so: `web_search_agent("your search query")`. You are responsible for making sure the web_search_agent returns highly relevent search results by suggesting specific highlevel, advanced search techniques, while leaving the actual query crafting and syntax to the web_search_agent. Every time it returns search results, you very carefully review the results and make sure they are highly relevant to the research objective. 
If the results do not provide a BROAD range (>10) of exactly relevant authoritative sources, or there is ANY missing information you require to answer your research objectives, you suggest alternative strategies, keywords, or search techniques and call the web_search_agent again as many times as needed. 
You persistently keep calling the web_search_agent, until you are certain you have enough information (at least 10 high quality, authoritative sources) to positively answer your research objectives. 
Only when you are sure the results are highly relevant to your initial research objectives, and ALL questions have been thoroughly answered, do you return the final results.
    
Always include all relevant source urls in your response, in clean markdown format, unless specifically instructed otherwise."""
    
    return agent

def ensure_rate_limited_model(model: LiteLLMModel) -> RateLimitedLiteLLMModel:
    """Ensures that a model is wrapped with rate limiting.
    
    Args:
        model: The model to ensure is rate-limited
        
    Returns:
        A rate-limited model
    """
    if isinstance(model, RateLimitedLiteLLMModel):
        return model
    
    if isinstance(model, LiteLLMModel):
        print(f"Wrapping model {model.model_id} with rate limiting...")
        return RateLimitedLiteLLMModel(model_id=model.model_id)
    
    raise ValueError("Model must be a LiteLLMModel instance")
