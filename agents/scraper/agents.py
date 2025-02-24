from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    LiteLLMModel
)

def create_web_agent(model: LiteLLMModel) -> ToolCallingAgent:
    """Creates a web search agent that can search and visit webpages
    
    Args:
        model: The LiteLLM model to use for the agent
        
    Returns:
        A configured web search agent
    """

    tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
    ]
    
    agent = ToolCallingAgent(
        tools=tools,
        model=model
    )
    setattr(agent, 'name', 'web_search_agent')
    setattr(agent, 'description', 'This is an agent that can run web searches and scrape webpages to gather information.')
    setattr(agent, 'max_steps', 8)
    return agent

def create_manager_agent(model: LiteLLMModel, web_search_agent: ToolCallingAgent) -> CodeAgent:
    """Creates a manager agent that coordinates the web agent
    
    Args:
        model: The LiteLLM model to use for the agent
        web_agent: The web search agent to manage
        
    Returns:
        A configured manager agent
    """
    agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_search_agent],
        additional_authorized_imports=["time"],
    )
    setattr(agent, 'name', 'manager_agent')
    setattr(agent, 'description', 'This is an agent that can manage the web search agent.')
    setattr(agent, 'max_steps', 4)
    return agent
