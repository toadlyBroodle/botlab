from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool
)
from .tools import visit_webpage

def create_web_agent(model):
    """Creates and returns a web search agent with DuckDuckGo search and webpage visit capabilities"""
    return ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), visit_webpage],
        model=model,
        max_steps=10,
        name="search",
        description="Runs web searches for you. Give it your query as an argument."
    )

def create_manager_agent(model, web_agent):
    """Creates and returns a manager agent that coordinates the web agent"""
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"]
    ) 
