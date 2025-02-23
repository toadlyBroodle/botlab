import os
from smolagents import LiteLLMModel
from .agents import create_web_agent, create_manager_agent

def setup_agents(gemini_api_key):
    """Sets up the multi-agent system with Gemini authentication"""
    # Set Gemini API key
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    
    # Initialize the model using LiteLLM with Gemini
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        temperature=0.7,
        max_tokens=30_000
    )
    
    # Create agents
    web_agent = create_web_agent(model)
    manager_agent = create_manager_agent(model, web_agent)
    
    return manager_agent

def run_query(manager_agent, query):
    """Runs a query through the multi-agent system"""
    return manager_agent.run(query) 