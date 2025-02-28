import os
from typing import List, Dict, Optional, Callable, Any
from dotenv import load_dotenv
from utils.telemetry import start_telemetry
from manager.agents import create_manager_agent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def setup_environment():
    """Set up environment variables and API keys"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return api_key

def initialize(
    enable_telemetry: bool = False,
    managed_agents: Optional[List] = None,
    agent_descriptions: Optional[Dict[str, str]] = None,
    agent_system_prompts: Optional[Dict[str, str]] = None,
    max_steps: int = 8, 
    base_wait_time: float = 2.0,
    max_retries: int = 3,
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    model_id: str = "gemini/gemini-2.0-flash"
) -> Callable[[str], str]:
    """Initialize the manager agent system with optional telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        managed_agents: List of pre-configured agents to manage
        agent_descriptions: Optional dictionary mapping agent names to description additions
        agent_system_prompts: Optional dictionary mapping agent names to system prompt additions
        max_steps: Maximum number of steps for the agents
        base_wait_time: Base wait time in seconds for rate limiting
        max_retries: Maximum number of retry attempts for rate limiting
        model_info_path: Path to the model info JSON file
        model_id: The model ID to use (default: gemini/gemini-2.0-flash)
        
    Returns:
        A function that can process queries through the manager agent
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    api_key = setup_environment()
    
    # Create a rate-limited model with model-specific rate limits
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        model_info_path=model_info_path
    )
    
    # Initialize the list of agents to manage
    agents_to_manage = managed_agents or []
    descriptions = agent_descriptions or {}
    
    # Create the manager agent that coordinates the managed agents
    manager_agent = create_manager_agent(
        model=model, 
        managed_agents=agents_to_manage,
        agent_descriptions=descriptions,
        max_steps=max_steps
    )
    
    def run_query(query: str) -> str:
        """Runs a query through the manager agent
        
        Args:
            query: The query to process
            
        Returns:
            The response from the manager agent
        """
        result = manager_agent.run(query)
        return result
        
    return run_query

def main():
    """Main entry point when run directly"""
    # When run directly, use a minimal configuration
    # For more complex setups, use the example.py file
    run_query = initialize(
        enable_telemetry=True, 
        max_steps=8, 
        base_wait_time=3.0, 
        max_retries=5,
        model_info_path="utils/gemini/gem_llm_info.json",
        model_id="gemini/gemini-2.0-flash"
    )
    return run_query

if __name__ == "__main__":
    main()