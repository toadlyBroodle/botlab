import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel
from utils.telemetry import start_telemetry
from scraper.agents import create_web_agent, create_manager_agent

def setup_environment():
    """Set up environment variables and API keys"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return api_key

def initialize(enable_telemetry: bool = False):
    """Initialize the system with optional telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        
    Returns:
        A function that can process queries
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    api_key = setup_environment()
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash"
    )
    
    # Create agents
    web_agent = create_web_agent(model)
    manager_agent = create_manager_agent(model, web_agent)
    
    def run_query(query: str) -> str:
        """Runs a query through the multi-agent system
        
        Args:
            query: The search query to process
            
        Returns:
            The response from the manager agent
        """
        result = manager_agent.run(query)
        return result
        
    return run_query

def main():
    """Main entry point when run directly"""
    run_query = initialize(enable_telemetry=True)
    return run_query

if __name__ == "__main__":
    main()