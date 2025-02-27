import os
from dotenv import load_dotenv
from utils.telemetry import start_telemetry
from researcher.agents import create_researcher_agent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def setup_environment():
    """Set up environment variables and API keys"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return api_key

def initialize(enable_telemetry: bool = False, max_steps: int = 20, 
               base_wait_time: float = 2.0, max_retries: int = 3,
               model_info_path: str = "utils/gemini/gem_llm_info.json",
               model_id: str = "gemini/gemini-2.0-flash"):
    """Initialize the system with optional telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        max_steps: Maximum number of steps for the agent
        base_wait_time: Base wait time in seconds for rate limiting
        max_retries: Maximum number of retry attempts for rate limiting
        model_info_path: Path to the model info JSON file
        model_id: The model ID to use (default: gemini/gemini-2.0-flash)
        
    Returns:
        A function that can process research queries
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
    
    # Create researcher agent
    researcher_agent = create_researcher_agent(model, max_steps=max_steps)
    
    def run_research(query: str) -> str:
        """Runs a research query through the researcher agent
        
        Args:
            query: The research query to process
            
        Returns:
            The response from the researcher agent
        """
        result = researcher_agent.run(query)
        return result
        
    return run_research

def main():
    """Main entry point when run directly"""
    run_research = initialize(
        enable_telemetry=True, 
        max_steps=20, 
        base_wait_time=3.0, 
        max_retries=5,
        model_info_path="utils/gemini/gem_llm_info.json",
        model_id="gemini/gemini-2.0-flash"
    )
    return run_research

if __name__ == "__main__":
    main()