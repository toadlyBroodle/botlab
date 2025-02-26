import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel
import sys
from utils.telemetry import start_telemetry

from writer_critic.agents import create_writer_agent, create_critic_agent
from writer_critic.tools import DRAFT_DIR, BASE_DIR

def setup_environment():
    """Set up environment variables and API keys"""

    # Ensure drafts directory exists
    os.makedirs(DRAFT_DIR, exist_ok=True)

    # Load .env from root directory
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    return api_key

def initialize(max_steps: int = 5, model_name: str = "gpt-4o-mini", enable_telemetry: bool = False):
    """Initialize the writer-critic system
    
    Args:
        max_steps: Maximum number of steps for the writer agent
        model_name: LLM model to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        
    Returns:
        A function that can process writing tasks
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    setup_environment()
    
    model = LiteLLMModel(
        model_id=model_name
    )
    
    # Create agents in the right order - critic first, then writer that manages critic
    critic_agent = create_critic_agent(model)
    writer_agent = create_writer_agent(model, critic_agent)
    
    # Set max steps
    writer_agent.max_steps = max_steps
    
    def run_writing_task(prompt: str) -> str:
        """Run a writing task through the writer-critic system
        
        Args:
            prompt: The writing prompt to process
            
        Returns:
            The final draft after iterations
        """
        # Run the writer agent with the prompt
        result = writer_agent.run(prompt)
        return result
        
    return run_writing_task

def main():
    """Main entry point when run directly"""
    run_writing_task = initialize(enable_telemetry=True)
    return run_writing_task

if __name__ == "__main__":
    main() 