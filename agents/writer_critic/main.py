import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel
import sys

# Add the current directory to the path so we can import the agents and tools modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the modules directly
from agents import create_writer_agent, create_critic_agent
from tools import DRAFT_DIR, BASE_DIR

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

def initialize(max_steps: int = 5, model_name: str = "gpt-4o-mini"):
    """Initialize the writer-critic system
    
    Args:
        max_steps: Maximum number of steps for the writer agent
        model_name: LLM model to use
        
    Returns:
        A function that can process writing tasks
    """
    
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
    run_writing_task = initialize()
    return run_writing_task

if __name__ == "__main__":
    main() 