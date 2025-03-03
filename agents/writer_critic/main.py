import os
from dotenv import load_dotenv
import sys
from typing import Optional
from utils.telemetry import start_telemetry

from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from writer_critic.agents import create_writer_agent, create_critic_agent
from writer_critic.tools import DRAFT_DIR, BASE_DIR

def setup_environment():
    """Set up environment variables and API keys"""

    # Ensure drafts directory exists
    os.makedirs(DRAFT_DIR, exist_ok=True)

    # Load .env from root directory
    load_dotenv()
    
    # Check for API key
    #api_key = os.getenv("OPENAI_API_KEY")
    #if not api_key:
    #    raise ValueError("OPENAI_API_KEY environment variable is not set")

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def initialize(
    max_steps: int = 5, 
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    model_id: str = "gemini/gemini-2.0-flash", 
    enable_telemetry: bool = False,
    writer_description: Optional[str] = None,
    critic_description: Optional[str] = None,
    writer_prompt: Optional[str] = None,
    critic_prompt: Optional[str] = None
):
    """Initialize the writer-critic system
    
    Args:
        max_steps: Maximum number of steps for the writer agent
        model_id: LLM model to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        writer_description: Optional additional description for the writer agent
        critic_description: Optional additional description for the critic agent
        writer_prompt: Optional custom system prompt for the writer agent
        critic_prompt: Optional custom system prompt for the critic agent
        
    Returns:
        A function that can process writing tasks
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    setup_environment()
    
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        model_info_path=model_info_path
    )
    
    # Create agents in the right order - critic first, then writer that manages critic
    critic_agent = create_critic_agent(
        model=model,
        agent_description=critic_description,
        system_prompt=critic_prompt
    )
    
    writer_agent = create_writer_agent(
        model=model,
        critic_agent=critic_agent, 
        max_steps=max_steps,
        agent_description=writer_description,
        system_prompt=writer_prompt
    )
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the writer-critic system with a prompt.")
    parser.add_argument("--prompt", type=str, default="Write a short story about a robot who discovers emotions.", 
                        help="The writing prompt to process")
    parser.add_argument("--enable-telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of steps")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--writer-description", type=str, default=None, help="Custom description for the writer agent")
    parser.add_argument("--critic-description", type=str, default=None, help="Custom description for the critic agent")
    parser.add_argument("--writer-prompt", type=str, default=None, help="Custom system prompt for the writer agent")
    parser.add_argument("--critic-prompt", type=str, default=None, help="Custom system prompt for the critic agent")
    
    args = parser.parse_args()
    
    # Initialize the writer-critic system with parameters from command line
    run_writing_task = initialize(
        max_steps=args.max_steps,
        model_id=args.model_id,
        enable_telemetry=args.enable_telemetry,
        writer_description=args.writer_description,
        critic_description=args.critic_description,
        writer_prompt=args.writer_prompt,
        critic_prompt=args.critic_prompt
    )
    
    # Run the writing task with the prompt
    result = run_writing_task(args.prompt)
    
    # Print the result
    print("\nFinal Draft:")
    print(result)
    
    return result

if __name__ == "__main__":
    main() 