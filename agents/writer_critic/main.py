import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel
import sys
from typing import Optional
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

def initialize(
    max_steps: int = 5, 
    model_name: str = "gpt-4o-mini", 
    enable_telemetry: bool = False,
    writer_description: Optional[str] = None,
    critic_description: Optional[str] = None,
    writer_system_prompt: Optional[str] = None,
    critic_system_prompt: Optional[str] = None
):
    """Initialize the writer-critic system
    
    Args:
        max_steps: Maximum number of steps for the writer agent
        model_name: LLM model to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        writer_description: Optional additional description for the writer agent
        critic_description: Optional additional description for the critic agent
        writer_system_prompt: Optional custom system prompt for the writer agent
        critic_system_prompt: Optional custom system prompt for the critic agent
        
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
    critic_agent = create_critic_agent(
        model, 
        agent_description=critic_description,
        system_prompt=critic_system_prompt
    )
    
    writer_agent = create_writer_agent(
        model, 
        critic_agent=critic_agent, 
        max_steps=max_steps,
        agent_description=writer_description,
        system_prompt=writer_system_prompt
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
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--writer-description", type=str, default=None, help="Custom description for the writer agent")
    parser.add_argument("--critic-description", type=str, default=None, help="Custom description for the critic agent")
    parser.add_argument("--writer-prompt", type=str, default=None, help="Custom system prompt for the writer agent")
    parser.add_argument("--critic-prompt", type=str, default=None, help="Custom system prompt for the critic agent")
    
    args = parser.parse_args()
    
    # Initialize the writer-critic system with parameters from command line
    run_writing_task = initialize(
        max_steps=args.max_steps,
        model_name=args.model_name,
        enable_telemetry=args.enable_telemetry,
        writer_description=args.writer_description,
        critic_description=args.critic_description,
        writer_system_prompt=args.writer_prompt,
        critic_system_prompt=args.critic_prompt
    )
    
    # Run the writing task with the prompt
    result = run_writing_task(args.prompt)
    
    # Print the result
    print("\nFinal Draft:")
    print(result)
    
    return result

if __name__ == "__main__":
    main() 