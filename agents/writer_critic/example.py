#!/usr/bin/env python3
"""
Example usage of the WriterAgent and CriticAgent classes.

This example shows how to create and use a WriterAgent instance with a CriticAgent.
It also provides a command-line interface for running writing tasks.

Usage:
    poetry run python -m writer_critic.example --prompt "Your writing prompt here"
"""

import os
import argparse
from dotenv import load_dotenv
from utils.telemetry import suppress_litellm_logs
from writer_critic.agents import WriterAgent, CriticAgent
from writer_critic.tools import DRAFT_DIR

def setup_basic_environment():
    """Set up basic environment for the example"""
    # Ensure drafts directory exists
    os.makedirs(DRAFT_DIR, exist_ok=True)
    
    # Load .env from root directory
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def run_example(prompt=None, max_steps=5, model_id="gemini/gemini-2.0-flash", 
                model_info_path="utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                writer_description=None, critic_description=None,
                writer_prompt=None, critic_prompt=None):
    """Run a writing task using the WriterAgent and CriticAgent classes
    
    Args:
        prompt: The writing prompt
        max_steps: Maximum number of steps for the writer agent
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        writer_description: Optional custom description for the writer agent
        critic_description: Optional custom description for the critic agent
        writer_prompt: Optional custom system prompt for the writer agent
        critic_prompt: Optional custom system prompt for the critic agent
        
    Returns:
        The final draft from the writer agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Use default descriptions if none provided
    if writer_description is None:
        writer_description = "Specialized in creative writing with a focus on engaging narratives."
    
    if critic_description is None:
        critic_description = "Specialized in providing constructive feedback on creative writing."
    
    print(f"Creating writer agent with max_steps={max_steps}")
    
    # Create the writer agent (which will create its own critic agent)
    writer = WriterAgent(
        max_steps=max_steps,
        agent_description=writer_description,
        critic_description=critic_description,
        system_prompt=writer_prompt,
        critic_prompt=critic_prompt,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )
    
    # Use default prompt if none provided
    if prompt is None:
        prompt = "Write a short story about a robot who discovers emotions."
    
    print(f"Running writing prompt: {prompt}")
    print("=" * 80)
    
    # Run the writing task and get the result
    result = writer.write_draft(prompt)
    
    print("=" * 80)
    print("Writing complete! The draft has been saved to the drafts directory.")
    
    return result

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the WriterAgent with a prompt.")
    parser.add_argument("--prompt", type=str, 
                        default="Write a short story about a robot who discovers emotions.",
                        help="The writing prompt")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--writer-description", type=str, help="Custom description for the writer agent")
    parser.add_argument("--critic-description", type=str, help="Custom description for the critic agent")
    parser.add_argument("--writer-prompt", type=str, help="Custom system prompt for the writer agent")
    parser.add_argument("--critic-prompt", type=str, help="Custom system prompt for the critic agent")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    run_example(
        prompt=args.prompt,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        writer_description=args.writer_description,
        critic_description=args.critic_description,
        writer_prompt=args.writer_prompt,
        critic_prompt=args.critic_prompt
    ) 