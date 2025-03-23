#!/usr/bin/env python3
"""
Example usage of the ResearcherAgent class.

This example shows how to create and use a ResearcherAgent instance directly.
It also provides a command-line interface for running research queries.

Usage:
    poetry run python -m agents.researcher.example --query "Your research query here"
"""

import os
import argparse
from dotenv import load_dotenv
from agents.utils.telemetry import suppress_litellm_logs
from agents.researcher.agents import ResearcherAgent
from agents.researcher.tools import PAPERS_DIR, REPORTS_DIR

def setup_basic_environment():
    """Set up basic environment for the example"""
    # Ensure directories exist
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load .env from root directory
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def run_example(query=None, max_steps=15, model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                researcher_description=None, researcher_prompt=None):
    """Run a research query using the ResearcherAgent class
    
    Args:
        query: The research query to run
        max_steps: Maximum number of steps for the agent
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        researcher_description: Optional custom description for the researcher agent
        researcher_prompt: Optional custom system prompt for the researcher agent
        
    Returns:
        The result from the agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Create the researcher agent
    researcher = ResearcherAgent(
        max_steps=max_steps,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        researcher_description=researcher_description,
        researcher_prompt=researcher_prompt
    )
    
    # Use default query if none provided
    if query is None:
        query = "What are the latest advancements in large language models? Focus on papers from the last year."
    
    print(f"Running research query: {query}")
    print("=" * 80)
    
    # Run the query and get the result
    result = researcher.run_query(query)
    
    print("\nResearch completed!")
    print("=" * 50)
    print(result)
    
    return result

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the ResearcherAgent with a query.")
    parser.add_argument("--query", type=str, 
                        default="What are the latest advancements in large language models? Focus on papers from the last year.",
                        help="The query to research")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--researcher-description", type=str, help="Custom description for the researcher agent")
    parser.add_argument("--researcher-prompt", type=str, help="Custom system prompt for the researcher agent")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    run_example(
        query=args.query,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        researcher_description=args.researcher_description,
        researcher_prompt=args.researcher_prompt
    ) 