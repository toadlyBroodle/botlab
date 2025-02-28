#!/usr/bin/env python3
import os
import sys
import time
import argparse
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
    """Initialize the system with optional telemetry and return the researcher agent
    
    The researcher agent now supports:
    1. Web search via DuckDuckGo (with rate limiting)
    2. Web page scraping and content extraction
    3. arXiv academic paper search with advanced query options
    4. PDF to markdown conversion for research papers
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        max_steps: Maximum number of steps for the agent
        base_wait_time: Base wait time in seconds for rate limiting
        max_retries: Maximum number of retry attempts for rate limiting
        model_info_path: Path to the model info JSON file
        model_id: The model ID to use (default: gemini/gemini-2.0-flash)
        
    Returns:
        The configured researcher CodeAgent
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
    
    return researcher_agent

def run_agent_with_query(agent, query, verbose=True):
    """Run the agent with a query and return the result
    
    Args:
        agent: The agent to run
        query: The query to run
        verbose: Whether to print progress information
        
    Returns:
        The result from the agent
    """
    if verbose:
        print(f"\nProcessing query: {query}")
        
    # Time the query execution
    start_time = time.time()
    
    # Run the query directly on the agent
    result = agent.run(query)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    
    if verbose:
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print("\nResult:")
        print(result)
    
    return result

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the researcher CodeAgent with a query.")
    parser.add_argument("--query", type=str, default="What are the latest advancements in large language models? Include information from recent arXiv papers.", 
                        help="The query to research")
    parser.add_argument("--enable-telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    return parser.parse_args()

def main():
    """Main entry point when run directly"""
    args = parse_arguments()
    
    # Initialize the researcher agent with parameters from command line
    researcher_agent = initialize(
        enable_telemetry=args.enable_telemetry,
        max_steps=args.max_steps,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path
    )
    query = args.query
    
    # Run the agent with the query
    return run_agent_with_query(researcher_agent, query, verbose=not args.quiet)

if __name__ == "__main__":
    main()