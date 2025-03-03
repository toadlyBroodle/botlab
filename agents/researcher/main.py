#!/usr/bin/env python3
import os
import sys
import time
import argparse
from typing import Optional
from dotenv import load_dotenv
from utils.telemetry import start_telemetry
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

from researcher.agents import create_researcher_agent
from researcher.tools import PAPERS_DIR, REPORTS_DIR

def setup_environment():
    """Set up environment variables and API keys"""
    # Ensure directories exist
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load .env from root directory
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return api_key

def initialize(
    max_steps: int = 20, 
    enable_telemetry: bool = False,
    base_wait_time: float = 2.0, 
    max_retries: int = 3,
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    model_id: str = "gemini/gemini-2.0-flash",
    researcher_description: Optional[str] = None,
    researcher_prompt: Optional[str] = None
):
    """Initialize the system with optional telemetry and return the researcher agent
    
    The researcher agent supports the following tools:
    1. Web search via DuckDuckGo (with rate limiting)
    2. Web page scraping and content extraction
    3. arXiv academic paper search with advanced query options
    4. PDF to markdown conversion for research papers
    
    Args:
        max_steps: Maximum number of steps for the agent
        enable_telemetry: Whether to enable OpenTelemetry tracing
        base_wait_time: Base wait time in seconds for rate limiting
        max_retries: Maximum number of retry attempts for rate limiting
        model_info_path: Path to the model info JSON file
        model_id: The model ID to use (default: gemini/gemini-2.0-flash)
        researcher_description: Optional additional description for the researcher agent
        researcher_prompt: Optional custom system prompt for the researcher agent
        
    Returns:
        A function that can process research queries
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    setup_environment()
    
    # Create a rate-limited model with model-specific rate limits
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
    )
    
    # Create researcher agent
    researcher_agent = create_researcher_agent(
        model=model, 
        max_steps=max_steps,
        researcher_description=researcher_description,
        researcher_prompt=researcher_prompt
    )
    
    def run_research_query(query: str) -> str:
        """Run the agent with a query and return the result
        
        Args:
            query: The query to run
            
        Returns:
            The result from the agent
        """
        # Time the query execution
        start_time = time.time()
        
        # Run the query directly on the agent
        result = researcher_agent.run(query)
        
        # Calculate and print execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        return result
        
    return run_research_query

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the researcher CodeAgent with a query.")
    parser.add_argument("--query", type=str, default="What are the latest advancements in agentic AI systems? Include information from recent arXiv papers.", 
                        help="The query to research")
    parser.add_argument("--enable-telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--researcher-description", type=str, default=None, help="Custom description for the researcher agent")
    parser.add_argument("--researcher-prompt", type=str, default=None, help="Custom system prompt for the researcher agent")
    
    return parser.parse_args()

def main():
    """Main entry point when run directly"""
    args = parse_arguments()
    
    # Initialize the researcher agent with parameters from command line
    run_research_query = initialize(
        enable_telemetry=args.enable_telemetry,
        max_steps=args.max_steps,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        researcher_description=args.researcher_description,
        researcher_prompt=args.researcher_prompt
    )
    
    # Run the agent with the query
    result = run_research_query(args.query)
    
    return result

if __name__ == "__main__":
    main()