#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the main module
from agents.researcher.main import initialize, run_agent_with_query, parse_arguments, main as run_main

def main():
    """Example script to demonstrate how to use the researcher CodeAgent.
    
    This script shows three ways to use the researcher agent:
    1. Using the main() function from main.py to handle everything
    2. Using the initialize() and run_agent_with_query() functions for more control
    3. Using a custom query from command line arguments
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the researcher CodeAgent with a custom query")
    parser.add_argument("query", nargs="?", default=None, help="Custom query to research")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of steps")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--skip-main", action="store_true", help="Skip running the main.py example")
    args = parser.parse_args()
    
    # Get custom query from command line if provided
    custom_query = args.query
    
    # Option 1: Use the main function from main.py
    # This will parse command-line arguments and run the agent
    if not args.skip_main:
        print("Running the researcher agent using main.py's main() function...")
        result = run_main()
        print("\n" + "="*80 + "\n")
    
    # Option 2: Initialize the agent and run a custom query
    # This demonstrates how to use the agent programmatically
    print("Running a custom query programmatically...")
    
    # Initialize the agent with parameters from command line
    agent = initialize(
        max_steps=args.max_steps,
        model_id=args.model
    )
    
    # Run a custom query (either from command line or default)
    if not custom_query:
        custom_query = "What is quantum computing and how does it differ from classical computing?"
    
    print(f"Query: {custom_query}")
    print("This CodeAgent can write Python code to call tools like web_search, visit_webpage, arxiv_search, and pdf_to_markdown.")
    
    # Run the query
    result = run_agent_with_query(agent, custom_query)
    
    return result

if __name__ == "__main__":
    main() 