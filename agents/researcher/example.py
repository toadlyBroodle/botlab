#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the main module
from agents.researcher.main import initialize, run_agent_with_query, parse_arguments, main as run_main

def main():
    """Example script to demonstrate how to use the researcher CodeAgent.
    
    This script shows two ways to use the researcher agent:
    1. Using the main() function from main.py to handle everything
    2. Using the initialize() and run_agent_with_query() functions for more control
    
    You can provide a custom query as a command-line argument:
    ./example.py "Your custom query here"
    """
    # Get custom query from command line if provided
    custom_query = None
    if len(sys.argv) > 1:
        custom_query = sys.argv[1]
        print(f"Using custom query from command line: {custom_query}")
    
    # Option 1: Use the main function from main.py
    # This will parse command-line arguments and run the agent
    print("Running the researcher agent using main.py's main() function...")
    result = run_main()
    
    # Option 2: Initialize the agent and run a custom query
    # This demonstrates how to use the agent programmatically
    print("\n\nRunning a custom query programmatically...")
    
    # Initialize the agent with default parameters
    agent = initialize()
    
    # Run a custom query (either from command line or default)
    if not custom_query:
        custom_query = "What is quantum computing and how does it differ from classical computing?"
    run_agent_with_query(agent, custom_query)
    
    return result

if __name__ == "__main__":
    main() 