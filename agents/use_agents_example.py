#!/usr/bin/env python3
"""
Example of using the agents package from the root directory.

This example shows how to import and use the agents when the code is located
at the root level, demonstrating the new standardized import system.

Usage:
    python use_agents_example.py --query "Your query here"
"""

import os
import argparse
from dotenv import load_dotenv

# Import from the agents package
from agents import AgentLoop
from agents.researcher.agents import ResearcherAgent
from agents.utils.telemetry import suppress_litellm_logs

def main():
    """Main entry point for the example."""
    # Set up environment
    load_dotenv()
    suppress_litellm_logs()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Example of using the agents package")
    parser.add_argument("--query", type=str, help="Query to run", 
                       default="What are the latest advancements in quantum computing?")
    args = parser.parse_args()
    
    print(f"Running query: {args.query}")
    
    # Create and run a ResearcherAgent
    researcher = ResearcherAgent(
        max_steps=15,
        model_id="gemini/gemini-2.0-flash",
        model_info_path="agents/utils/gemini/gem_llm_info.json",
        base_wait_time=2.0,
        max_retries=3
    )
    
    # Run the query
    result = researcher.run_query(args.query)
    
    print("\nResearch completed!")
    print("=" * 50)
    
    # You can use an agent loop as well
    print("\nNow running the same query through an agent loop...")
    agent_loop = AgentLoop(
        agent_sequence=["researcher"],
        max_iterations=1,
        max_steps_per_agent=15
    )
    
    loop_result = agent_loop.run(args.query)
    print("\nAgent loop completed!")
    
    return result

if __name__ == "__main__":
    main() 