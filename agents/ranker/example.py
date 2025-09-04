#!/usr/bin/env python3
"""Example usage of the RankingAgent.

This script demonstrates how to use the RankingAgent to evaluate and rank
different versions of a document or artifact. It can be run on its own or
integrated with a RankedAgentLoop.
"""

import os
import argparse
from dotenv import load_dotenv
from ..utils.agents.rate_lim_llm import RateLimitedLiteLLMModel
from .agents import RankingAgent

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ranking agent to evaluate and compare artifacts.")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID to analyze (if not provided, a new one is generated)")
    parser.add_argument("--logical-artifact-id", type=str, default="final_report", help="The logical artifact ID to evaluate")
    parser.add_argument("--max-ranklist-size", type=int, default=10, help="Maximum number of artifacts to retain in the ranked list")
    parser.add_argument("--run-data-base-dir", type=str, default="run_data", help="Base directory for run data")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-1.5-flash", help="Model ID for the LLM")
    parser.add_argument("--goal", type=str, default="Create a comprehensive and accurate report", help="The goal for artifact evaluation")
    args = parser.parse_args()

    # Load API keys
    load_dotenv()

    # Check if the run ID exists if provided
    if args.run_id:
        run_dir = os.path.join(args.run_data_base_dir, args.run_id)
        if not os.path.exists(run_dir):
            print(f"Error: Run directory not found for run_id '{args.run_id}'. Please provide a valid run_id.")
            return
        print(f"Using existing run_id: {args.run_id}")
    else:
        print("No run_id provided, a new one will be generated.")

    # Create the ranking agent
    ranking_agent = RankingAgent(
        model_id=args.model_id,
        max_ranklist_size=args.max_ranklist_size,
        run_data_base_dir=args.run_data_base_dir,
        run_id=args.run_id,
        logical_artifact_id=args.logical_artifact_id
    )

    # Create a prompt for ranking
    prompt = f"""Goal: {args.goal}

Please analyze and rank the artifacts for '{args.logical_artifact_id}'.
Provide a summary of the current ranking and comparative analysis of the top artifacts.
If there are at least two ranked artifacts, compare the top two.
"""

    # Run the ranking agent
    print(f"Running ranking agent for '{args.logical_artifact_id}'...")
    ranking_result = ranking_agent.rank_artifacts(prompt)
    
    # Print the result
    print("\n=== Ranking Results ===\n")
    print(ranking_result)
    
    # Optionally start background monitoring
    if input("\nWould you like to start background monitoring for new artifacts? (y/n): ").lower().strip() == 'y':
        ranking_agent.start_background_ranking()
        print(f"Background monitoring started. Press Ctrl+C to stop.")
        try:
            # Keep the main thread running until interrupted
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping background monitoring...")
        finally:
            ranking_agent.stop_background_ranking()
    
    print("Ranking agent example completed.")

if __name__ == "__main__":
    main() 