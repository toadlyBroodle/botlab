#!/usr/bin/env python3
"""
Standalone script to compare two files using the RankingAgent.

Example usage:
python -m agents.ranker.compare_files file1.txt file2.txt --goal "Determine which file has better prose style" --output results.txt
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path if running as a script
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from agents.ranker.agents import RankingAgent
from agents.utils.telemetry import suppress_litellm_logs

def format_winner_message(winner, file_a_path, file_b_path):
    """Format a consistent winner message based on the comparison result.
    
    Args:
        winner: The comparison result ('A', 'B', 'Equal', or 'Error')
        file_a_path: Path to file A
        file_b_path: Path to file B
        
    Returns:
        A formatted winner message string
    """
    if winner == 'A':
        return f"WINNER: File A - {os.path.basename(file_a_path)}"
    elif winner == 'B':
        return f"WINNER: File B - {os.path.basename(file_b_path)}"
    elif winner == 'Equal':
        return "EQUAL - Both files judged equivalent"
    else:
        return "ERROR - Could not determine winner"

def main():
    # Load environment variables and suppress logs
    load_dotenv()
    suppress_litellm_logs()
    
    parser = argparse.ArgumentParser(description='Compare two files and rank them.')
    parser.add_argument('file_a', type=str, help='Path to the first file')
    parser.add_argument('file_b', type=str, help='Path to the second file')
    parser.add_argument('--goal', type=str, default=None, 
                        help='Goal for comparison (e.g., "Determine which file has better clarity")')
    parser.add_argument('--model', type=str, default="gemini/gemini-1.5-flash", 
                        help='Model ID to use')
    parser.add_argument('--model-info-path', type=str, 
                        default="agents/utils/gemini/gem_llm_info.json",
                        help='Path to model info JSON')
    parser.add_argument('--output', type=str, default=None, 
                        help='Optional file to write full output to')
    parser.add_argument('--logical-artifact-id', type=str, default="document",
                        help='Logical artifact ID for default goal')
    
    args = parser.parse_args()
    
    # Create the model
    model = RateLimitedLiteLLMModel(
        model_id=args.model,
        model_info_path=args.model_info_path,
        base_wait_time=2.0,
        max_retries=3
    )
    
    # Create the ranking agent
    ranking_agent = RankingAgent(
        model=model, 
        logical_artifact_id=args.logical_artifact_id
    )
    
    # Verify files exist
    for file_path in [args.file_a, args.file_b]:
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return 1
    
    print(f"Comparing files:")
    print(f"A: {os.path.abspath(args.file_a)}")
    print(f"B: {os.path.abspath(args.file_b)}")
    if args.goal:
        print(f"Goal: {args.goal}")
    else:
        print(f"Goal: Improve the {args.logical_artifact_id}")
    
    # Compare the files
    winner, rationale = ranking_agent.compare_files(args.file_a, args.file_b, args.goal)
    
    # Print the result
    print("\n" + "="*80)
    print(format_winner_message(winner, args.file_a, args.file_b))
    print("="*80)
    print("RATIONALE:")
    print(rationale)
    print("="*80)
    
    # Write to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"File A: {args.file_a}\n")
                f.write(f"File B: {args.file_b}\n")
                f.write(f"Goal: {args.goal or 'Default: Improve the ' + args.logical_artifact_id}\n")
                
                # Write winner with filename
                f.write(f"Winner: {format_winner_message(winner, args.file_a, args.file_b)}\n")
                f.write("\n")
                f.write("Rationale:\n")
                f.write(rationale)
            print(f"Full results written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 