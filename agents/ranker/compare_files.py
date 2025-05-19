#!/usr/bin/env python3
"""
Standalone script to rank a list of files using the RankingAgent.
Files can be specified directly, via directory paths (all files in the directory),
or via glob patterns.

Example usage:
python -m agents.ranker.compare_files ./docs/*.md my_report.txt ./text_archive/ --goal "Determine which file has better prose style" --output results.txt
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from typing import List, Tuple
import glob

# Add the parent directory to the path if running as a script
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from agents.ranker.agents import RankingAgent
from agents.utils.telemetry import suppress_litellm_logs

def rank_files_pairwise(
    files: List[str], agent: RankingAgent, goal: str
) -> Tuple[List[str], List[str]]:
    """Ranks files from best to worst using pairwise comparisons.
    
    Args:
        files: A list of file paths to rank.
        agent: An instance of RankingAgent.
        goal: The goal for comparison.
        
    Returns:
        A tuple containing:
            - ranked_files: A list of file paths sorted from best to worst.
            - comparison_logs: A list of strings detailing each comparison.
    """
    if not files:
        return [], []
    if len(files) == 1:
        return files, [f"Only one file provided ('{os.path.basename(files[0])}'), no ranking needed."]

    # Start with the first file as the initial ranked list
    ranked_list = [files[0]]
    comparison_logs = []
    
    for i in range(1, len(files)):
        current_file_to_insert = files[i]
        inserted = False
        
        # Compare current_file_to_insert with files in ranked_list from right to left
        for j in range(len(ranked_list) - 1, -1, -1):
            comparison_file_in_ranked_list = ranked_list[j]
            
            # Determine the goal for this specific comparison
            # The RankingAgent's compare_files method will use its default if goal is None.
            # Here, we ensure 'goal' (which might be user-provided or a default) is passed.
            
            log_message_prefix = f"Comparing (for insertion) '{os.path.basename(current_file_to_insert)}' (New) vs. '{os.path.basename(comparison_file_in_ranked_list)}' (Ranked)"
            print(f"  {log_message_prefix} with goal: {goal}")
            
            # 'A' is current_file_to_insert, 'B' is comparison_file_in_ranked_list
            winner, rationale = agent.compare_files(
                current_file_to_insert, comparison_file_in_ranked_list, goal
            )
            
            log_entry = (
                f"{log_message_prefix}\\n"
                f"  - Goal: {goal}\\n"
                f"  - Winner: {winner} ({'New (A)' if winner == 'A' else 'Ranked (B)' if winner == 'B' else 'Equal'})\\n"
                f"  - Rationale: {rationale}"
            )
            comparison_logs.append(log_entry)
            print(f"    -> Winner: {winner}, Rationale: {rationale.strip().replace(os.linesep, ' ')}...")

            if winner == 'A': # New file is better
                # If new file is better than ranked_list[j], and j is 0, it's the new best.
                # Otherwise, continue left to see if it's even better than ranked_list[j-1].
                if j == 0:
                    ranked_list.insert(0, current_file_to_insert)
                    inserted = True
                    break
                # else, continue loop to compare with ranked_list[j-1]
            elif winner == 'Equal': # New file is equal to ranked_list[j]
                # Insert new_file immediately after (to the right of) ranked_list[j].
                # This maintains stability for equal items relative to their input order if j was processed left-to-right.
                # Since we are going right-to-left, inserting at j+1 places it after current ranked_list[j]
                ranked_list.insert(j + 1, current_file_to_insert)
                inserted = True
                break
            else: # Ranked file (B) is better
                # new_file is worse than ranked_list[j]. So, insert new_file after ranked_list[j].
                ranked_list.insert(j + 1, current_file_to_insert)
                inserted = True
                break
        
        if not inserted:
            # This happens if new_file was better than all elements in ranked_list
            # (i.e., loop completed, all comparisons resulted in 'A', and it was inserted at index 0 inside loop)
            # This specific `if not inserted:` outside loop should ideally be covered by logic within.
            # If loop finished and not inserted, means it's better than all, so insert at 0.
            # This case is handled if `j==0 and winner=='A'` leads to insertion.
            # This path should not be hit if logic is correct. Adding safety print if it is.
            # Correction: If the loop finishes, it means current_file_to_insert was better than all ranked_list items
            # or the ranked_list was empty before this (which is handled by initialization).
            # If it's better than all, it should have been inserted at index 0.
            # Let's re-verify insertion sort logic for right-to-left pass.
            # If loop completes (j goes to -1) without insertion, it means file_to_insert is better than all.
            # The 'if j == 0 and winner == 'A':' path correctly inserts at 0.
            # 'Equal' also inserts and breaks. 'B' (ranked is better) inserts and breaks.
            # This 'if not inserted:' should not be strictly necessary if the inner logic is exhaustive.
            # For safety, if somehow it wasn't inserted, it implies it's the best.
            print(f"  Warning: file {os.path.basename(current_file_to_insert)} was not inserted in loop, placing at front.")
            ranked_list.insert(0, current_file_to_insert)

            
        print(f"  Current ranked order: {[os.path.basename(f) for f in ranked_list]}")

    return ranked_list, comparison_logs

def main():
    # Load environment variables and suppress logs
    load_dotenv()
    suppress_litellm_logs()
    
    parser = argparse.ArgumentParser(
        description='Rank a list of files specified directly, by directory, or by glob patterns.'
    )
    parser.add_argument('inputs', nargs='+', type=str, 
                        help='Paths to the files, directories, or glob patterns for files to rank')
    parser.add_argument('--goal', type=str, default=None, 
                        help='Goal for comparison (e.g., "Determine which file has better clarity")')
    parser.add_argument('--model', type=str, default="gemini/gemini-2.0-flash", 
                        help='Model ID to use')
    parser.add_argument('--model-info-path', type=str, 
                        default="agents/utils/gemini/gem_llm_info.json",
                        help='Path to model info JSON')
    parser.add_argument('--output', type=str, default=None, 
                        help='Optional file to write full output to')
    parser.add_argument('--logical-artifact-id', type=str, default="document",
                        help='Logical artifact ID for default goal')
    parser.add_argument('--use-fallback', action='store_true', default=False,
                        help='Enable model fallback mechanism if the primary model encounters rate limits.')
    
    args = parser.parse_args()

    # Expand file inputs
    resolved_files_set = set()
    for item in args.inputs:
        # Use glob to expand patterns. It also handles direct file/dir paths.
        expanded_paths = glob.glob(item)
        if not expanded_paths and not os.path.exists(item): # Check if glob found nothing AND item itself doesn't exist
            print(f"Warning: Input '{item}' did not match any files or directories and does not exist.")
            continue

        for path in expanded_paths:
            if os.path.isfile(path):
                resolved_files_set.add(os.path.abspath(path))
            elif os.path.isdir(path):
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    if os.path.isfile(filepath):
                        resolved_files_set.add(os.path.abspath(filepath))
            # If path is neither file nor dir (e.g. broken symlink globbed), it's ignored.
    
    actual_files_to_rank = sorted(list(resolved_files_set))

    if len(actual_files_to_rank) < 1: # Changed from < 2 to < 1 as rank_files_pairwise handles N=1
        print("Error: No files found to rank after expanding inputs. Please check your input paths/patterns.")
        return 1
    if len(actual_files_to_rank) == 1:
        print(f"Only one file found after expanding inputs: {actual_files_to_rank[0]}")
        print("No ranking needed for a single file.")
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Input Arguments: {args.inputs}\n")
                    f.write(f"Resolved File (1): {actual_files_to_rank[0]}\n")
                    f.write("Goal: Not applicable (single file)\n\n")
                    f.write("Ranking: Only one file, no ranking performed.\n")
                print(f"Output written to {args.output}")
            except Exception as e:
                print(f"Error writing to output file: {e}")
        return 0 # Successful exit for N=1 case.
    
    # Create the model
    model = RateLimitedLiteLLMModel(
        model_id=args.model,
        model_info_path=args.model_info_path,
        base_wait_time=2.0,
        max_retries=3,
        enable_fallback=args.use_fallback
    )
    
    # Create the ranking agent
    ranking_agent = RankingAgent(
        model=model, 
        logical_artifact_id=args.logical_artifact_id
    )
    
    # File existence is implicitly checked by the expansion logic (os.path.isfile)
    # So, no separate loop for file existence check is needed here for actual_files_to_rank.
    
    print(f"Found {len(actual_files_to_rank)} unique files to rank after expanding inputs ({args.inputs}):")
    for i, file_path in enumerate(actual_files_to_rank):
        # File paths are already absolute from the expansion logic
        print(f"{i+1}. {file_path}") 
    
    effective_goal = args.goal
    if not effective_goal:
        effective_goal = f"Improve the {args.logical_artifact_id}" # Default goal uses logical_artifact_id
        print(f"No specific goal provided, using default goal: \"{effective_goal}\"")
    else:
        print(f"Goal: \"{effective_goal}\"")
    
    # Rank the files
    print("\nStarting pairwise ranking process...")
    ranked_files, comparison_logs = rank_files_pairwise(
        actual_files_to_rank, ranking_agent, effective_goal
    )
    
    # Print the result
    print("\n" + "="*80)
    print("FINAL RANKING (BEST TO WORST):")
    for i, file_path in enumerate(ranked_files):
        print(f"{i+1}. {os.path.basename(file_path)} ({file_path})")
    print("="*80)
    
    # Write to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Input Arguments: {args.inputs}\n") # Log original user inputs
                f.write(f"Resolved Files To Rank ({len(actual_files_to_rank)}):\n")
                for file_path in actual_files_to_rank: # Log resolved files
                    f.write(f"- {file_path}\n")
                f.write(f"Goal: {effective_goal}\n\n")
                
                f.write("Final Ranking (Best to Worst):\n")
                for i, file_path in enumerate(ranked_files):
                    f.write(f"{i+1}. {os.path.basename(file_path)} ({file_path})\n")
                f.write("\n" + "="*80 + "\n")
                f.write("Comparison Logs:\n")
                for log_entry in comparison_logs:
                    f.write(f"\n{log_entry}\n")
                f.write("="*80 + "\n")

            print(f"Full results written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 