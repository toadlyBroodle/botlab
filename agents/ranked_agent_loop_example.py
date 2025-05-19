#!/usr/bin/env python3
"""
Example usage of the RankedAgentLoop class.

This example shows how to create and use a RankedAgentLoop instance with different agent types,
focusing on artifact generation and ranking.

Usage:
    python -m agents.ranked_agent_loop_example --query "Your query here"
"""

import os
import sys
import argparse
import json # For reading final ranklist
from agents.ranked_agent_loop import RankedAgentLoop 


def main():
    """Main entry point for the ranked agent loop example."""
    args = parse_args()
    
    print("=== Ranked Agent Loop Example ===")
    print(f"Query: {args.query}")
    print(f"Agent Sequence: {args.agent_sequence}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Max Steps Per Agent: {args.max_steps_per_agent}")
    print(f"Use Custom Prompts: {args.use_custom_prompts}")
    print(f"Enable Telemetry: {args.enable_telemetry}")
    
    # New params
    print(f"Run ID: {args.run_id}")
    print(f"Load Run State: {args.load_run_state}")
    print(f"Ranking LLM Model ID: {args.ranking_llm_model_id}")
    print(f"Max Ranklist Size: {args.max_ranklist_size}")
    print(f"Poll Interval: {args.poll_interval}")
    print(f"Primary Logical Artifact ID: {args.primary_logical_artifact_id}")
    print(f"Run Data Base Dir: {args.run_data_base_dir}")

    if args.user_email:
        # This will be picked up by UserFeedbackAgent if it's in the sequence
        # and no specific 'user_email' is passed via agent_configs to RankedAgentLoop
        os.environ["REMOTE_USER_EMAIL"] = args.user_email
        print(f"User Email (set as REMOTE_USER_EMAIL env var): {args.user_email}")
    print(f"Report Frequency (initial for UserFeedbackAgent): {args.report_frequency}")
    print()
    
    agent_sequence = [agent_type.strip() for agent_type in args.agent_sequence.split(",")]
    
    # Agent configs (e.g., for UserFeedbackAgent if used)
    agent_configs = {}
    if "user_feedback" in agent_sequence:
        # RankedAgentLoop's _get_default_agent_configs will provide defaults
        # for user_feedback_agent_description and user_feedback_agent_prompt.
        # We only need to override specific things like report_frequency or user_email if desired.
        agent_configs["report_frequency"] = args.report_frequency
        if args.user_email: # If CLI arg provided, ensure it's used
            agent_configs["user_email"] = args.user_email

    # Initialize the ranked agent loop parameters
    ranked_agent_loop_params = {
        "agent_sequence": agent_sequence,
        "max_iterations": args.max_iterations,
        "max_retries": args.max_retries, 
        "model_id": args.model_id, 
        "use_custom_prompts": args.use_custom_prompts,
        "enable_telemetry": args.enable_telemetry,
        "ranking_llm_model_id": args.ranking_llm_model_id,
        "max_ranklist_size": args.max_ranklist_size,
        "poll_interval": args.poll_interval,
        "primary_logical_artifact_id": args.primary_logical_artifact_id,
        "run_data_base_dir": args.run_data_base_dir,
        "run_id": args.run_id,
        "load_run_state": args.load_run_state,
        "agent_configs": agent_configs if agent_configs else None # Pass if not empty
    }
    
    if args.max_steps_per_agent is not None:
        ranked_agent_loop_params["max_steps_per_agent"] = args.max_steps_per_agent
    
    # Initialize the RankedAgentLoop
    agent_loop = RankedAgentLoop(**ranked_agent_loop_params)
    
    final_state = None
    try:
        final_state = agent_loop.run(args.query)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down loop...")
    finally:
        agent_loop.close() # Important to shut down the ranking agent thread
    
    if final_state:
        print(f"\n=== Loop Ended for Run ID: {final_state.get('run_id')} ===")
        print(f"Status: {final_state.get('status')}")
        print(f"Iterations Completed: {final_state.get('current_iteration')}")
        if final_state.get("error"):
            print(f"Error: {final_state['error']}")

        # Display the top-ranked artifact details
        if final_state.get('status') in ["completed", "running", "error"]: # Even if error, ranklist might exist
            ranklist_path = os.path.join(agent_loop.run_dir, "ranking_state", f"{agent_loop.primary_logical_artifact_id}.ranklist.json")
            print(f"\nAttempting to read final ranklist from: {ranklist_path}")
            if os.path.exists(ranklist_path):
                try:
                    with open(ranklist_path, 'r') as f_rank:
                        content = f_rank.read().strip()
                        if content:
                            final_ranklist = json.loads(content)
                            if final_ranklist:
                                best_artifact_id = final_ranklist[0]
                                print(f"Top ranked artifact ID for '{agent_loop.primary_logical_artifact_id}': {best_artifact_id}")
                                
                                # Load and print a preview of the best artifact's content
                                best_content = agent_loop._load_artifact_content(best_artifact_id)
                                if best_content:
                                    print("\n--- Preview of Top Ranked Artifact ---")
                                    print(best_content[:500] + ("..." if len(best_content) > 500 else ""))
                                else:
                                    print(f"Could not load content for artifact ID: {best_artifact_id}")
                            else:
                                print(f"Final ranklist for '{agent_loop.primary_logical_artifact_id}' is empty.")
                        else:
                             print(f"Final ranklist file '{ranklist_path}' is empty.")
                except Exception as e:
                    print(f"Error reading or displaying final ranklist: {e}")
            else:
                print(f"Final ranklist file not found: {ranklist_path}")

        # Display UserFeedbackAgent command log if it exists
        if "user_feedback_commands_log" in final_state and final_state["user_feedback_commands_log"]:
            print("\n=== User Feedback Command Log ===")
            for log_entry in final_state["user_feedback_commands_log"]:
                print(f"- Iteration {log_entry['iteration']}: Commands {log_entry['commands']}")
        elif "user_feedback" in agent_sequence: # If UFA was in sequence but no commands logged
            print("\nNo commands were logged from UserFeedbackAgent during this run.")

    else:
        print("\nLoop did not return a final state (e.g. due to early exit or critical error during init).")


def parse_args():
    parser = argparse.ArgumentParser(description="Example of using the RankedAgentLoop class with artifact ranking.")
    
    parser.add_argument("--query", type=str, default="Write a brief report on recent (2025) advances in AI agents", help="The query or goal for the primary artifact")
    parser.add_argument("--agent-sequence", type=str, default="writer,user_feedback", help="Comma-separated list of agent types to call in sequence") # Added user_feedback for testing
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum number of iterations through the entire sequence")
    parser.add_argument("--max-steps-per-agent", type=str, default="2,1", help="Maximum steps for each agent. Example: '2,1' for [writer,user_feedback]")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID for the main agents (researcher, writer, editor, etc.)")
    parser.add_argument("--use-custom-prompts", action="store_true", help="Whether to use custom agent descriptions and prompts loaded by AgentLoop")
    parser.add_argument("--enable-telemetry", action="store_true", help="Whether to enable OpenTelemetry tracing")
    
    # Args for UserFeedbackAgent (if included in sequence)
    parser.add_argument("--user-email", type=str, default=os.getenv("REMOTE_USER_EMAIL"), help="Email address for user feedback (UserFeedbackAgent). Uses REMOTE_USER_EMAIL env var if not set.")
    parser.add_argument("--report-frequency", type=int, default=1, help="Initial report frequency for UserFeedbackAgent (1 = every iteration it runs)")

    # New arguments for ranking and run management
    parser.add_argument("--ranking-llm-model-id", type=str, default="gemini/gemini-1.5-flash", help="Model ID for the LLM judge in the ranking agent")
    parser.add_argument("--max-ranklist-size", type=int, default=5, help="Maximum number of artifact IDs to retain in each ranked list")
    parser.add_argument("--poll-interval", type=float, default=10, help="Ranking agent poll interval for new metadata (seconds)")
    parser.add_argument("--primary-logical-artifact-id", type=str, default="collaboration_protocol_draft", help="Identifier for the main conceptual artifact being evolved by the loop")
    parser.add_argument("--run-data-base-dir", type=str, default="run_data", help="Base directory where all run-specific subdirectories will be created")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run ID to use. If None, a new one is generated. Useful for resuming or analyzing a specific run.")
    parser.add_argument("--load-run-state", action="store_true", help="If --run-id is provided, attempt to load its state and resume the loop.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM calls")

    return parser.parse_args()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main() 