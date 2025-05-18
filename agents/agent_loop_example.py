#!/usr/bin/env python3
"""
Example usage of the AgentLoop class.

This example shows how to create and use an AgentLoop instance with different agent types.
It also provides a command-line interface for running agent loops.

Usage:
    python -m agents.agent_loop_example --query "Your query here"
"""

import os
import sys
import argparse
from agents.agent_loop import AgentLoop


def main():
    """Main entry point for the agent loop example."""
    args = parse_args()
    
    print("=== Agent Loop Example with User Feedback ===")
    print(f"Query: {args.query}")
    print(f"Agent Sequence: {args.agent_sequence}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Max Steps Per Agent: {args.max_steps_per_agent}")
    print(f"State File: {args.state_file}")
    print(f"Load State: {args.load_state}")
    print(f"Use Custom Prompts: {args.use_custom_prompts}")
    print(f"Enable Telemetry: {args.enable_telemetry}")
    if args.user_email:
        print(f"User Email (for REMOTE_USER_EMAIL): {args.user_email}")
    print(f"Report Frequency (initial): {args.report_frequency}")
    print()
    
    # Parse agent sequence
    agent_sequence = [agent_type.strip() for agent_type in args.agent_sequence.split(",")]
    
    # If user_email is provided via CLI, set it as an environment variable
    # so UserFeedbackAgent can pick it up as REMOTE_USER_EMAIL if not already set.
    # UserFeedbackAgent constructor also accepts user_email directly which takes precedence.
    if args.user_email:
        os.environ["REMOTE_USER_EMAIL"] = args.user_email
        print(f"Set REMOTE_USER_EMAIL environment variable to: {args.user_email}")

    agent_configs = {}
    if "user_feedback" in agent_sequence:
        agent_configs["report_frequency"] = args.report_frequency
        # If args.user_email is provided, it will be used by UserFeedbackAgent
        # either through its constructor (if passed in agent_configs) or via env var.
        # AgentLoop passes agent_configs.user_email to UserFeedbackAgent constructor.
        if args.user_email:
             agent_configs["user_email"] = args.user_email

    # Initialize the agent loop with default parameters
    agent_loop_params = {
        "agent_sequence": agent_sequence,
        "max_iterations": args.max_iterations,
        "max_retries": 3,
        "use_custom_prompts": args.use_custom_prompts,
        "enable_telemetry": args.enable_telemetry,
        "state_file": args.state_file,
        "load_state": args.load_state,
        "agent_configs": agent_configs if agent_configs else None # Pass if not empty
    }
    
    # Only add max_steps_per_agent if explicitly provided (it has a default in AgentLoop)
    if args.max_steps_per_agent is not None:
        agent_loop_params["max_steps_per_agent"] = args.max_steps_per_agent
    
    # Initialize the agent loop
    agent_loop = AgentLoop(**agent_loop_params)
    
    # Run the agent loop
    result = agent_loop.run(args.query)
    
    # Print the final result
    if result["status"] == "completed":
        print("\n=== Final Results ===")
        
        # Get the final result from the last non-QAQC and non-user_feedback agent in the sequence
        final_agent = None
        for agent in reversed(agent_sequence):
            if agent not in ["qaqc", "user_feedback"]:
                final_agent = agent
                break
        
        if not final_agent and agent_sequence:
            final_agent = agent_sequence[-1]  # Default to last agent if all are QAQC/user_feedback or empty
            
        final_content = result["results"].get(final_agent) if final_agent else None
        
        if final_content:
            print(f"\n{final_content}")
        else:
            print("No final content available from primary agents.")
    else:
        print(f"\nAgent loop ended with status: {result['status']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
    
    # Print some statistics
    print("\n=== Statistics ===")
    print(f"Total iterations completed: {result.get('current_iteration', result.get('iteration', 0))}") # Use current_iteration if available
    
    # Print QAQC results if available
    if "qaqc" in agent_sequence and "qaqc" in result["results"]:
        print("\n=== QAQC Results ===")
        qaqc_result = result["results"]["qaqc"]
        print(qaqc_result[:500] + "..." if len(qaqc_result) > 500 else qaqc_result)
    
    # Print a summary of user feedback interactions if available
    if "user_feedback_commands_log" in result and result["user_feedback_commands_log"]:
        print("\n=== User Feedback Command Log ===")
        for log_entry in result["user_feedback_commands_log"]:
            print(f"- Iteration {log_entry['iteration']}: Commands {log_entry['commands']}")
    elif "user_feedback" in agent_sequence:
        print("\n=== User Feedback Results ===")
        # The direct 'user_feedback' key in results now holds a summary string per iteration
        feedback_iteration_key = f"user_feedback_{result.get('current_iteration', result.get('iteration', 1)) -1}"
        feedback_summary = result["results"].get(feedback_iteration_key, "No specific feedback summary for last iteration.")
        print(feedback_summary)

    
    # Print results from each agent in the final iteration
    print("\n=== Results from each agent (final iteration) ===")
    final_iteration_index = result.get('current_iteration', result.get('iteration', 1)) -1
    if final_iteration_index < 0: final_iteration_index = 0 # Handle case where loop didn't run an iteration

    for agent_type in agent_sequence:
        # The result key in AgentLoop is now <agent_type>_iteration for history
        # and just <agent_type> for the very latest.
        # For per-iteration display, we use the iteration-specific key.
        result_key_iterated = f"{agent_type}_{final_iteration_index}"
        agent_result = result["results"].get(result_key_iterated)
        
        if agent_result:
            print(f"\n--- {agent_type.upper()} (Iteration {final_iteration_index + 1}) ---")
            preview = agent_result[:200] + "..." if len(agent_result) > 200 else agent_result
            print(preview)
        # else:
            # print(f"\n--- {agent_type.upper()} (Iteration {final_iteration_index + 1}) --- No result stored.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example of using the AgentLoop class")
    
    parser.add_argument("--query", type=str, default="Write brief report on recent (2025) advances in AI agents", help="The query to process")
    parser.add_argument("--agent-sequence", type=str, default="researcher,writer,editor,user_feedback,qaqc", help="Comma-separated list of agent types to call in sequence")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum number of iterations through the entire sequence")
    parser.add_argument("--max-steps-per-agent", type=str, default="5,4,9,3,1", help="Maximum steps for each agent. Default values are for [researcher,writer,editor,user_feedback,qaqc]. UserFeedback is 1 as it's mostly programmatic.")
    parser.add_argument("--state-file", type=str, default="agents/logs/agent_loop_example_state.json", help="Path to a file for persisting state between runs") # Different state file for example
    parser.add_argument("--load-state", action="store_true", help="Whether to load state from state_file if it exists (default: False)")
    parser.add_argument("--use-custom-prompts", action="store_true", help="Whether to use custom agent descriptions and prompts loaded by AgentLoop")
    parser.add_argument("--enable-telemetry", action="store_true", help="Whether to enable OpenTelemetry tracing")
    parser.add_argument("--user-email", type=str, default=None, help="Email address for user feedback (sets REMOTE_USER_EMAIL for the run)")
    parser.add_argument("--report-frequency", type=int, default=1, help="Initial report frequency for UserFeedbackAgent (1 = every iteration)")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 