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
        print(f"User Email: {args.user_email}")
    print(f"Report Frequency: {args.report_frequency}")
    print()
    
    # Parse agent sequence
    agent_sequence = [agent_type.strip() for agent_type in args.agent_sequence.split(",")]
    
    # Set user email environment variable if provided
    if args.user_email:
        os.environ["LOCAL_USER_EMAIL"] = args.user_email
    
    # Initialize the agent loop with default parameters
    agent_loop_params = {
        "agent_sequence": agent_sequence,
        "max_iterations": args.max_iterations,
        "max_retries": 3,
        "use_custom_prompts": args.use_custom_prompts,
        "enable_telemetry": args.enable_telemetry,
        "state_file": args.state_file,
        "load_state": args.load_state
    }
    
    # Only add max_steps_per_agent if explicitly provided
    if args.max_steps_per_agent is not None:
        agent_loop_params["max_steps_per_agent"] = args.max_steps_per_agent
    
    # Add user feedback configuration
    if "user_feedback" in agent_sequence:
        agent_loop_params["agent_configs"] = {
            "report_frequency": args.report_frequency
        }
    
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
        
        if not final_agent:
            final_agent = agent_sequence[-1]  # Default to last agent if all are QAQC/user_feedback
            
        final_result = result["results"].get(final_agent)
        
        if final_result:
            print(f"\n{final_result}")
        else:
            print("No final result available.")
    else:
        print(f"\nAgent loop ended with status: {result['status']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
    
    # Print some statistics
    print("\n=== Statistics ===")
    print(f"Total iterations completed: {result['iteration']}")
    
    # Print QAQC results if available
    if "qaqc" in agent_sequence and "qaqc" in result["results"]:
        print("\n=== QAQC Results ===")
        qaqc_result = result["results"]["qaqc"]
        print(qaqc_result[:500] + "..." if len(qaqc_result) > 500 else qaqc_result)
    
    # Print user feedback results if available
    if "user_feedback" in agent_sequence and "user_feedback" in result["results"]:
        print("\n=== User Feedback Results ===")
        feedback_result = result["results"]["user_feedback"]
        print(feedback_result)
    
    # Print results from each agent in the final iteration
    print("\n=== Results from each agent ===")
    for agent_type in agent_sequence:
        result_key = f"{agent_type}_{result['iteration'] - 1}" if result['iteration'] > 0 else agent_type
        agent_result = result["results"].get(result_key)
        if agent_result:
            print(f"\n--- {agent_type.upper()} ---")
            # Print a preview of the result (first 200 characters)
            preview = agent_result[:200] + "..." if len(agent_result) > 200 else agent_result
            print(preview)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example of using the AgentLoop class")
    
    parser.add_argument("--query", type=str, default="Write brief report on recent (2025) advances in AI agents",
                        help="The query to process")
    parser.add_argument("--agent-sequence", type=str, default="researcher,writer,editor,user_feedback,qaqc", 
                        help="Comma-separated list of agent types to call in sequence")
    parser.add_argument("--max-iterations", type=int, default=10, 
                        help="Maximum number of iterations through the entire sequence")
    parser.add_argument("--max-steps-per-agent", type=str, default="5,4,9,3,1",
                        help="Maximum steps for each agent. Can be either: "
                             "- An integer (same value for all agents) "
                             "- A comma-separated string (e.g., '5,4,9,3,1' for different values per agent)")
    parser.add_argument("--state-file", type=str, default="agents/logs/agent_loop_state.json",
                        help="Path to a file for persisting state between runs")
    parser.add_argument("--load-state", action="store_true", 
                        help="Whether to load state from state_file if it exists (default: False)")
    parser.add_argument("--use-custom-prompts", action="store_true", 
                        help="Whether to use custom agent descriptions and prompts")
    parser.add_argument("--enable-telemetry", action="store_true", 
                        help="Whether to enable OpenTelemetry tracing")
    parser.add_argument("--user-email", type=str, default=None,
                        help="Email address for user feedback (defaults to LOCAL_USER_EMAIL env var)")
    parser.add_argument("--report-frequency", type=int, default=1,
                        help="How often to send user feedback reports (1 = every iteration)")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 