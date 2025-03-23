#!/usr/bin/env python3
"""
Example usage of the UserFeedbackAgent class.

This example shows how to create and use a UserFeedbackAgent instance directly.
It also provides a command-line interface for testing email communication.

Usage:
    poetry run python -m agents.user_feedback.example --email your-email@example.com
"""

import os
import argparse
from dotenv import load_dotenv
from agents.utils.telemetry import suppress_litellm_logs
from agents.user_feedback.agents import UserFeedbackAgent

def setup_basic_environment():
    """Set up basic environment for the example"""
    # Load .env from root directory
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def run_example(user_email=None, max_steps=4, model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                report_frequency=1,
                mailbox_path=None,
                agent_description=None, agent_prompt=None):
    """Run a test of the UserFeedbackAgent
    
    Args:
        user_email: Email address to communicate with
        max_steps: Maximum number of steps for the agent
        report_frequency: How often to send reports (1 = every iteration)
        mailbox_path: Path to the mailbox file to check for emails
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        agent_description: Optional custom description for the agent
        agent_prompt: Optional custom system prompt for the agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Create the agent
    agent = UserFeedbackAgent(
        max_steps=max_steps,
        user_email=user_email,
        report_frequency=report_frequency,
        mailbox_path=mailbox_path,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        agent_description=agent_description,
        agent_prompt=agent_prompt
    )
    
    print(f"Created UserFeedbackAgent with email: {agent.user_email}")
    print(f"Report frequency: Every {agent.report_frequency} iterations")
    print(f"Mailbox path: {agent.mailbox_path}")
    
    # Create a sample state
    sample_state = {
        "iteration": 1,
        "current_agent": "researcher",
        "progress": {
            "researcher": "Completed initial search on topic",
            "writer": "Not started",
            "editor": "Not started",
            "qaqc": "Not started"
        },
        "query": "Write a report on recent advances in AI"
    }
    
    # Process feedback and update state
    print("\nProcessing feedback and checking for emails...")
    updated_state = agent.process_feedback(sample_state)
    
    # Generate a report
    print("\nGenerating a sample report...")
    report = agent.generate_report(updated_state)
    print(f"\nSample report:\n{report}")
    
    return agent

def main():
    """Main entry point for the example."""
    args = parse_arguments()
    
    print("=== UserFeedbackAgent Example ===")
    agent = run_example(
        user_email=args.email,
        report_frequency=args.frequency,
        max_steps=args.max_steps,
        model_id=args.model,
        mailbox_path=args.mailbox
    )
    
    print("\nExample completed successfully.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example of using the UserFeedbackAgent")
    
    parser.add_argument("--email", type=str, default=None,
                        help="Email address to communicate with")
    parser.add_argument("--frequency", type=int, default=1,
                        help="How often to send reports (1 = every iteration)")
    parser.add_argument("--max-steps", type=int, default=4,
                        help="Maximum number of steps for the agent")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.0-flash",
                        help="The model ID to use")
    parser.add_argument("--mailbox", type=str, default="/var/mail/rob",
                        help="Path to the mailbox file to check for emails")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 