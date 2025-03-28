#!/usr/bin/env python3
"""
Example usage of the UserFeedbackAgent class.

This example shows how to create and use a UserFeedbackAgent instance directly.
It also provides a command-line interface for testing email communication.

The agent uses the dedicated 'fb_agent' system user to handle email communication.
To use this functionality, ensure that:
1. The fb_agent user exists on the system
2. The main application user has read (and ideally write) access to /var/mail/fb_agent
3. The proper mail forwarding is set up for fb_agent

Email Configuration:
- LOCAL_USER_EMAIL: Used for the fb_agent's local mailbox configuration
- REMOTE_USER_EMAIL: The external user's email for sending reports and receiving commands

Usage:
    python -m agents.user_feedback.example [--email your-email@example.com]
"""

import os
import argparse
import grp
import pwd
from dotenv import load_dotenv
from agents.utils.telemetry import suppress_litellm_logs
from agents.user_feedback.agents import UserFeedbackAgent
from agents.user_feedback.tools import FB_AGENT_USER

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
    
    # Check for email environment variables
    local_email = os.getenv("LOCAL_USER_EMAIL")
    remote_email = os.getenv("REMOTE_USER_EMAIL")
    if not local_email:
        print("Warning: LOCAL_USER_EMAIL environment variable is not set. This may affect mailbox functionality.")
    if not remote_email:
        print("Warning: REMOTE_USER_EMAIL environment variable is not set. Email sending and checking will not work.")
        print("  - Set REMOTE_USER_EMAIL to the external user's email address")
        print("  - Or use the --email command line parameter")
    
    print("\n=== Email Configuration ===")
    print(f"LOCAL_USER_EMAIL: {local_email or 'Not set'} (used for mailbox)")
    print(f"REMOTE_USER_EMAIL: {remote_email or 'Not set'} (used for sending/receiving)")
    
    # Check for fb_agent user
    try:
        fb_agent_info = pwd.getpwnam(FB_AGENT_USER)
        print(f"\n✓ {FB_AGENT_USER} user exists (UID: {fb_agent_info.pw_uid})")
    except KeyError:
        print(f"\n✗ ERROR: {FB_AGENT_USER} user does not exist.")
        print("  Please follow the setup instructions in INSTALL.md")
        return
    
    # Check fb_agent mailbox
    fb_maildir = os.path.join("/home/fb_agent/var/mail")
    if os.path.exists(fb_maildir):
        print(f"✓ {FB_AGENT_USER} maildir exists at {fb_maildir}")
        
        # Check Maildir structure
        new_dir = os.path.join(fb_maildir, "new")
        cur_dir = os.path.join(fb_maildir, "cur")
        tmp_dir = os.path.join(fb_maildir, "tmp")
        
        if os.path.exists(new_dir) and os.path.isdir(new_dir):
            print(f"✓ Maildir 'new' directory exists")
            read_access = os.access(new_dir, os.R_OK)
            write_access = os.access(new_dir, os.W_OK)
            
            if read_access:
                print(f"✓ Current user has read access to {new_dir}")
            else:
                print(f"✗ ERROR: Current user doesn't have read access to {new_dir}")
                print("  Please add the current user to the mail group and ensure proper permissions:")
                print(f"  sudo usermod -a -G mail $(whoami)")
                print("  You'll need to log out and back in for this to take effect.")
            
            if write_access:
                print(f"✓ Current user has write access to {new_dir}")
            else:
                print(f"! WARNING: Current user doesn't have write access to {new_dir}")
                print("  Emails can be read but not marked as read.")
                print("  For full functionality, adjust permissions:")
                print(f"  sudo chmod -R g+w {fb_maildir}")
        else:
            print(f"✗ ERROR: Maildir structure not complete. Missing 'new' directory")
            print(f"  Please create the required Maildir structure:")
            print(f"  sudo mkdir -p {fb_maildir}/new {fb_maildir}/cur {fb_maildir}/tmp")
            print(f"  sudo chown -R {FB_AGENT_USER}:mail {fb_maildir}")
            print(f"  sudo chmod -R 750 {fb_maildir}")
    else:
        print(f"✗ ERROR: No maildir found at {fb_maildir}")
        print("  Please create the maildir and set proper permissions:")
        print(f"  sudo mkdir -p {fb_maildir}/new {fb_maildir}/cur {fb_maildir}/tmp")
        print(f"  sudo chown -R {FB_AGENT_USER}:mail {fb_maildir}")
        print(f"  sudo chmod -R 750 {fb_maildir}")
        
    # Check if current user is in the mail group
    try:
        current_user = os.getlogin()
        mail_group = grp.getgrnam('mail')
        if current_user in mail_group.gr_mem:
            print(f"✓ Current user ({current_user}) is in the mail group")
        else:
            print(f"! WARNING: Current user ({current_user}) is not in the mail group")
            print("  Consider adding the user to the mail group for better permission handling:")
            print(f"  sudo usermod -a -G mail {current_user}")
    except Exception as e:
        print(f"! Could not check mail group membership: {e}")

def run_example(user_email=None, max_steps=4, model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                report_frequency=1,
                agent_description=None, agent_prompt=None):
    """Run a test of the UserFeedbackAgent
    
    Args:
        user_email: Email address to override REMOTE_USER_EMAIL env var
        max_steps: Maximum number of steps for the agent
        report_frequency: How often to send reports (1 = every iteration)
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
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        agent_description=agent_description,
        agent_prompt=agent_prompt
    )
    
    print(f"\nCreated UserFeedbackAgent:")
    print(f"- External email (sending to): {agent.remote_email or 'Not configured'}")
    print(f"- Local mailbox: {agent.local_email or 'Not configured'}")
    print(f"- Report frequency: Every {agent.report_frequency} iterations")
    print(f"- Feedback agent user: {agent.feedback_agent_user}")
    
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
    if agent.remote_email:
        report = agent.generate_report(updated_state)
        print(f"\nSample report (sent to {agent.remote_email}):\n{report}")
    else:
        print("\nCannot generate report: No external email address configured.")
        print("Please set REMOTE_USER_EMAIL environment variable or use --email parameter.")
    
    return agent

def main():
    """Main entry point for the example."""
    args = parse_arguments()
    
    print("=== UserFeedbackAgent Example ===")
    agent = run_example(
        user_email=args.email,
        report_frequency=args.frequency,
        max_steps=args.max_steps,
        model_id=args.model
    )
    
    print("\nExample completed successfully.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example of using the UserFeedbackAgent")
    
    parser.add_argument("--email", type=str, help="Email address to communicate with (overrides REMOTE_USER_EMAIL)")
    parser.add_argument("--frequency", type=int, default=1, help="How often to send reports (1 = every iteration)")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum number of steps for the agent")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.0-flash", help="The model ID to use")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 