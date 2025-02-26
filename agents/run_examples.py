#!/usr/bin/env python
"""
Run the agent examples from the agents directory.

Usage:
    poetry run python -m run_examples researcher "your query"     # Run the researcher with a custom query
    poetry run python run_examples.py researcher "query"          # Run the researcher with a custom query
    poetry run python run_examples.py writer "query" --telemetry  # Run with telemetry enabled
"""

import sys

def run_researcher(query=None, telemetry=False):
    from researcher import example as researcher_example
    researcher_example.main(query, telemetry)
    print("\n")

def run_writer_critic():
    from writer_critic import example as writer_critic_example
    writer_critic_example.main()
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Include one arg: researcher, writer")
        sys.exit(1)
    
    choice = sys.argv[1].lower()
    
    # Check if telemetry flag is present
    telemetry = "--telemetry" in sys.argv
    # Remove telemetry flag from args if present to not interfere with other argument parsing
    args = [arg for arg in sys.argv if arg != "--telemetry"]
    
    if choice == "researcher":
        # Check if a query was provided
        query = args[2] if len(args) > 2 else None
        run_researcher(query, telemetry)
    elif choice == "writer":
        run_writer_critic()
    else:
        print(f"Unknown option: {choice}")
        print("Include one arg: researcher, writer")
        sys.exit(1) 