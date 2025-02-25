#!/usr/bin/env python
"""
Run the agent examples from the agents directory.

Usage:
    python run_examples.py scraper      # Run the scraper example
    python run_examples.py writer       # Run the writer-critic example
    python run_examples.py all          # Run both examples
"""

import sys
import importlib

def run_scraper():
    print("=== Running Scraper Example ===")
    from scraper import example as scraper_example
    scraper_example.main()
    print("\n")

def run_writer_critic():
    print("=== Running Writer-Critic Example ===")
    from writer_critic import example as writer_critic_example
    writer_critic_example.main()
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify which example to run: scraper, writer, or all")
        sys.exit(1)
    
    choice = sys.argv[1].lower()
    
    if choice == "scraper":
        run_scraper()
    elif choice == "writer":
        run_writer_critic()
    elif choice == "all":
        run_scraper()
        run_writer_critic()
    else:
        print(f"Unknown option: {choice}")
        print("Please specify: scraper, writer, or all")
        sys.exit(1) 