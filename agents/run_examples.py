#!/usr/bin/env python
"""
Run the agent examples from the agents directory.

Usage:
    poetry run python -m run_examples researcher "your query"     # Run the researcher with a custom query
    poetry run python run_examples.py manager "query"             # Run the manager with a custom query
    poetry run python run_examples.py manager-advanced "query"    # Run the manager with custom agents
    poetry run python run_examples.py writer "query" --telemetry  # Run with telemetry enabled
    poetry run python run_examples.py all                         # Run all examples
"""

import sys

def run_researcher(query=None, telemetry=False):
    from researcher import example as researcher_example
    researcher_example.main(query, telemetry)
    print("\n")

def run_manager(query=None, telemetry=False, advanced=False, agent_config=None):
    from manager import example as manager_example
    
    if advanced:
        # Create and use a more complex setup with custom agents
        custom_agents, custom_descriptions = manager_example.create_example_custom_setup()
        manager_example.main(
            query, 
            telemetry=telemetry, 
            custom_agents=custom_agents, 
            custom_descriptions=custom_descriptions
        )
    elif agent_config:
        # Use a specified agent configuration
        create_researcher = 'researcher' in agent_config
        create_writer = 'writer' in agent_config
        
        # Extract other agent types
        agent_types = [agent for agent in agent_config 
                      if agent not in ['researcher', 'writer']]
        
        manager_example.main(
            query,
            telemetry=telemetry,
            create_researcher=create_researcher,
            create_writer=create_writer,
            agent_types=agent_types if agent_types else None
        )
    else:
        # Use the default simple configuration with just researcher
        manager_example.main(query, telemetry=telemetry)
    
    print("\n")

def run_writer_critic():
    from writer_critic import example as writer_critic_example
    writer_critic_example.main()
    print("\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run agent examples")
    parser.add_argument("example", choices=["researcher", "manager", "manager-advanced", 
                                            "manager-custom", "writer", "all"],
                      help="Which example to run")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--agents", nargs="+", help="Agent types to use with manager-custom")
    
    args = parser.parse_args()
    
    if args.example == "researcher":
        run_researcher(args.query, args.telemetry)
    elif args.example == "manager":
        run_manager(args.query, args.telemetry)
    elif args.example == "manager-advanced":
        run_manager(args.query, args.telemetry, advanced=True)
    elif args.example == "manager-custom":
        if not args.agents:
            print("Error: --agents argument is required for manager-custom")
            print("Example: --agents researcher writer")
            sys.exit(1)
        run_manager(args.query, args.telemetry, agent_config=args.agents)
    elif args.example == "writer":
        run_writer_critic()
    elif args.example == "all":
        run_researcher()
        run_manager()
        run_writer_critic()
    else:
        print(f"Unknown option: {args.example}")
        sys.exit(1) 