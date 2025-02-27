import time
import sys
from typing import List, Optional, Dict
import manager.main as manager_main
from researcher.agents import create_researcher_agent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

# Example usage
# From agents directory: python -m manager.example "Your search query here"
# Or run without arguments to use the default query

def main(
    query=None, 
    telemetry=False, 
    max_steps=8, 
    base_wait_time=3.0, 
    max_retries=5,
    custom_agents: Optional[List] = None,
    custom_descriptions: Optional[Dict[str, str]] = None,
    agent_types: Optional[List[str]] = None,
    create_researcher: bool = True,
    create_writer: bool = False
):
    """Run the manager agent example with optional custom configuration
    
    Args:
        query: The query to process
        telemetry: Whether to enable OpenTelemetry tracing
        max_steps: Maximum steps for each agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        custom_agents: Optional list of custom agents to add to the manager
        custom_descriptions: Optional descriptions for the custom agents
        agent_types: Optional list of agent types to create ("researcher", "writer", etc.)
        create_researcher: Whether to automatically create a researcher agent
        create_writer: Whether to automatically create a writer agent
    """
    # Create example custom agents if not provided
    if custom_agents is None:
        # This example doesn't create custom agents by default
        custom_agents = []
    
    # Create example agent descriptions if not provided
    if custom_descriptions is None:
        custom_descriptions = {}
        
    # Create and configure the manager agent system
    run_query = manager_main.initialize(
        enable_telemetry=telemetry,
        managed_agents=custom_agents,
        agent_descriptions=custom_descriptions,
        create_researcher=create_researcher,
        create_writer=create_writer,
        create_custom_agents=agent_types,
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )

    # Use provided query or fall back to default
    if query is None:
        query = """What is the current state of quantum computing as of 2025?"""
    
    print(f"\nProcessing query: {query}")
    
    # Display configuration information
    agent_list = []
    if create_researcher:
        agent_list.append("researcher_agent")
    if create_writer:
        agent_list.append("writer_agent")
    if agent_types:
        for agent_type in agent_types:
            if agent_type not in ["researcher", "writer"] or \
               (agent_type == "researcher" and not create_researcher) or \
               (agent_type == "writer" and not create_writer):
                agent_list.append(f"{agent_type}_agent")
    if custom_agents:
        agent_list.extend([agent.name if hasattr(agent, "name") else "custom_agent" for agent in custom_agents])
    
    print(f"Manager is configured with: {', '.join(agent_list) if agent_list else 'no agents'}")
    
    # Time the query execution
    start_time = time.time()
    
    # Run the query
    result = run_query(query)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("\nResult:")
    print(result)

def create_example_custom_setup():
    """Create an example with multiple custom agents
    
    This is an example of how to create a more complex setup with
    multiple custom agents. Not used in the default implementation.
    """
    # Create a model for the agents
    model = RateLimitedLiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        base_wait_time=3.0,
        max_retries=5
    )
    
    # Create a researcher agent as a custom agent
    researcher = create_researcher_agent(model, max_steps=15)
    
    # Define custom descriptions for the agents
    descriptions = {
        "researcher_agent": (
            "Performs web searches and information gathering. "
            "Provide specific search queries to get the most relevant results."
        ),
        # Add descriptions for other custom agents here
    }
    
    # Return the custom configuration
    return [researcher], descriptions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the manager agent example")
    
    # Add command line arguments
    parser.add_argument("query", nargs="?", default=None, help="Query to process")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--advanced", action="store_true", help="Use advanced setup with pre-configured agents")
    parser.add_argument("--no-researcher", action="store_true", help="Don't create researcher agent")
    parser.add_argument("--writer", action="store_true", help="Create writer agent")
    parser.add_argument("--agents", nargs="+", type=str, 
                       help="List of agent types to create (e.g. researcher writer)")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps for agents")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    if args.advanced:
        # Create custom agents and descriptions from the example setup
        custom_agents, custom_descriptions = create_example_custom_setup()
        main(args.query, 
             telemetry=args.telemetry,
             custom_agents=custom_agents, 
             custom_descriptions=custom_descriptions,
             max_steps=args.max_steps)
    else:
        # Use configuration based on command line arguments
        main(args.query,
             telemetry=args.telemetry,
             create_researcher=not args.no_researcher,
             create_writer=args.writer,
             agent_types=args.agents,
             max_steps=args.max_steps)