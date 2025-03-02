#!/usr/bin/env python3
import time
import sys
from typing import List, Optional, Dict

import manager.main as manager_main
from researcher.main import initialize as initialize_researcher
from writer_critic.main import initialize as initialize_writer
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

# Example usage
# From agents directory: poetry run python -m manager.example "Your search query here"
# Or run without arguments to use the default query

def run_example(
    query=None, 
    telemetry=False, 
    max_steps=8, 
    base_wait_time=3.0, 
    max_retries=5,
    custom_agents=None,
    verbose=True
):
    """Run the manager agent example with optional custom configuration
    
    Args:
        query: The query to process
        telemetry: Whether to enable OpenTelemetry tracing
        max_steps: Maximum steps for each agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        custom_agents: List of custom agents to use
        verbose: Whether to print progress information
        
    Returns:
        The result from the manager agent
    """
    # Create example custom agents if not provided
    if custom_agents is None:
        custom_agents = []
    
    # Create and configure the manager agent system
    run_query = manager_main.initialize(
        enable_telemetry=telemetry,
        managed_agents=custom_agents,
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )

    # Use provided query or fall back to default
    if query is None:
        query = """What is the current state of quantum computing as of 2025?"""
    
    # Display configuration information
    if verbose:
        agent_list = [agent.name if hasattr(agent, "name") else "custom_agent" for agent in custom_agents]
        print(f"Manager is configured with: {', '.join(agent_list) if agent_list else 'no agents'}")
    
    # Run the query
    result = run_query(query, verbose=verbose)
    
    return result

def create_researcher_agent(
    max_steps: int = 15,
    base_wait_time: float = 3.0,
    max_retries: int = 5,
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False
):
    """Create a researcher agent
    
    Args:
        max_steps: Maximum steps for the agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        
    Returns:
        The researcher agent
    """
    researcher = initialize_researcher(
        enable_telemetry=enable_telemetry,
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        model_id=model_id
    )
    
    return researcher

def create_writer_agent(
    max_steps: int = 15,
    base_wait_time: float = 3.0,
    max_retries: int = 5,
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False
):
    """Create a writer agent
    
    Args:
        max_steps: Maximum steps for the agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        
    Returns:
        The writer agent
    """
    try:
        from writer_critic.agents import create_writer_agent as create_writer
        from writer_critic.agents import create_critic_agent as create_critic
        
        # Create a model for the agents
        model = RateLimitedLiteLLMModel(
            model_id=model_id,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            enable_telemetry=enable_telemetry
        )
        
        # Create the critic agent first
        critic_agent = create_critic(model=model)
        
        # Create the writer agent that manages the critic
        writer_agent = create_writer(
            model=model,
            critic_agent=critic_agent,
            max_steps=max_steps
        )
        
        return writer_agent
        
    except ImportError:
        print("Warning: Could not create writer agent. Make sure writer_critic module is available.")
        return None


if __name__ == "__main__":
    # Use the main function from manager.main
    from manager.main import main
    main()