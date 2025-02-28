import time
import sys
from typing import List, Optional, Dict
import manager.main as manager_main
from researcher.main import initialize as initialize_researcher
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from smolagents import ToolCallingAgent

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
    custom_system_prompts: Optional[Dict[str, str]] = None
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
        custom_system_prompts: Optional system prompts for the custom agents
    """
    # Create example custom agents if not provided
    if custom_agents is None:
        custom_agents = []
    
    # Create example agent descriptions if not provided
    if custom_descriptions is None:
        custom_descriptions = {}
    
    # Create example agent system prompts if not provided
    if custom_system_prompts is None:
        custom_system_prompts = {}
    
    # Create and configure the manager agent system
    run_query = manager_main.initialize(
        enable_telemetry=telemetry,
        managed_agents=custom_agents,
        agent_descriptions=custom_descriptions,
        agent_system_prompts=custom_system_prompts,
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )

    # Use provided query or fall back to default
    if query is None:
        query = """What is the current state of quantum computing as of 2025?"""
    
    # Display configuration information
    agent_list = [agent.name if hasattr(agent, "name") else "custom_agent" for agent in custom_agents]
    print(f"Manager is configured with: {', '.join(agent_list) if agent_list else 'no agents'}")
    
    if custom_descriptions:
        print("\nAgent descriptions:")
        for agent_name, description in custom_descriptions.items():
            print(f"  - {agent_name}: {description[:50]}..." if len(description) > 50 else f"  - {agent_name}: {description}")
    
    if custom_system_prompts:
        print("\nCustom system prompts provided for:")
        for agent_name in custom_system_prompts.keys():
            print(f"  - {agent_name}")
    
    # Time the query execution
    start_time = time.time()
    
    # Process the query and get the result
    print(f"\nProcessing query: {query}")
    result = run_query(query)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal run time: {execution_time:.2f} seconds")
    
    # Print the result
    print("\nResult:")
    print(result)
    
    return result

def create_researcher_agent(
    max_steps: int = 15,
    base_wait_time: float = 3.0,
    max_retries: int = 5,
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False,
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
):
    """Create a researcher agent
    
    Args:
        max_steps: Maximum steps for the agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        agent_description: Optional custom description for the agent
        system_prompt: Optional custom system prompt for the agent
        
    Returns:
        A tuple of (agent, description, system_prompt)
    """
    # Use provided description or default
    researcher_description = agent_description or (
        "Writes python code to perform complex research workflows to perform web and arXiv searches, "
        "scrape webpages, and convert PDFs to markdown. Returns a detailed markdown report. "
        "Provide specific, detailed search query descriptions and instructions to get the most relevant results."
    )
    
    researcher = initialize_researcher(
        enable_telemetry=enable_telemetry,
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        model_id=model_id,
        agent_description=researcher_description,
        system_prompt=system_prompt
    )
    
    return researcher, researcher_description, system_prompt

def create_writer_agent(
    max_steps: int = 15,
    base_wait_time: float = 3.0,
    max_retries: int = 5,
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False,
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
):
    """Create a writer agent
    
    Args:
        max_steps: Maximum steps for the agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        agent_description: Optional custom description for the writer agent
        system_prompt: Optional custom system prompt for the writer agent
        
    Returns:
        A tuple of (agent, descriptions_dict, system_prompts_dict)
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
        
        # Define writer and critic descriptions and system prompts with sci-fi theme as default
        writer_description = agent_description or "An eccentric sci-fi writer tasked with creating stories about humans and AGI-powered robots colonizing the Moon."
        critic_description = "A brutally honest literary critic who analyzes and provides constructive feedback on creative content."
        
        writer_system_prompt = system_prompt or """You are a uniquely talented, often eccentric, esoteric, science fiction writer tasked with creating a riveting story about humans and AGI-powered robots to colonize the Moon. Your writing is vivid, engaging, and scientifically plausible, however, often includes realistically speculative tech enabled by scientific breakthroughs discovered by powerful AGI."""
        
        critic_system_prompt = """You are an insightful, brutally honest literary critic with expertise in science fiction. Your role is to analyze the story's structure, themes, character arcs, and scientific elements. Provide cutting feedback where necessary to improve the narrative's impact."""
        
        # Create the critic agent first
        critic_agent = create_critic(
            model=model,
            agent_description=critic_description,
            system_prompt=critic_system_prompt
        )
        
        # Create the writer agent that manages the critic
        writer_agent = create_writer(
            model=model,
            critic_agent=critic_agent,
            agent_description=writer_description,
            system_prompt=writer_system_prompt
        )
        
        # Set max steps for the agents
        writer_agent.max_steps = max_steps
        critic_agent.max_steps = 1  # Critic just needs one step
        
        # Create dictionaries for descriptions and system prompts
        descriptions = {
            "writer_agent": writer_description,
            "critic_agent": critic_description
        }
        
        system_prompts = {
            "writer_agent": writer_system_prompt,
            "critic_agent": critic_system_prompt
        }
        
        return writer_agent, descriptions, system_prompts
        
    except ImportError:
        print("Warning: Could not create writer agent. Make sure writer_critic module is available.")
        return None, {}, {}

def create_agent_by_type(
    agent_type: str,
    max_steps: int = 15,
    base_wait_time: float = 3.0,
    max_retries: int = 5,
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False,
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
):
    """Create an agent by type
    
    Args:
        agent_type: The type of agent to create ("researcher", "writer")
        max_steps: Maximum steps for the agent
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        agent_description: Optional custom description to use for the agent
        system_prompt: Optional custom system prompt to use for the agent
        
    Returns:
        A tuple of (agent, descriptions_dict, system_prompts_dict)
    """
    if agent_type == "researcher":
        # Get the researcher agent with custom description and system prompt if provided
        agent, description, agent_system_prompt = create_researcher_agent(
            max_steps=max_steps,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            model_id=model_id,
            enable_telemetry=enable_telemetry,
            agent_description=agent_description,
            system_prompt=system_prompt
        )
        
        # Create dictionaries for descriptions and system prompts
        descriptions_dict = {f"{agent_type}_agent": description}
        system_prompts_dict = {} if agent_system_prompt is None else {f"{agent_type}_agent": agent_system_prompt}
        
        return agent, descriptions_dict, system_prompts_dict
    
    elif agent_type == "writer":
        # Get the writer agent with custom description and system prompt if provided
        agent, descriptions, system_prompts = create_writer_agent(
            max_steps=max_steps,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            model_id=model_id,
            enable_telemetry=enable_telemetry,
            agent_description=agent_description,
            system_prompt=system_prompt
        )
        
        # If custom description is provided, update the writer agent description
        if agent_description and agent:
            descriptions["writer_agent"] = agent_description
            
        # If custom system prompt is provided, update the writer agent system prompt
        if system_prompt and agent:
            system_prompts["writer_agent"] = system_prompt
            
        return agent, descriptions, system_prompts
    
    else:
        print(f"Warning: Unknown agent type '{agent_type}'")
        return None, {}, {}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the manager agent example")
    
    # Add command line arguments
    parser.add_argument("query", nargs="?", default=None, help="Query to process")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--agent-types", nargs="+", type=str, default=[], 
                       help="List of agent types to create (e.g. researcher writer)")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps for agents")
    parser.add_argument("--researcher-description", type=str, default=None, 
                       help="Custom description for the researcher agent")
    parser.add_argument("--writer-description", type=str, default=None, 
                       help="Custom description for the writer agent")
    parser.add_argument("--researcher-prompt", type=str, default=None, 
                       help="Custom system prompt for the researcher agent")
    parser.add_argument("--writer-prompt", type=str, default=None, 
                       help="Custom system prompt for the writer agent")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Initialize lists and dictionaries for agents and descriptions
    agents = []
    descriptions = {}
    system_prompts = {}
    
    # Use default agent types if none specified
    agent_types = args.agent_types
    if not agent_types:
        agent_types = ["researcher"]  # Default to researcher agent if none specified
    
    # Create agents based on specified types
    for agent_type in agent_types:
        # Get custom description and system prompt for this agent type
        agent_description = None
        agent_system_prompt = None
        
        if agent_type == "researcher":
            agent_description = args.researcher_description
            agent_system_prompt = args.researcher_prompt
        elif agent_type == "writer":
            agent_description = args.writer_description
            agent_system_prompt = args.writer_prompt
        
        # Create the agent with custom description and system prompt if provided
        agent, agent_descriptions, agent_system_prompts = create_agent_by_type(
            agent_type=agent_type,
            max_steps=args.max_steps,
            enable_telemetry=args.telemetry,
            agent_description=agent_description,
            system_prompt=agent_system_prompt
        )
        
        if agent:
            agents.append(agent)
            descriptions.update(agent_descriptions)
            system_prompts.update(agent_system_prompts)
    
    # Run the example with the specified configuration
    main(
        query=args.query,
        telemetry=args.telemetry,
        max_steps=args.max_steps,
        custom_agents=agents,
        custom_descriptions=descriptions,
        custom_system_prompts=system_prompts
    )