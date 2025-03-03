#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
from typing import List, Optional, Callable, Any

from dotenv import load_dotenv
from utils.telemetry import start_telemetry
from manager.agents import create_manager_agent
from researcher.agents import create_researcher_agent
from writer_critic.agents import create_writer_agent, create_critic_agent
from editor.agents import create_editor_agent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def setup_environment():
    """Set up environment variables and API keys"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return api_key

def initialize(
    enable_telemetry: bool = False,
    managed_agents: Optional[List] = None,
    max_steps: int = 8,
    max_retries: int = 3,
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    model_id: str = "gemini/gemini-2.0-flash",
    agent_configs: Optional[dict] = None
) -> Callable[[str], str]:
    """Initialize the manager agent system with optional telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        managed_agents: List of pre-configured agents to manage
        max_steps: Maximum number of steps for the agents
        max_retries: Maximum number of retry attempts for rate limiting
        model_info_path: Path to the model info JSON file
        model_id: The model ID to use (default: gemini/gemini-2.0-flash)
        agent_configs: Dictionary containing agent configurations
        
    Returns:
        A function that can process queries through the manager agent
    """
    
    # Start telemetry if enabled
    if enable_telemetry:
        start_telemetry()
    
    api_key = setup_environment()
    
    # Create a rate-limited model with model-specific rate limits
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        max_retries=max_retries,
        model_info_path=model_info_path
    )
    
    # Initialize the list of agents to manage
    agents_to_manage = managed_agents or []
    
    # Create the manager agent that coordinates the managed agents
    manager_agent = create_manager_agent(
        model=model, 
        managed_agents=agents_to_manage,
        max_steps=max_steps
    )
    
    def run_query(query: str) -> str:
        """Runs a query through the manager agent
        
        Args:
            query: The query to process
            
        Returns:
            The response from the manager agent
        """

        # Run the query
        result = manager_agent.run(query)

        return result
        
    return run_query

def create_agent_by_type(
    agent_type: str,
    max_steps: int = 15,
    model_id: str = "gemini/gemini-2.0-flash",
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    max_retries: int = 3,
    agent_configs: Optional[dict] = None,
):
    """Create an agent by type
    
    Args:
        agent_type: The type of agent to create ('researcher', 'writer', or 'editor')
        max_steps: Maximum steps for the agent
        model_id: The model ID to use
        model_info_path: Path to model info JSON file
        max_retries: Maximum retries for rate limiting
        agent_configs: Dictionary containing agent configurations with keys like:
                      'researcher_description', 'researcher_prompt',
                      'writer_description', 'writer_prompt',
                      'critic_description', 'critic_prompt',
                      'fact_checker_description', 'fact_checker_prompt',
                      'editor_description', 'editor_prompt'
        
    Returns:
        The created agent
    """
    # Create a rate-limited model with model-specific rate limits
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        max_retries=max_retries,
        model_info_path=model_info_path
    )
    
    # Default empty dict if None provided
    agent_configs = agent_configs or {}
    
    if agent_type.lower() == 'researcher':
        return create_researcher_agent(
            model=model,
            max_steps=max_steps,
            researcher_description=agent_configs.get('researcher_description'),
            researcher_prompt=agent_configs.get('researcher_prompt')
        )
    
    elif agent_type.lower() == 'writer':
        # For writer agent, we need to create a critic agent first
        critic_agent = create_critic_agent(
            model=model,
            agent_description=agent_configs.get('critic_description'),
            system_prompt=agent_configs.get('critic_prompt')
        )
        
        return create_writer_agent(
            model=model,
            critic_agent=critic_agent,
            max_steps=max_steps,
            agent_description=agent_configs.get('writer_description'),
            system_prompt=agent_configs.get('writer_prompt')
        )
    
    elif agent_type.lower() == 'editor':
        # The editor agent will create its own fact checker agent internally
        return create_editor_agent(
            model=model,
            max_steps=max_steps,
            agent_description=agent_configs.get('editor_description'),
            system_prompt=agent_configs.get('editor_prompt'),
            # Pass fact checker configurations to be used internally
            fact_checker_description=agent_configs.get('fact_checker_description'),
            fact_checker_prompt=agent_configs.get('fact_checker_prompt')
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def load_agent_configs(config_path=None, config_dict=None):
    """Load agent configurations from a file or dictionary
    
    Args:
        config_path: Path to a JSON file containing agent configurations
        config_dict: Dictionary containing agent configurations
        
    Returns:
        Dictionary with agent configurations
        
    Example config format:
    {
        "researcher_description": "Expert researcher with focus on scientific papers",
        "researcher_prompt": "You are a meticulous researcher who prioritizes academic sources",
        "writer_description": "Creative writer with journalistic style",
        "writer_prompt": "Write engaging content with a focus on clarity and accuracy",
        "critic_description": "Detail-oriented literary critic with high standards",
        "critic_prompt": "Evaluate writing for clarity, accuracy, and engagement",
        "fact_checker_description": "Thorough fact checker with attention to detail",
        "fact_checker_prompt": "Verify claims against reliable sources with precision",
        "editor_description": "Skilled editor with focus on accuracy and clarity",
        "editor_prompt": "Edit content to ensure factual accuracy while maintaining style"
    }
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config from {config_path}: {e}")
            return {}
    
    if config_dict and isinstance(config_dict, dict):
        return config_dict
    
    return {}

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the manager agent system")
    
    # Add command line arguments
    parser.add_argument("--query", nargs="?", help="Query to process")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--managed-agents", type=str, default=None, help="Comma-separated list of agent types to create (e.g. researcher,writer,editor)")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps for agents")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    
    # Add arguments for agent configurations - simplified to a single toggle
    parser.add_argument("--use-custom-prompts", action="store_true", help="Use custom agent descriptions and prompts")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to a JSON file containing agent configurations")
    
    return parser.parse_args()

def main():
    """Main entry point when run directly"""
    args = parse_arguments()
    
    # Load agent configurations
    agent_configs = {}
    
    # First try loading from config file if specified
    if args.config_file:
        agent_configs = load_agent_configs(config_path=args.config_file)
    
    # Create agents based on the managed-agents argument
    agents = []
    if args.managed_agents:
        managed_agents = args.managed_agents.split(",")
        for agent_type in managed_agents:
            agent_type = agent_type.strip()
            
            # Create the agent
            agent = create_agent_by_type(
                agent_type=agent_type,
                model_id=args.model_id,
                max_steps=args.max_steps,
                model_info_path=args.model_info_path,
                agent_configs=agent_configs
            )
            
            if agent:
                agents.append(agent)
    
    # Display configuration information
    agent_list = [agent.name if hasattr(agent, "name") else "custom_agent" for agent in agents]
    print(f"Manager is configured with: {', '.join(agent_list) if agent_list else 'no agents'}")
    
    # Initialize the manager agent with parameters from command line
    run_query = initialize(
        enable_telemetry=args.telemetry,
        managed_agents=agents,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        agent_configs=agent_configs
    )

    return run_query(args.query)

if __name__ == "__main__":
    main()