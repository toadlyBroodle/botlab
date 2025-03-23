#!/usr/bin/env python3
"""
Example usage of the ManagerAgent class.

This example shows how to create and use a ManagerAgent instance directly.
It also provides a command-line interface for running management queries.

Usage:
    python -m agents.manager.example --query "Your management query here"

Note:
    When running this example directly, the manager agent will save its output to
    the manager/data/ directory with a human-readable date prefix (YYYY-MM-DD_HH-MM).
    You can access these files using the load_file function:
    
    ```python
    from agents.utils.agents.tools import load_file
    content = load_file(agent_type="manager")
    ```
"""

import os
import argparse
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from agents.utils.telemetry import suppress_litellm_logs

from agents.manager.manager_agent import ManagerAgent
from agents.researcher.agents import ResearcherAgent
from agents.writer_critic.agents import WriterAgent
from agents.editor.editor_agent import EditorAgent
from agents.qaqc.qaqc_agent import QAQCAgent
from agents.utils.agents.tools import load_file
from agents.utils.file_manager.file_manager import AGENT_DIRS

# Path to project directory for all agents to operate on
PROJECT_DIR = os.path.join(os.getcwd(), "agent_project")

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

def run_example(query=None, max_steps=15, managed_agents=None, custom_configs=None,
                model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3):
    """Run a management query using the ManagerAgent class
    
    Args:
        query: The management query to run
        max_steps: Maximum number of steps for the agent
        managed_agents: List of managed agents
        custom_configs: Dictionary containing custom configurations
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        use_custom_prompts: Whether to use custom prompts
        
    Returns:
        The result from the agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Print the current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Print the location of the manager data directory
    print(f"Manager data directory: {AGENT_DIRS['manager_agent']}")
    print(f"Manager data directory exists: {os.path.exists(AGENT_DIRS['manager_agent'])}")
    
    # Create a shared model for all agents
    shared_model = RateLimitedLiteLLMModel(
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
    )
    
    # Default agent types if none provided
    if managed_agents is None:
        managed_agents = ["researcher", "writer", "editor", "qaqc"]
    
    # Default empty dict if None provided
    custom_configs = custom_configs or {}
    
    print(f"Creating agents: {', '.join(managed_agents)}")
    
    # Create the managed agents
    managed_agents_instances = []
    for agent_type in managed_agents:
        try:
            if agent_type.lower() == 'researcher':
                researcher = ResearcherAgent(
                    model=shared_model,  # Use the shared model
                    max_steps=max_steps,
                    researcher_description=custom_configs.get('researcher_description'),
                    researcher_prompt=custom_configs.get('researcher_prompt')
                )
                managed_agents_instances.append(researcher.agent)
                print(f"Created researcher agent: {researcher.agent.name}")
            
            elif agent_type.lower() == 'writer':
                # Create the writer agent with its own critic
                writer = WriterAgent(
                    model=shared_model,  # Use the shared model
                    max_steps=max_steps,
                    agent_description=custom_configs.get('writer_description'),
                    system_prompt=custom_configs.get('writer_prompt'),
                    critic_description=custom_configs.get('critic_description'),
                    critic_prompt=custom_configs.get('critic_prompt')
                )
                managed_agents_instances.append(writer.agent)
                print(f"Created writer agent: {writer.agent.name}")
            
            elif agent_type.lower() == 'editor':
                # Create the editor agent with its own fact checker
                editor = EditorAgent(
                    model=shared_model,  # Use the shared model
                    max_steps=max_steps,
                    agent_description=custom_configs.get('editor_description'),
                    system_prompt=custom_configs.get('editor_prompt'),
                    fact_checker_description=custom_configs.get('fact_checker_description'),
                    fact_checker_prompt=custom_configs.get('fact_checker_prompt')
                )
                managed_agents_instances.append(editor.agent)
                print(f"Created editor agent: {editor.agent.name}")
            
            elif agent_type.lower() == 'qaqc':
                # Create the QAQC agent
                qaqc = QAQCAgent(
                    model=shared_model,  # Use the shared model
                    max_steps=max_steps,
                    agent_description=custom_configs.get('qaqc_description'),
                    system_prompt=custom_configs.get('qaqc_prompt')
                )
                managed_agents_instances.append(qaqc.agent)
                print(f"Created QAQC agent: {qaqc.agent.name}")
            
            else:
                print(f"Unknown agent type: {agent_type}")
                
        except Exception as e:
            print(f"Error creating {agent_type} agent: {str(e)}")
    
    print(f"Creating manager agent with {len(managed_agents_instances)} managed agents")
    
    # Create the manager agent with the shared model
    manager = ManagerAgent(
        managed_agents=managed_agents_instances,
        model=shared_model,  # Use the shared model
        max_steps=max_steps
    )
    
    # Use default query if none provided
    if query is None:
        query = "Research and write a comprehensive report on the latest advancements in quantum computing, focusing on practical applications."
    
    print(f"Running management query: {query}")
    print("=" * 80)
    
    # Run the query and get the result
    result = manager.run_query(query)
    
    print("=" * 80)
    print("Management task complete! The report has been saved to the reports directory.")
    
    return result

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the ManagerAgent with a query.")
    parser.add_argument("--query", type=str, default="Write brief report on recent (2025) advances in AI agents", help="The query to process")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--agent-types", type=str, default="researcher,writer,editor,qaqc", help="Comma-separated list of agent types to create")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse agent types from comma-separated string
    agent_types = [t.strip() for t in args.agent_types.split(",") if t.strip()]
    
    run_example(
        query=args.query,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        managed_agents=agent_types
    )
