#!/usr/bin/env python3
import time
import sys
import argparse
from typing import List, Optional, Dict, Tuple, Any

import manager.main as manager_main
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

# Example usage
# From agents directory: poetry run python -m manager.example --query "Your search query here" --managed-agents researcher,writer,editor --use-custom-prompts

def get_default_agent_configs():
    """Get default agent configurations
    
    Returns:
        Dictionary with default agent configurations
    """
    return {
        "researcher_description": "Expert researcher with focus on scientific papers and academic sources",
        "researcher_prompt": "You are a meticulous researcher who prioritizes academic sources and provides comprehensive information with proper citations.",
        
        "writer_description": "Creative writer with journalistic style and clear explanations",
        "writer_prompt": "Write engaging content with a focus on clarity, accuracy, and reader engagement. Use a journalistic style that makes complex topics accessible.",
        
        "critic_description": "Detail-oriented editor with high standards for clarity and accuracy",
        "critic_prompt": "Evaluate writing for clarity, accuracy, engagement, and logical flow. Provide constructive feedback that improves the content without changing its voice.",
        
        "editor_description": "Skilled editor with focus on accuracy, clarity, and factual correctness",
        "editor_prompt": "Edit content to ensure factual accuracy while maintaining style and readability. Focus on improving clarity without changing the author's voice.",
        
        "fact_checker_description": "Thorough fact checker with attention to detail and source verification",
        "fact_checker_prompt": "Verify claims against reliable sources with precision. Identify potential inaccuracies and suggest corrections based on authoritative references."
    }

def main(
    query=None, 
    telemetry=False, 
    managed_agents=None, 
    use_custom_prompts=False,
    max_steps=8,
    max_retries=3,
    model_id="gemini/gemini-2.0-flash",
    model_info_path="utils/gemini/gem_llm_info.json"
):
    """Main entry point for the manager example
    
    Args:
        query: The query to process
        telemetry: Whether to enable OpenTelemetry tracing
        managed_agents: List of agent types to create (comma-separated string)
        use_custom_prompts: Whether to use custom agent descriptions and prompts
        max_steps: Maximum steps for each agent
        max_retries: Maximum retries for rate limiting
        model_id: The model ID to use
        model_info_path: Path to model info JSON file
        
    Returns:
        The result from the manager agent
    """
    
    # Parse managed agents string into a list
    agent_types = []
    if managed_agents:
        agent_types = [agent_type.strip() for agent_type in managed_agents.split(",")]
    
    # Load agent configurations if using custom prompts
    agent_configs = {}
    if use_custom_prompts:
        agent_configs = get_default_agent_configs()
    
    # Create agents based on the agent types
    agents = []
    for agent_type in agent_types:
        agent = manager_main.create_agent_by_type(
            agent_type=agent_type,
            max_steps=max_steps,
            model_id=model_id,
            model_info_path=model_info_path,
            max_retries=max_retries,
            agent_configs=agent_configs
        )
        if agent:
            agents.append(agent)
    
    # Display configuration information
    agent_list = [agent.name if hasattr(agent, "name") else "custom_agent" for agent in agents]
    print(f"Manager is configured with: {', '.join(agent_list) if agent_list else 'no agents'}")
    
    # Initialize the manager agent system
    run_query = manager_main.initialize(
        enable_telemetry=telemetry,
        managed_agents=agents,
        max_steps=max_steps,
        max_retries=max_retries,
        model_id=model_id,
        model_info_path=model_info_path,
        agent_configs=agent_configs
    )
    
    # Run the query
    if query:
        start_time = time.time()
        result = run_query(query)
        execution_time = time.time() - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        return result
    else:
        print("No query provided.")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the manager example")
    parser.add_argument("--query", type=str, required=True, help="Query to process")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--managed-agents", type=str, required=True, 
                        help="Comma-separated list of agent types to create (e.g. researcher,writer,editor)")
    parser.add_argument("--use-custom-prompts", action="store_true", 
                        help="Use custom agent descriptions and prompts")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps for agents")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", 
                        help="Path to model info JSON file")
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(
        query=args.query,
        telemetry=args.telemetry,
        managed_agents=args.managed_agents,
        use_custom_prompts=args.use_custom_prompts,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path
    )
