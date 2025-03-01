import os
import yaml
from typing import Optional, Dict, Any, Literal

def get_agent_prompt_templates(agent: Any) -> Dict[str, Any]:
    """Load prompt templates specific to an agent type from the corresponding YAML file
    
    Args:
        agent_type: The type of agent ("code" or "toolcalling")
        
    Returns:
        A dictionary containing the agent-specific prompt templates
    """

    # Determine agent type from the agent's class name
    agent_class_name = agent.__class__.__name__
    
    if "CodeAgent" in agent_class_name:
        agent_type = "code"
    elif "ToolCallingAgent" in agent_class_name:
        agent_type = "toolcalling"
    else:
        print(f"Unknown agent type: {agent_class_name}. Cannot apply templates.")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
    
    if agent_type == "code":
        yaml_path = os.path.join(current_dir, "prompts_code_agent.yaml")
    elif agent_type == "toolcalling":
        yaml_path = os.path.join(current_dir, "prompts_toolcalling_agent.yaml")
    else:
        print(f"Unknown agent type: {agent_type}")
        return {}
    
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                yaml_prompts = yaml.safe_load(f)
                return yaml_prompts
        except Exception as e:
            print(f"Error loading agent-specific prompt templates from YAML: {e}")
    else:
        print(f"Agent-specific prompt template file not found: {yaml_path}")
    return {}

def apply_agent_specific_templates(agent) -> None:
    """Apply agent-specific prompt templates to an agent
    
    This function automatically determines the agent type based on the agent's class name
    and applies the appropriate templates.
    
    Args:
        agent: The agent instance whose prompt templates should be updated
        
    Returns:
        None - modifies the agent in place
    """
    
    templates = get_agent_prompt_templates(agent)
    if templates and hasattr(agent, 'prompt_templates'):
        # Update the agent's prompt templates with the agent-specific ones
        for key, value in templates.items():
            if key in agent.prompt_templates:
                # If it's a nested dictionary, update each nested key
                if isinstance(value, dict) and isinstance(agent.prompt_templates[key], dict):
                    for nested_key, nested_value in value.items():
                        if nested_key in agent.prompt_templates[key]:
                            agent.prompt_templates[key][nested_key] = nested_value
                else:
                    # Otherwise just update the top-level key
                    agent.prompt_templates[key] = value
        
        # Reinitialize the system prompt with the new templates
        if hasattr(agent, 'initialize_system_prompt'):
            agent.system_prompt = agent.initialize_system_prompt()
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'system_prompt'):
                agent.memory.system_prompt.system_prompt = agent.system_prompt


