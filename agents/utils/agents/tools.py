import os
import yaml
from typing import Dict, Any, Tuple

def load_agent_template(template_path: str) -> Dict[str, Any]:
    """Load a single agent template from a YAML file.
    
    Args:
        template_path: Path to the template YAML file
        
    Returns:
        The loaded template as a dictionary
        
    Raises:
        FileNotFoundError: If the template file is not found
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    try:
        with open(template_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def load_agent_template_by_type(agent_type: str, base_dir: str = 'utils/agents') -> Dict[str, Any]:
    """Load an agent template based on the agent type.
    
    Args:
        agent_type: Type of agent ('code' or 'toolcalling')
        base_dir: Base directory for template files
        
    Returns:
        The loaded template as a dictionary
        
    Raises:
        ValueError: If the agent type is invalid
        FileNotFoundError: If the template file is not found
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    if agent_type.lower() == 'code':
        template_path = os.path.join(base_dir, 'code_agent.yaml')
    elif agent_type.lower() == 'toolcalling':
        template_path = os.path.join(base_dir, 'toolcalling_agent.yaml')
    else:
        raise ValueError(f"Invalid agent type: {agent_type}. Must be 'code' or 'toolcalling'")
    
    return load_agent_template(template_path)

def apply_custom_agent_prompts(agent) -> None:
    """Apply a template to an agent instance, completely replacing the default templates.
    The agent type is automatically determined from the agent's class.
    
    Args:
        agent: The agent instance
        
    Raises:
        ValueError: If the agent type cannot be determined
        FileNotFoundError: If the template file is not found
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    # Determine agent type from the agent's class
    agent_class_name = agent.__class__.__name__
    
    if 'CodeAgent' in agent_class_name:
        agent_type = 'code'
    elif 'ToolCallingAgent' in agent_class_name:
        agent_type = 'toolcalling'
    else:
        raise ValueError(f"Cannot determine agent type from class: {agent_class_name}")
    
    # Load the appropriate template
    template = load_agent_template_by_type(agent_type)
    
    # Apply the template to the agent's prompt_templates
    agent.prompt_templates = template
    
    # Reinitialize the system prompt to apply the new template
    agent.system_prompt = agent.initialize_system_prompt()
