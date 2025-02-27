from typing import List, Dict, Any, Optional
from smolagents import CodeAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def create_manager_agent(
    model: RateLimitedLiteLLMModel, 
    managed_agents: List,
    agent_descriptions: Optional[Dict[str, str]] = None,
    max_steps: int = 8
) -> CodeAgent:
    """Creates a manager agent that can coordinate multiple agents
    
    Args:
        model: The RateLimitedLiteLLMModel model to use for the agent
        managed_agents: List of agents that the manager will coordinate
        agent_descriptions: Optional dictionary mapping agent names to descriptions 
                          of how to use them effectively
        max_steps: Maximum steps the manager agent can take
        
    Returns:
        A configured manager agent
    """
    
    # Create agent descriptions text if provided
    agent_descriptions_text = ""
    if agent_descriptions:
        agent_descriptions_text = "\n\nAvailable agents and how to use them:\n"
        for agent_name, description in agent_descriptions.items():
            agent_descriptions_text += f"- {agent_name}: {description}\n"
    
    # Create a list of available agent names
    available_agents = [agent.name for agent in managed_agents]
    available_agents_text = ", ".join(available_agents)
            
    agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=managed_agents,
        additional_authorized_imports=["time", "json", "re"],
        name='manager_agent',
        description=f"""This is a manager agent that coordinates multiple specialized agents ({available_agents_text}). It can receive high-level objectives and determine which agents to deploy for different subtasks, how to sequence their use, and how to synthesize their outputs.""",
        max_steps=max_steps
    )

    agent.prompt_templates["system_prompt"] += f"""\n\nYou are a manager agent that coordinates multiple specialized agents. You have access to the following agents: {available_agents_text}.{agent_descriptions_text}

When given a task, you should:
1. Break down complex tasks into smaller subtasks
2. Determine which agent is best suited for each subtask
3. Call the appropriate agent(s) with clear, specific instructions
4. Review the results from each agent for quality and relevance
5. Refine your approach if needed and call agents again with improved instructions
6. Synthesize the results into a cohesive response

For example, to use an agent named 'researcher_agent', you would call it like:
`researcher_agent("your specific research task")`

Be persistent and iterative in your approach. If an agent's results aren't satisfactory, refine your instructions and try again. Only when you are sure all aspects of the original task have been thoroughly addressed should you provide your final response.

Always maintain a clean, organized format in your responses, including citations and sources where appropriate unless instructed otherwise."""
    
    return agent
