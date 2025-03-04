from typing import List, Optional
from smolagents import CodeAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from utils.agents.tools import apply_custom_agent_prompts

def create_manager_agent(
    model: RateLimitedLiteLLMModel, 
    managed_agents: List,
    max_steps: int = 8
) -> CodeAgent:
    """Creates a manager agent that can coordinate multiple agents
    
    Args:
        model: The RateLimitedLiteLLMModel model to use for the agent
        managed_agents: List of agents that the manager will coordinate
        max_steps: Maximum steps the manager agent can take
        
    Returns:
        A configured manager agent
    """
    
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
    
    # Define the custom system prompt
    custom_system_prompt = f"""You are a manager agent in charge of coordinating a team of specialized agents. You don't have access to tools to call yourself, but instead you can call and manage the following agents: {available_agents_text}.

When given a task, you should:
1. Make a detailed plan for how to complete the task
2. Break down complex tasks into smaller subtasks
3. Determine which agent is best suited for each subtask
4. Call the appropriate agent(s) with clear, specific instructions
5. Review the results from each agent for quality and relevance
6. Refine your plan and approach if needed, based on the results of the agents
7. Repeat steps 4-6 until all aspects of the task are 100% complete
8. Synthesize the final results into a complete, cohesive response

Be persistent and iterative in your approach. If an agent's results aren't satisfactory, refine your instructions and try again. Only when you are sure all aspects of the original task have been thoroughly addressed should you provide your final response.

Always maintain a clean, organized format in your responses, including citations and sources where appropriate unless instructed otherwise.

IMPORTANT: To call an agent, use Python code like this:
```python
# Example of calling the researcher_agent
research_results = researcher_agent(task="Find information about X")
print(research_results)

# Example of calling the writer_agent
writing_results = writer_agent(task="Write a report about X based on this information: " + research_results)
print(writing_results)

# Example of calling the editor_agent
final_report = editor_agent(task="Edit this report for clarity and accuracy: " + writing_results)
print(final_report)

# When you're done, use the final_answer tool
final_answer(final_report)
```"""

    # Apply custom templates with the custom system prompt
    apply_custom_agent_prompts(agent, custom_system_prompt)
    
    return agent
