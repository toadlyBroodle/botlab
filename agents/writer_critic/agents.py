from smolagents import (
    ToolCallingAgent,
    CodeAgent
)
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from writer_critic.tools import save_draft
from utils.agents.tools import apply_custom_agent_prompts
from typing import Optional

def create_critic_agent(model: RateLimitedLiteLLMModel, 
                        agent_description: Optional[str] = None, 
                        system_prompt: Optional[str] = None, 
                        max_steps: int = 1) -> ToolCallingAgent:
    """Creates a critic agent that reviews and provides feedback on creative content
    
    Args:
        model: The LiteLLM model to use for the agent
        agent_description: Optional additional description to append to the base description
        system_prompt: Optional custom system prompt to use instead of the default
        
    Returns:
        A configured critic agent
    """

    # Append additional description if provided
    base_description = 'A literary critic who analyzes and provides constructive feedback on creative content.'
    if agent_description:
        description = f"{base_description} {agent_description}"
    else:
        description = base_description
    
    agent = ToolCallingAgent(
        tools=[],  # Critic doesn't need tools - it just provides feedback
        model=model,
        name='critic_agent',
        description=description,
        max_steps=max_steps,  # Critic just needs one step to analyze and respond
    )

    # Apply custom templates to initialize all prompt templates
    apply_custom_agent_prompts(agent)
    
    # Get the base system prompt and append to it
    base_sys_prompt = agent.prompt_templates["system_prompt"]
    
    # Append provided system prompt or default to a generic one
    if system_prompt:
        sys_prompt_appended = base_sys_prompt + f"\n\n{system_prompt}"
    else:
        sys_prompt_appended = base_sys_prompt + """\n\nYou are a literary critic who analyzes and provides constructive feedback on creative content. 
Your role is to provide constructive feedback to your managing writer agent on the content, style, structure, themes, and overall quality of their latest draft.

Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements. Do not make any changes to the draft yourself.
Provide your feedback as plain text, without any special tags.
"""

    agent.prompt_templates["system_prompt"] = sys_prompt_appended
    return agent

def create_writer_agent(model: RateLimitedLiteLLMModel, critic_agent: ToolCallingAgent, 
                        agent_description: Optional[str] = None, 
                        system_prompt: Optional[str] = None, 
                        max_steps: int = 5) -> CodeAgent:
    """Creates a writer agent that drafts creative content and manages the critic
    
    Args:
        model: The LiteLLM model to use for the agent
        critic_agent: The critic agent to be managed
        agent_description: Optional additional description to append to the base description
        system_prompt: Optional custom system prompt to use instead of the default
        
    Returns:
        A configured writer agent that manages the critic
    """

    # Append additional description if provided
    base_description = 'A creative writer who drafts content based on prompts and iteratively improves it with feedback.'
    if agent_description:
        description = f"{base_description} {agent_description}"
    else:
        description = base_description
    
    agent = CodeAgent(
        tools=[save_draft],
        additional_authorized_imports=["json"],
        model=model,
        managed_agents=[critic_agent],  # Writer can call the critic
        name='writer_agent',
        description=description,
        max_steps=max_steps,
    )

    # Apply custom templates to initialize all prompt templates
    apply_custom_agent_prompts(agent)
    
    # Get the base system prompt and append to it
    base_sys_prompt = agent.prompt_templates["system_prompt"]
    
    # Append provided system prompt or default to a generic one
    if system_prompt:
        sys_prompt_appended = base_sys_prompt + f"\n\n{system_prompt}"
    else:
        sys_prompt_appended = base_sys_prompt + """\n\nYou are a creative writer who drafts and edits writing content to fulfil your assigned task. 
Your writing should be interesting, engaging, well-structured, and tailored to the requested style, topic, and likely audience.

Your task is to write and iteratively improve drafts. Here's how you should approach this task:

1. Write an initial draft based on the input prompt
2. Save your draft using the save_draft tool
3. Call the critic_agent with your draft to get feedback, e.g. critic_agent(task="provide feedback on this draft: {draft}")
4. Consider the resulting critic's feedback very carefully and always use it to guide your next draft
5. Write a new draft incorporating the feedback
6. Repeat steps 2-5 until you are completely satisfied with the result

Always save each version of your draft so there's a record of your progress by using the save_draft tool.

In your final answer, always include your complete final draft.
"""
    agent.prompt_templates["system_prompt"] = sys_prompt_appended # no need to reinitialize the system prompt, as no variables are used in the prompt

    return agent 