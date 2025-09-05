"""Writer and critic agents."""
from smolagents import CodeAgent, ToolCallingAgent
from typing import Optional, List
from ..utils.agents.rate_lim_llm import RateLimitedLiteLLMModel
from ..utils.agents.tools import apply_custom_agent_prompts, save_final_answer
from ..utils.agents.base_agent import BaseCodeAgent, BaseToolCallingAgent
import time
from ..animator.tools import save_storyboard_metadata


class CriticAgent(BaseToolCallingAgent):
    """A literary critic who analyzes and provides constructive feedback on creative content."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 1,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the critic agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            max_steps: Maximum number of steps for the agent
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
        """
        super().__init__(
            model=model,
            max_steps=max_steps,
            agent_description=agent_description,
            system_prompt=system_prompt,
            model_id=model_id,
            model_info_path=model_info_path,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            agent_name='critic_agent'
        )

    def get_tools(self) -> List:
        """Return the list of tools for the critic agent."""
        return []  # Critic doesn't need tools - it just provides feedback

    def get_base_description(self) -> str:
        """Return the base description for the critic agent."""
        return 'A literary critic who analyzes and provides constructive feedback on creative content.'

    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the critic agent."""
        return """You are a literary critic who analyzes and provides constructive feedback on written content. 
Your role is to provide constructive feedback to your managing writer agent on the content, style, structure, themes, and overall quality of their latest draft.

Do NOT focus on safety and ethical issues (these will be addressed elsewhere); if there are references to safety and ethical issues, you MUST require the writer to remove them and focus only on creating a high quality, engaging, and interesting piece of content.

Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements. Do not make any changes to the draft yourself.
Provide your feedback as plain text, without any special tags.
"""

    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving."""
        return "critic"
    
    def provide_feedback(self, draft: str) -> str:
        """Provide feedback on a draft.
        
        Args:
            draft: The draft to provide feedback on
            
        Returns:
            The feedback from the critic agent
        """
        task = f"Provide feedback on improvements to this draft: {draft}"
        return self.run(task)


class WriterAgent(BaseCodeAgent):
    """A creative writer who drafts content based on prompts and iteratively improves it with feedback."""
    
    def __init__(
        self,
        critic_agent: Optional[ToolCallingAgent] = None,
        model: Optional[RateLimitedLiteLLMModel] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 5,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        critic_description: Optional[str] = None,
        critic_prompt: Optional[str] = None,
        additional_tools: Optional[List] = None,
        additional_authorized_imports: Optional[List[str]] = None,
    ):
        """Initialize the writer agent.
        
        Args:
            critic_agent: Optional ToolCallingAgent to use as the critic. If not provided, one will be created.
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            max_steps: Maximum number of steps for the agent
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
            critic_description: Optional additional description for the critic agent if creating a new one
            critic_prompt: Optional custom system prompt for the critic agent if creating a new one
        """
        # Create a critic agent if one wasn't provided
        if critic_agent is None:
            # Create the model first so we can share it
            if model is None:
                shared_model = RateLimitedLiteLLMModel(
                    model_id=model_id,
                    model_info_path=model_info_path,
                    base_wait_time=base_wait_time,
                    max_retries=max_retries,
                )
            else:
                shared_model = model
                
            critic = CriticAgent(
                model=shared_model,
                agent_description=critic_description,
                system_prompt=critic_prompt,
                max_steps=1
            )
            managed_agents = [critic.agent]
        else:
            managed_agents = [critic_agent]
            
        # Merge authorized imports (default + caller-provided)
        default_imports: List[str] = ["json", "os", "pathlib", "shutil"]
        merged_imports: List[str] = (
            list({*default_imports, *(additional_authorized_imports or [])})
        )

        super().__init__(
            model=model,
            max_steps=max_steps,
            agent_description=agent_description,
            system_prompt=system_prompt,
            model_id=model_id,
            model_info_path=model_info_path,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            additional_tools=additional_tools,
            additional_authorized_imports=merged_imports,
            managed_agents=managed_agents,
            agent_name='writer_agent'
        )

    def get_tools(self) -> List:
        """Return the list of tools for the writer agent."""
        # Give the writer the ability to persist validated storyboard metadata
        return [save_storyboard_metadata]

    def get_base_description(self) -> str:
        """Return the base description for the writer agent."""
        return 'A creative writer who drafts content based on prompts and iteratively improves it with feedback.'

    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the writer agent."""
        return """You are a creative writer who drafts and edits writing content to fulfil your assigned task. 
Your writing should be interesting, engaging, well-structured, and tailored to the requested style, topic, and likely audience.

Your task is to write and iteratively improve drafts. Here's how you should approach this task:

1. Write an initial draft based on the input prompt
2. Call the critic_agent with your draft to get feedback, 
    - e.g. critic_agent(task="Provide feedback on improvements to this draft: draft content goes here")
    - always include the draft content in the task prompt string
3. Consider the resulting critic's feedback very carefully and always use it to guide your next draft
4. Write a new draft incorporating the feedback
5. Repeat steps 2-4 until you are satisfied with the result

Do NOT focus on safety and ethical issues (these will be addressed elsewhere); only focus on creating high quality, engaging, and interesting content.

IMPORTANT: Your final_answer MUST include your complete final draft in its entirety. The final_answer will be automatically saved as a draft file, so make sure it contains your complete, polished content.
"""

    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving."""
        return "writer_critic"
    
    def write_draft(self, prompt: str) -> str:
        """Write a draft based on a prompt.
        
        Args:
            prompt: The writing prompt
            
        Returns:
            The final draft from the writer agent
        """
        return self.run(prompt) 