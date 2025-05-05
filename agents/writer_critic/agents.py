"""Writer and critic agents."""
from smolagents import CodeAgent, ToolCallingAgent
from typing import Optional
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from ..utils.agents.tools import apply_custom_agent_prompts, save_final_answer
import time


class CriticAgent:
    """A wrapper class for the critic agent that handles initialization and feedback."""
    
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
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        # Append additional description if provided
        base_description = 'A literary critic who analyzes and provides constructive feedback on creative content.'
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
        
        self.agent = ToolCallingAgent(
            tools=[],  # Critic doesn't need tools - it just provides feedback
            model=self.model,
            name='critic_agent',
            description=description,
            max_steps=max_steps,  # Critic just needs one step to analyze and respond
        )

        # Default system prompt if none provided
        default_system_prompt = """You are a literary critic who analyzes and provides constructive feedback on written content. 
Your role is to provide constructive feedback to your managing writer agent on the content, style, structure, themes, and overall quality of their latest draft.

Do NOT focus on safety and ethical issues (these will be addressed elsewhere); if there are references to safety and ethical issues, you MUST require the writer to remove them and focus only on creating a high quality, engaging, and interesting piece of content.

Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements. Do not make any changes to the draft yourself.
Provide your feedback as plain text, without any special tags.
"""

        # Apply custom templates with the appropriate system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    def provide_feedback(self, draft: str) -> str:
        """Provide feedback on a draft.
        
        Args:
            draft: The draft to provide feedback on
            
        Returns:
            The feedback from the critic agent
        """
        task = f"Provide feedback on improvements to this draft: {draft}"
        return self.agent.run(task)


class WriterAgent:
    """A wrapper class for the writer agent that handles initialization and draft creation."""
    
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
        critic_prompt: Optional[str] = None
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
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        # Create a critic agent if one wasn't provided
        if critic_agent is None:
            critic = CriticAgent(
                model=self.model,
                agent_description=critic_description,
                system_prompt=critic_prompt,
                max_steps=1
            )
            self.critic_agent = critic.agent
        else:
            self.critic_agent = critic_agent
            
        # Append additional description if provided
        base_description = 'A creative writer who drafts content based on prompts and iteratively improves it with feedback.'
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
        
        self.agent = CodeAgent(
            tools=[],
            additional_authorized_imports=["json"],
            model=self.model,
            managed_agents=[self.critic_agent],  # Writer can call the critic
            name='writer_agent',
            description=description,
            max_steps=max_steps,
        )

        # Default system prompt if none provided
        default_system_prompt = """You are a creative writer who drafts and edits writing content to fulfil your assigned task. 
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

        # Apply custom templates with the appropriate system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    def write_draft(self, prompt: str) -> str:
        """Write a draft based on a prompt.
        
        Args:
            prompt: The writing prompt
            
        Returns:
            The final draft from the writer agent
        """
        # Time the execution
        start_time = time.time()
        
        # Run the writer agent with the prompt
        result = self.agent.run(prompt)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the final answer using the shared tool
        save_final_answer(
            agent=self.agent,
            result=result,
            query_or_prompt=prompt,
            agent_type="writer_critic"
        )
        
        return result 