from typing import List, Optional
from smolagents import CodeAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from utils.agents.tools import apply_custom_agent_prompts, load_latest_draft, load_latest_report, load_file, save_final_answer
import time


class ManagerAgent:
    """A wrapper class for the manager agent that handles initialization and query execution."""
    
    def __init__(
        self,
        managed_agents: List,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 20,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the manager agent.
        
        Args:
            managed_agents: List of agents that the manager will coordinate
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum steps the manager agent can take
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
            
        self.managed_agents = managed_agents
        self.max_steps = max_steps
        
        # Create the agent
        # Create a list of available agent names
        available_agents = [agent.name for agent in self.managed_agents]
        available_agents_text = ", ".join(available_agents)
                
        self.agent = CodeAgent(
            tools=[load_latest_draft, load_latest_report, load_file],
            model=self.model,
            managed_agents=self.managed_agents,
            additional_authorized_imports=["time", "json", "re"],
            name='manager_agent',
            description=f"""This is a manager agent that coordinates other specialized agents. It can receive high-level objectives and determine which agents to deploy for different subtasks, how to sequence their use, and how to synthesize their outputs.""",
            max_steps=self.max_steps
        )
        
        # Define the custom system prompt
        custom_system_prompt = f"""You are a manager agent in charge of coordinating a team of specialized agents.

When given a task, you should:
1. Make a detailed plan for how to complete the task
2. Break down complex tasks into smaller subtasks
3. Determine which agent is best suited for each subtask
4. Call the appropriate agent(s) with clear, specific instructions
5. Review the results from each agent for quality and relevance
6. Refine your plan and approach if needed, based on the results of the agents
7. Repeat steps 4-6 until all aspects of the task are 100% complete and you are SURE that the task is fully addressed
8. Always make sure that you call the editor_agent as the final step, with instructions to thoroughly edit and fact check the end result
9. Synthesize the editor's final results into a complete, cohesive response

If an agent failed to return their complete draft or report result, make sure to load the latest draft or report and input it into the task instructions of the next agent you call.
Be persistent and iterative in your approach. If an agent's results aren't satisfactory, refine your instructions and try again. Only when you are sure all aspects of the original task have been thoroughly addressed should you provide your final response, so the user doesn't have to send the task back to you for additional iterations.

IMPORTANT: Your final_answer MUST include your complete, synthesized final result in markdown format. The final_answer will be automatically saved as a file, so make sure it contains the complete, cohesive response that addresses all aspects of the original task.

Always maintain a clean, organized format in your responses, including citations and sources where appropriate.
"""

        # Apply custom templates with the custom system prompt
        apply_custom_agent_prompts(self.agent, custom_system_prompt)
    
    def run_query(self, query: str) -> str:
        """Run the agent with a management query and return the result.
        
        Args:
            query: The management query to run
            
        Returns:
            The result from the agent containing the managed response
        """
        # Time the query execution
        start_time = time.time()
        
        # Run the query directly on the agent
        result = self.agent.run(query)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the final answer using the shared tool
        save_final_answer(
            agent=self.agent,
            result=result,
            query_or_prompt=query,
            agent_name="manager_agent",
            file_type="report",
            additional_metadata={"execution_time": execution_time}
        )
        
        return result
