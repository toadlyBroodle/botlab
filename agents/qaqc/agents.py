from typing import List, Optional, Dict, Any, Tuple
from smolagents import CodeAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from utils.agents.tools import apply_custom_agent_prompts, save_final_answer
from .tools import select_best_output
import time


class QAQCAgent:
    """A wrapper class for the QAQC agent that handles initialization and output selection."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 15,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the QAQC agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the agent
            agent_description: Optional custom description for the agent
            system_prompt: Optional custom system prompt for the agent
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
            
        # Use custom description if provided, otherwise use default
        description = agent_description or """This is a Quality Assurance/Quality Control agent that compares multiple outputs (e.g., reports, drafts, content) and selects the best one based on quality, accuracy, completeness, and relevance to the original query."""
        
        self.agent = CodeAgent(
            tools=[select_best_output],  # Add the selection tool
            model=self.model,
            additional_authorized_imports=["json"],
            name='qaqc_agent',
            description=description,
            max_steps=max_steps
        )
        
        # Define the custom system prompt if not provided
        custom_system_prompt = system_prompt or """You are a Quality Assurance/Quality Control agent responsible for comparing multiple outputs and selecting the best one.

When given multiple outputs to compare:

1. Carefully analyze each output for:
   - Accuracy of information
   - Completeness of coverage
   - Clarity and readability
   - Relevance to the original query
   - Logical structure and flow
   - Quality of writing and presentation

2. Create a detailed comparison table or rubric showing how each output performs on these criteria

3. Provide specific examples from each output to justify your assessment

4. Select the best output based on your analysis, clearly explaining why it's superior

5. If neither output is clearly superior overall, identify which parts of each are strongest and suggest how they could be combined

6. Always maintain objectivity in your assessment, focusing on measurable quality factors rather than subjective preferences

7. IMPORTANT: After completing your analysis, you MUST use the select_best_output tool to programmatically select the best output. This tool requires:
   - output_number: The number of the selected output (1 or 2)
   - output_text: The complete text of the selected output (copy and paste it exactly)

Your goal is to ensure that only the highest quality output moves forward in the process, improving the overall quality of the final result.

Do not modify either of the outputs you are given, only provide a comparison and selection of the best output.
"""

        # Apply custom templates with the custom system prompt
        apply_custom_agent_prompts(self.agent, custom_system_prompt)
    
    def compare_outputs(self, outputs: List[str], query: Optional[str] = None) -> str:
        """Compare multiple outputs and select the best one.
        
        Args:
            outputs: List of outputs to compare
            query: Optional original query for context
            
        Returns:
            The selected best output with analysis
        """
        # Time the execution
        start_time = time.time()
        
        # Prepare the prompt for the agent
        if len(outputs) != 2:
            raise ValueError("Currently only supports comparing exactly 2 outputs")
            
        prompt = f"Compare these two outputs and select the best one:\n\n"
        if query:
            prompt += f"Original Query: {query}\n\n"
            
        prompt += f"Output 1:\n{outputs[0]}\n\nOutput 2:\n{outputs[1]}"
        
        # Run the QAQC agent with the outputs
        result = self.agent.run(prompt)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the final answer using the shared tool
        save_final_answer(
            agent=self.agent,
            result=result,
            query_or_prompt=prompt,
            agent_name="qaqc_agent",
            file_type="comparison",
            additional_metadata={"execution_time": execution_time}
        )
        
        return result 