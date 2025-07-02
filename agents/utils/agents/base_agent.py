"""Base agent class providing common functionality for all specialized agents.

This base class encapsulates:
- Rate-limited model management
- Agent initialization (CodeAgent or ToolCallingAgent)
- Retry logic with exponential backoff
- Execution timing and result saving
- Custom prompt application
- Error handling and logging

All specialized agents should inherit from this base class and only implement
their specific tools, descriptions, and business logic.
"""

import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from smolagents import CodeAgent, ToolCallingAgent
from smolagents.utils import AgentGenerationError

from ..gemini.rate_lim_llm import RateLimitedLiteLLMModel, parse_retry_delay_from_error
from .tools import apply_custom_agent_prompts, save_final_answer

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all specialized agents with common functionality.
    
    This class provides:
    - Automatic model initialization with rate limiting
    - Agent creation (CodeAgent or ToolCallingAgent)  
    - Retry logic with exponential backoff and jitter
    - Execution timing and result saving
    - Custom prompt application
    - Consistent error handling and logging
    
    Subclasses need to implement:
    - get_agent_type(): Return "code" or "toolcalling"
    - get_tools(): Return list of tools for the agent
    - get_base_description(): Return base description for the agent
    - get_default_system_prompt(): Return default system prompt
    - Optionally override execute_task() for custom execution logic
    """
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 20,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        additional_tools: Optional[List] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        managed_agents: Optional[List] = None,
        agent_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize the base agent with common functionality.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the agent
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
            additional_tools: Optional list of additional tools to include with the agent
            additional_authorized_imports: Optional list of additional imports for CodeAgent
            managed_agents: Optional list of managed agents for the agent
            agent_name: Optional custom name for the agent (defaults to class-based name)
            **kwargs: Additional keyword arguments for agent creation
        """
        # Store configuration
        self.max_steps = max_steps
        self.base_wait_time = base_wait_time
        self.max_retries = max_retries
        self.additional_tools = additional_tools or []
        self.additional_authorized_imports = additional_authorized_imports or []
        self.managed_agents = managed_agents or []
        self.agent_kwargs = kwargs
        
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
        
        # Initialize the agent
        self._initialize_agent(agent_description, system_prompt, agent_name)
        
    def _initialize_agent(
        self, 
        agent_description: Optional[str], 
        system_prompt: Optional[str],
        agent_name: Optional[str]
    ) -> None:
        """Initialize the underlying smolagents agent."""
        # Get agent-specific configuration
        agent_type = self.get_agent_type()
        base_tools = self.get_tools()
        base_description = self.get_base_description()
        default_system_prompt = self.get_default_system_prompt()
        
        # Combine tools
        all_tools = base_tools + self.additional_tools
        
        # Create description
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
            
        # Create agent name
        if agent_name is None:
            agent_name = f"{self.__class__.__name__.lower().replace('agent', '')}_agent"
        
        # Create the appropriate agent type
        common_args = {
            'model': self.model,
            'name': agent_name,
            'description': description,
            'max_steps': self.max_steps,
            **self.agent_kwargs
        }
        
        if agent_type.lower() == 'code':
            self.agent = CodeAgent(
                tools=all_tools,
                additional_authorized_imports=self.additional_authorized_imports,
                managed_agents=self.managed_agents,
                **common_args
            )
        elif agent_type.lower() == 'toolcalling':
            self.agent = ToolCallingAgent(
                tools=all_tools,
                **common_args
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}. Must be 'code' or 'toolcalling'")
        
        # Apply custom system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the agent type: 'code' or 'toolcalling'."""
        pass
    
    @abstractmethod 
    def get_tools(self) -> List:
        """Return the list of tools for this agent type."""
        pass
    
    @abstractmethod
    def get_base_description(self) -> str:
        """Return the base description for this agent type."""
        pass
    
    @abstractmethod
    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type."""
        pass
    
    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving (e.g., 'researcher', 'writer_critic')."""
        # Default implementation - subclasses can override
        return self.__class__.__name__.lower().replace('agent', '')
    
    def execute_task(self, task: str) -> str:
        """Execute a task using the agent. Can be overridden by subclasses.
        
        Args:
            task: The task/query to execute
            
        Returns:
            The result from the agent
        """
        return self.agent.run(task)
    
    def run(self, query: str, max_retries: Optional[int] = None, base_wait_time: Optional[float] = None) -> str:
        """Run the agent with a query, handling rate limits, retries, and result saving.
        
        This method provides the complete execution framework including:
        - Retry logic with exponential backoff
        - Rate limit error handling with API-specified retry delays
        - Execution timing
        - Automatic result saving
        
        Args:
            query: The query/task to run
            max_retries: Maximum number of retries (uses instance default if None)
            base_wait_time: Base wait time for exponential backoff (uses instance default if None)
            
        Returns:
            The result from the agent containing the response
        """
        if max_retries is None:
            max_retries = self.max_retries
        if base_wait_time is None:
            base_wait_time = self.base_wait_time
            
        last_error = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Time the query execution
                start_time = time.time()
                
                logger.info(f"Running {self.__class__.__name__} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Execute the task using the agent
                result = self.execute_task(query)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                print(f"\nExecution time: {execution_time:.2f} seconds")
                
                # Save the final answer using the shared tool
                save_final_answer(
                    agent=self.agent,
                    result=result,
                    query_or_prompt=query,
                    agent_type=self.get_agent_type_name()
                )
                
                return result
                
            except AgentGenerationError as e:
                last_error = e
                
                # Check if this is a rate limit error that we should retry
                error_str = str(e).lower()
                is_rate_limit = any(keyword in error_str for keyword in [
                    "rate limit", "quota", "429", "resource_exhausted", 
                    "resource exhausted", "too many requests"
                ])
                
                if not is_rate_limit:
                    logger.error(f"Non-rate-limit AgentGenerationError encountered: {str(e)[:200]}")
                    raise  # Re-raise non-rate-limit errors immediately
                    
                if attempt >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {self.__class__.__name__} execution")
                    raise  # Re-raise after max retries
                    
                # Try to parse the retry delay from the error
                retry_delay = parse_retry_delay_from_error(e)
                
                if retry_delay is not None:
                    # Use the specific retry delay from the API
                    wait_time = retry_delay
                    logger.warning(f"Rate limit error on {self.__class__.__name__} attempt {attempt + 1}. "
                                 f"Waiting {wait_time}s as specified by API retryDelay")
                else:
                    # Fall back to exponential backoff
                    wait_time = base_wait_time * (2 ** attempt)
                    logger.warning(f"Rate limit error on {self.__class__.__name__} attempt {attempt + 1}. "
                                 f"No retryDelay found, using exponential backoff: {wait_time}s")
                
                # Add some jitter to avoid thundering herd
                jitter = random.uniform(0.1, 0.3) * wait_time
                total_wait_time = wait_time + jitter
                
                logger.info(f"Waiting {total_wait_time:.2f}s before retry...")
                time.sleep(total_wait_time)
                
            except Exception as e:
                # Handle other types of exceptions
                logger.error(f"Non-AgentGenerationError encountered: {type(e).__name__}: {str(e)[:200]}")
                raise  # Re-raise other exceptions immediately
        
        # This should not be reached due to the logic above, but just in case
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"Unexpected error in {self.__class__.__name__} run method")


class BaseCodeAgent(BaseAgent):
    """Base class for agents that use CodeAgent."""
    
    def get_agent_type(self) -> str:
        return "code"


class BaseToolCallingAgent(BaseAgent):
    """Base class for agents that use ToolCallingAgent."""
    
    def get_agent_type(self) -> str:
        return "toolcalling" 