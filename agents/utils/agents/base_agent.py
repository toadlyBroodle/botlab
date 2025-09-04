"""Base agent class providing common functionality for all specialized agents.

This base class encapsulates:
- Rate-limited model management
- Agent initialization (CodeAgent or ToolCallingAgent)
- Retry logic with exponential backoff
- Execution timing and result saving
- Custom prompt application
- Error handling and logging
- Automatic daily quota fallback

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

from .rate_lim_llm import RateLimitedLiteLLMModel, parse_retry_delay_from_error, is_per_minute_rate_limit_error, handle_per_minute_rate_limit, DAILY_QUOTA_ID, check_and_handle_search_error_message, safe_search_with_quota_detection, mark_model_daily_quota_exhausted, are_all_fallback_models_exhausted, is_model_daily_quota_exhausted, AllDailySearchRateLimsExhausted
from .simple_llm import SimpleLiteLLMModel
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
    - Automatic daily quota fallback to lower-tier models
    
    Subclasses need to implement:
    - get_agent_type(): Return "code" or "toolcalling"
    - get_tools(): Return list of tools for the agent
    - get_base_description(): Return base description for the agent
    - get_default_system_prompt(): Return default system prompt
    - Optionally override execute_task() for custom execution logic
    """
    
    def __init__(
        self,
        model: Optional[Union[RateLimitedLiteLLMModel, SimpleLiteLLMModel]] = None,
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
        enable_daily_quota_fallback: bool = True,
        use_rate_limiting: bool = True,
        **kwargs
    ):
        """Initialize the base agent with common functionality.
        
        Args:
            model: Optional LiteLLM model to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the agent
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model (only used with rate limiting)
            max_retries: Maximum retries for rate limiting if creating a new model (only used with rate limiting)
            additional_tools: Optional list of additional tools to include with the agent
            additional_authorized_imports: Optional list of additional imports for CodeAgent
            managed_agents: Optional list of managed agents for the agent
            agent_name: Optional custom name for the agent (defaults to class-based name)
            enable_daily_quota_fallback: Whether to enable automatic fallback on daily quota errors (only used with rate limiting)
            use_rate_limiting: Whether to use RateLimitedLiteLLMModel (True) or SimpleLiteLLMModel (False)
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
        self.enable_daily_quota_fallback = enable_daily_quota_fallback
        self.use_rate_limiting = use_rate_limiting
        
        # Extract cost_callback from kwargs if present
        cost_callback = kwargs.get('cost_callback', None)
        
        # Create a model if one wasn't provided
        if model is None:
            if use_rate_limiting:
                logger.info(f"Creating RateLimitedLiteLLMModel with rate limiting enabled")
                self.model = RateLimitedLiteLLMModel(
                    model_id=model_id,
                    model_info_path=model_info_path,
                    base_wait_time=base_wait_time,
                    max_retries=max_retries,
                    enable_fallback=enable_daily_quota_fallback,  # Enable fallback for daily quota handling
                    cost_callback=cost_callback,  # Pass cost callback
                )
            else:
                logger.info(f"Creating SimpleLiteLLMModel without rate limiting")
                self.model = SimpleLiteLLMModel(
                    model_id=model_id,
                    model_info_path=model_info_path,
                    cost_callback=cost_callback,  # Pass cost callback
                )
        else:
            self.model = model
            # If daily quota fallback is enabled but the model doesn't have fallback enabled, update it
            if (enable_daily_quota_fallback and 
                hasattr(self.model, 'enable_fallback') and 
                not getattr(self.model, 'enable_fallback', False)):
                logger.warning(f"Enabling fallback on provided model {self.model.model_id} for daily quota handling")
                self.model.enable_fallback = True
                # Also pre-initialize fallback models if the method exists
                if hasattr(self.model, '_initialize_fallback_models'):
                    try:
                        self.model._initialize_fallback_models()
                    except Exception as e:
                        logger.warning(f"Failed to pre-initialize fallback models: {e}")
            
            # Set cost callback on provided model if it doesn't have one
            if cost_callback and hasattr(self.model, 'cost_callback') and not self.model.cost_callback:
                logger.info(f"Setting cost callback on provided model")
                self.model.cost_callback = cost_callback
        
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
            
        Raises:
            AgentGenerationError: If daily quota error is detected in the response or execution steps
            AllDailySearchRateLimsExhausted: If all search options are exhausted
        """
        try:
            result = self.agent.run(task)
            
            # Check for AllDailySearchRateLimsExhausted in the main result
            if isinstance(result, str):
                if ("AllDailySearchRateLimsExhausted" in result or 
                    "SEARCH_EXHAUSTION_CRITICAL_ERROR" in result or
                    ("DuckDuckGo search failed after" in result and "Google search is disabled" in result)):
                    logger.error(f"All search options exhausted detected in agent result: {result[:300]}...")
                    raise AllDailySearchRateLimsExhausted(f"All search options exhausted in agent result: {result}")
            
            # Check if the final result contains a daily quota error message
            if isinstance(result, str) and "DAILY_QUOTA_ERROR:" in result:
                # Check if this is a search quota error vs model quota error
                if self._is_search_quota_error(result):
                    logger.warning(f"Search quota error detected in agent response (not a model error): {result[:300]}...")
                    # For search quota errors, don't raise an exception - let the agent continue
                    # The search tool should have handled this internally by disabling Google search
                    return result
                else:
                    logger.error(f"Model daily quota error detected in agent response: {result[:300]}...")
                    # Convert to AgentGenerationError so it can be handled by the run method's daily quota logic
                    raise AgentGenerationError(f"Daily quota error detected in agent response: {result}", self.agent.logger)
            
            # Check for AllDailySearchRateLimsExhausted in the agent's execution steps
            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'steps'):
                for step in self.agent.memory.steps:
                    if hasattr(step, 'tool_calls') and step.tool_calls:
                        for tool_call in step.tool_calls:
                            if hasattr(tool_call, 'result') and tool_call.result:
                                tool_result = str(tool_call.result)
                                
                                # Check for AllDailySearchRateLimsExhausted exception in tool execution
                                if ("AllDailySearchRateLimsExhausted" in tool_result or 
                                    "SEARCH_EXHAUSTION_CRITICAL_ERROR" in tool_result or
                                    "DuckDuckGo search failed after" in tool_result and "Google search is disabled" in tool_result):
                                    logger.error(f"All search options exhausted detected in tool execution: {tool_result[:300]}...")
                                    raise AllDailySearchRateLimsExhausted(f"All search options exhausted during tool execution: {tool_result}")
                                
                                if "DAILY_QUOTA_ERROR:" in tool_result:
                                    # Check if this is a search quota error vs model quota error
                                    if self._is_search_quota_error(tool_result):
                                        logger.warning(f"Search quota error detected in tool call result (not a model error): {tool_result[:300]}...")
                                        # For search quota errors, don't raise an exception - let the agent continue
                                        continue
                                    else:
                                        logger.error(f"Model daily quota error detected in tool call result: {tool_result[:300]}...")
                                        # Convert to AgentGenerationError so it can be handled by the run method's daily quota logic
                                        raise AgentGenerationError(f"Daily quota error detected in tool execution: {tool_result}", self.agent.logger)
                                        
                    # Also check step error if it exists
                    if hasattr(step, 'error') and step.error:
                        step_error = step.error
                        # Check if error is an AllDailySearchRateLimsExhausted exception object
                        if isinstance(step_error, AllDailySearchRateLimsExhausted):
                            logger.error(f"AllDailySearchRateLimsExhausted exception found in step: {str(step_error)}")
                            raise step_error
                        # Check if error is a string containing the exception details
                        step_error_str = str(step_error)
                        if ("AllDailySearchRateLimsExhausted" in step_error_str or 
                            "SEARCH_EXHAUSTION_CRITICAL_ERROR" in step_error_str or
                            "DuckDuckGo search failed after" in step_error_str and "Google search is disabled" in step_error_str):
                            logger.error(f"All search options exhausted detected in step error: {step_error_str[:300]}...")
                            raise AllDailySearchRateLimsExhausted(f"All search options exhausted in step execution: {step_error_str}")
            
            return result
            
        except AllDailySearchRateLimsExhausted as e:
            # Re-raise AllDailySearchRateLimsExhausted without conversion
            logger.error(f"All search options exhausted: {str(e)}")
            raise
    
    def _is_search_quota_error(self, message: str) -> bool:
        """Check if a DAILY_QUOTA_ERROR message is related to search quota vs model quota.
        
        Args:
            message: The error message to check
            
        Returns:
            True if this is a search quota error, False if it's a model quota error
        """
        message_lower = message.lower()
        
        # Check for search-related keywords in the error message
        search_keywords = [
            "search", "gemini search", "google search", "web search",
            "search api", "search daily quota", "search quota",
            "performing search", "performing gemini search"
        ]
        
        # If the message contains search-related keywords, it's likely a search quota error
        for keyword in search_keywords:
            if keyword in message_lower:
                return True
        
        return False
    
    def _is_daily_quota_error(self, error: Exception) -> bool:
        """Check if the error is a daily quota exhaustion error using Google API quotaId.
        
        Args:
            error: The exception to check
            
        Returns:
            True if this is a daily quota error that should trigger fallback
        """
        error_str = str(error)
        
        # Check if daily quotaId is present in the error (with or without suffixes like "-FreeTier")
        is_daily_quota = DAILY_QUOTA_ID in error_str
        
        if is_daily_quota:
            logger.info(f"Detected daily quota error via quotaId: {error_str[:200]}...")
            
        return is_daily_quota
    
    def _handle_daily_quota_fallback(self, error: Exception) -> bool:
        """Handle daily quota errors by falling back to a lower-tier model.
        
        Args:
            error: The daily quota error
            
        Returns:
            True if fallback was successful, False otherwise
        """
        if not self.enable_daily_quota_fallback:
            logger.info("Daily quota fallback is disabled")
            return False
            
        logger.warning(f"Daily quota exceeded for {self.model.model_id}. Attempting to fallback to lower-tier model (skipping any models with exhausted daily quotas).")
        logger.debug(f"Daily quota error details: {str(error)[:500]}")
        
        # Try to get a fallback model using the existing fallback system
        # This will automatically skip models that have exhausted their daily quotas
        current_model_id = self.model.model_id
        
        try:
            fallback_model_id = self.model._get_fallback_model(current_model_id)
            
            if fallback_model_id and fallback_model_id != current_model_id:
                logger.warning(f"Falling back from {current_model_id} to {fallback_model_id} due to daily quota exhaustion")
                
                # Perform the model switch using the existing method
                success = self.model._perform_model_switch(
                    fallback_model_id, 
                    f"daily_quota_exhausted_for_{current_model_id}"
                )
                
                if success:
                    logger.info(f"Successfully switched to fallback model {fallback_model_id} for daily quota")
                    
                    # Update the agent's model reference to use the new model_id
                    self.agent.model = self.model
                    
                    # Log the fallback status
                    self.model.print_rate_limit_status(use_logger=True)
                    
                    return True
                else:
                    logger.error(f"Failed to switch to fallback model {fallback_model_id}")
                    return False
            else:
                if fallback_model_id is None:
                    logger.warning(f"No suitable fallback model available for {current_model_id}")
                else:
                    logger.warning(f"Fallback model is same as current model: {current_model_id}")
                return False
                
        except Exception as fallback_e:
            logger.error(f"Exception during daily quota fallback for {current_model_id}: {fallback_e}")
            return False
    
    def run(self, query: str, max_retries: Optional[int] = None, base_wait_time: Optional[float] = None) -> str:
        """Run the agent with a query, handling rate limits, retries, and result saving.
        
        This method provides the complete execution framework including:
        - Retry logic with exponential backoff
        - Rate limit error handling with API-specified retry delays
        - Automatic daily quota fallback to lower-tier models
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
        daily_quota_fallback_attempted = False
        
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
                
                # Save the final answer using the shared tool with daily master files
                save_final_answer(
                    agent=self.agent,
                    result=result,
                    query_or_prompt=query,
                    agent_type=self.get_agent_type_name(),
                    use_daily_master=True
                )
                
                return result
                
            except AgentGenerationError as e:
                last_error = e
                
                # Check if this is a daily quota error first
                if self._is_daily_quota_error(e) and not daily_quota_fallback_attempted:
                    logger.warning("Daily quota error detected - attempting model fallback")
                    daily_quota_fallback_attempted = True
                    
                    # Mark this model as daily quota exhausted
                    mark_model_daily_quota_exhausted(self.model.model_id)
                    
                    # Check if all models in the fallback chain are exhausted
                    if are_all_fallback_models_exhausted(self.model.model_id):
                        logger.error(f"All models in fallback chain are daily quota exhausted. Terminating execution.")
                        raise AgentGenerationError(f"All models in fallback chain are daily quota exhausted", self.agent.logger) from e
                    
                    if self._handle_daily_quota_fallback(e):
                        # Successfully fell back, retry immediately with new model
                        logger.info("Daily quota fallback successful - retrying with fallback model")
                        continue
                    else:
                        # Daily quota fallback failed or is disabled - but check if all models are truly exhausted
                        if not self.enable_daily_quota_fallback:
                            logger.error(f"Daily quota exceeded for {self.model.model_id} and fallback is disabled. Terminating execution.")
                            raise AgentGenerationError(f"Daily quota exceeded for {self.model.model_id} and fallback is disabled", self.agent.logger) from e
                        else:
                            logger.error(f"Daily quota exceeded for {self.model.model_id} and no suitable fallback model available (all may be exhausted). Terminating execution.")
                            raise AgentGenerationError(f"Daily quota exceeded for {self.model.model_id} and no suitable fallback model available", self.agent.logger) from e
                
                # If we've already attempted daily quota fallback, don't retry daily quota errors
                if self._is_daily_quota_error(e) and daily_quota_fallback_attempted:
                    logger.error(f"Daily quota error encountered again after fallback attempt. All available models may be exhausted. Terminating execution.")
                    raise AgentGenerationError(f"Daily quota error encountered again after fallback attempt", self.agent.logger) from e
                
                # Check if this is a regular rate limit error that we should retry
                error_str = str(e).lower()
                is_rate_limit = any(keyword in error_str for keyword in [
                    "rate limit", "quota", "429", "resource_exhausted", 
                    "resource exhausted", "too many requests"
                ])
                
                # Don't retry daily quota errors - they should have been handled above
                if self._is_daily_quota_error(e):
                    logger.error(f"Daily quota error not properly handled above. Terminating execution.")
                    raise AgentGenerationError(f"Daily quota error not properly handled", self.agent.logger) from e
                
                if not is_rate_limit:
                    logger.error(f"Non-rate-limit AgentGenerationError encountered: {str(e)[:200]}")
                    raise  # Re-raise non-rate-limit errors immediately
                    
                if attempt >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {self.__class__.__name__} execution")
                    raise  # Re-raise after max retries
                
                # Check if this is specifically a per-minute rate limit error
                if is_per_minute_rate_limit_error(e):
                    wait_time = handle_per_minute_rate_limit(e, f"{self.__class__.__name__} (attempt {attempt + 1}/{max_retries + 1})")
                    continue  # Continue to the next retry attempt
                    
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
                
                # Handle AllDailySearchRateLimsExhausted specifically
                if isinstance(e, AllDailySearchRateLimsExhausted):
                    logger.error(f"All search options exhausted during agent execution: {str(e)}")
                    raise  # Re-raise to terminate the calling process
                
                # For non-daily quota errors, re-raise as-is (daily quota errors are now handled in execute_task)
                raise  # Re-raise other exceptions immediately
        
        # This should not be reached due to the logic above, but just in case
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"Unexpected error in {self.__class__.__name__} run method")
    
    def get_current_call_cost_info(self) -> Optional[Dict[str, Any]]:
        """Get cost information for the current API call from the underlying model.
        
        Returns:
            Dictionary with current call cost information, or None if not available
        """
        if hasattr(self.model, 'get_current_call_cost_info'):
            return self.model.get_current_call_cost_info()
        return None
    
    def get_total_cost_info(self) -> Dict[str, Any]:
        """Get total accumulated cost information from the underlying model.
        
        Returns:
            Dictionary with total cost information
        """
        if hasattr(self.model, 'get_total_cost_info'):
            return self.model.get_total_cost_info()
        return {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_cost_cents': 0.0
        }
    
    def get_model_cost_info(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive cost information from the underlying model.
        
        This method provides both current call and total cost information
        in a format compatible with the CSV agent loop cost tracking.
        
        Returns:
            Dictionary with current and total cost information, or None if not available
        """
        try:
            current_cost = self.get_current_call_cost_info()
            total_cost = self.get_total_cost_info()
            
            if current_cost or total_cost:
                return {
                    'current_call': current_cost,
                    'total': total_cost,
                    'prompt_tokens': total_cost.get('prompt_tokens', 0),
                    'completion_tokens': total_cost.get('completion_tokens', 0),
                    'total_cost_cents': total_cost.get('total_cost_cents', 0.0)
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to get model cost info: {e}")
            return None


class BaseCodeAgent(BaseAgent):
    """Base class for agents that use CodeAgent."""
    
    def get_agent_type(self) -> str:
        return "code"


class BaseToolCallingAgent(BaseAgent):
    """Base class for agents that use ToolCallingAgent."""
    
    def get_agent_type(self) -> str:
        return "toolcalling" 