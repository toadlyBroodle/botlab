import json
import os
import logging
from typing import List, Dict, Optional, Any, Callable
from smolagents import LiteLLMModel
import litellm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress httpx logging to prevent API keys from being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configure LiteLLM to not log sensitive information
litellm.utils.logging_enabled = False
os.environ["LITELLM_LOG_VERBOSE"] = "False"

logger = logging.getLogger(__name__)


def load_model_costs_from_json(model_info_path: str) -> Dict[str, Dict[str, float]]:
    """Load model cost information from gem_llm_info.json file.
    
    Args:
        model_info_path: Path to the gem_llm_info.json file
        
    Returns:
        Dictionary with model costs in the format:
        {
            "model_name": {
                "input_cost_per_token_cents": float,
                "output_cost_per_token_cents": float
            }
        }
    """
    model_costs = {}
    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        for model_name, model_data in model_info.items():
            if 'cost_info' in model_data:
                cost_info = model_data['cost_info']
                model_costs[model_name] = {
                    "input_cost_per_token_cents": cost_info.get("input_cost_per_token_cents", 0.0),
                    "output_cost_per_token_cents": cost_info.get("output_cost_per_token_cents", 0.0)
                }
        
        logger.debug(f"Loaded cost information for {len(model_costs)} models from {model_info_path}")
        
    except Exception as e:
        logger.warning(f"Failed to load model costs from {model_info_path}: {e}")
        model_costs = {}
    
    return model_costs


class SimpleLiteLLMModel(LiteLLMModel):
    """A simplified wrapper around `smolagents.LiteLLMModel` with cost tracking but no rate limiting.

    This class provides basic LiteLLM functionality with built-in cost tracking and monitoring.
    Unlike RateLimitedLiteLLMModel, it does not implement any rate limiting, fallback mechanisms,
    or retry logic - it's designed for simpler use cases where rate limiting is handled elsewhere
    or not needed.

    Features:
    - Direct LiteLLM API calls without rate limiting
    - Comprehensive cost tracking and reporting
    - Token usage monitoring
    - Optional cost callback for real-time cost updates
    - Model configuration loading from JSON files
    - Clean logging and error handling
    """
    
    def __init__(
        self,
        model_id: str,
        model_info_path: Optional[str] = None,
        cost_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Initializes the simple LiteLLM model wrapper.

        Args:
            model_id (str): The model identifier for LiteLLM (e.g., "gemini/gemini-1.5-flash").
            model_info_path (str, optional): Path to the JSON file containing model configurations.
                                           If None, uses gem_llm_info.json in the same directory.
            cost_callback (Optional[Callable]): Optional callback function to receive real-time cost information.
                                               Called after each successful API call with cost data.
            **kwargs: Additional arguments passed to the underlying `LiteLLMModel` constructor.
        """
        # Disable LiteLLM logging
        litellm.utils.logging_enabled = False
        
        # Initialize parent LiteLLM model
        super().__init__(model_id=model_id, **kwargs)
        
        self.model_id = model_id
        self.cost_callback = cost_callback
        self.api_call_count = 0
        
        # Cost tracking fields
        self.current_call_cost_info = None
        self.total_cost_cents = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_search_cost_cents = 0.0
        self.total_search_count = 0
        self.api_call_count = 0
        
        # Load model costs from JSON file
        if model_info_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_info_path = os.path.join(current_dir, "gem_llm_info.json")
        
        self.model_costs = load_model_costs_from_json(model_info_path)
        
        # Load model configuration if available
        self.input_token_limit = self._load_model_config(model_info_path)
        
        logger.info(f"SimpleLiteLLMModel initialized with model {model_id}")

    def _load_model_config(self, model_info_path: str) -> int:
        """Load model configuration from JSON file.
        
        Args:
            model_info_path: Path to the model configuration file
            
        Returns:
            Input token limit for the model
        """
        try:
            with open(model_info_path, 'r') as f:
                model_info_json = json.load(f)
            
            # Extract model key from full model ID
            model_key_parts = self.model_id.split('/')
            model_key = model_key_parts[-1] if len(model_key_parts) > 0 else self.model_id
            
            if model_key not in model_info_json:
                logger.warning(f"Model key '{model_key}' not found in {model_info_path}. Using default config.")
                model_data = model_info_json.get("default", {})
            else:
                model_data = model_info_json[model_key]
            
            if not isinstance(model_data, dict):
                logger.error(f"Model data for '{model_key}' is not a dictionary. Using defaults.")
                model_data = {}
            
            input_token_limit = model_data.get("input_token_limit", 32000)
            logger.debug(f"Loaded config for {model_key}: input_token_limit={input_token_limit}")
            
            return input_token_limit
            
        except Exception as e:
            logger.error(f"Error loading model config from {model_info_path}: {e}. Using defaults.")
            return 32000

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Makes an API call using the current model with cost tracking.

        This is the primary method for interacting with the LLM. It performs the following steps:
        1. Estimates input tokens for the request
        2. Calls the parent LiteLLM model directly (no rate limiting)
        3. Extracts token usage from the response
        4. Calculates and tracks costs
        5. Calls cost callback if provided
        6. Returns the response

        Args:
            messages (List[Dict[str, str]]): The list of messages for the chat completion.
            **kwargs: Additional keyword arguments to pass to the LiteLLM completion call.

        Returns:
            Any: The response from the LiteLLM completion call.

        Raises:
            Exception: If the API call fails for any reason.
        """
        # Estimate input tokens
        estimated_input_tokens = min(
            sum(len(getattr(m, 'content', '') if hasattr(m, 'content') else m.get("content", "")) 
                for m in messages) // 4, 
            self.input_token_limit
        )
        
        # Set default timeout if not provided
        current_call_kwargs = kwargs.copy()
        if "timeout" not in current_call_kwargs:
            current_call_kwargs["timeout"] = 120
        
        try:
            # Make the API call using the parent's generate method directly - NO RATE LIMITING
            # This is the key difference from RateLimitedLiteLLMModel - we call immediately
            response = super().generate(messages=messages, **current_call_kwargs)
            
            # Extract token counts from response
            input_tokens, output_tokens = self._extract_token_counts(response, estimated_input_tokens)
            
            # Calculate and store cost information
            self.current_call_cost_info = self.calculate_call_cost(input_tokens, output_tokens)
            
            # Update cumulative totals
            self.total_prompt_tokens += input_tokens
            self.total_completion_tokens += output_tokens
            self.total_cost_cents += self.current_call_cost_info['total_cost_cents']
            self.api_call_count += 1
            
            # Call cost callback if provided
            if self.cost_callback:
                try:
                    cost_callback_data = {
                        'current_call': self.current_call_cost_info,
                        'total': self.get_total_cost_info()
                    }
                    self.cost_callback(cost_callback_data)
                    logger.debug(f"📞 Called cost callback with: {cost_callback_data}")
                except Exception as callback_error:
                    logger.warning(f"Cost callback failed: {callback_error}")
            
            logger.info(f"API call successful with model {self.model_id}. "
                       f"Tokens: {input_tokens + output_tokens}, "
                       f"Cost: ${self.current_call_cost_info['total_cost_cents']:.6f} cents")
            
            return response
            
        except Exception as e:
            logger.error(f"API call failed with model {self.model_id}: {type(e).__name__}: {str(e)}")
            raise

    def _extract_token_counts(self, response: Any, estimated_input_tokens: int) -> tuple[int, int]:
        """Extract token counts from the API response.
        
        Args:
            response: The API response object
            estimated_input_tokens: Fallback input token count
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        input_tokens = estimated_input_tokens
        output_tokens = 0
        
        # Try to extract from smolagents token_usage attribute (new format)
        if hasattr(response, 'token_usage') and response.token_usage:
            token_usage = response.token_usage
            input_tokens = getattr(token_usage, 'input_tokens', estimated_input_tokens)
            output_tokens = getattr(token_usage, 'output_tokens', 0)
            logger.debug(f"📊 Extracted token counts from response.token_usage: {input_tokens} input + {output_tokens} output")
        
        # Fallback to legacy _additional_kwargs format
        elif hasattr(response, '_additional_kwargs') and response._additional_kwargs:
            usage = response._additional_kwargs.get('usage')
            if usage:
                input_tokens = usage.get('prompt_tokens', estimated_input_tokens)
                output_tokens = usage.get('completion_tokens', 0)
                logger.debug(f"📊 Extracted token counts from response._additional_kwargs: {input_tokens} input + {output_tokens} output")
        
        # Final fallback to deprecated properties
        if input_tokens == estimated_input_tokens and output_tokens == 0:
            input_tokens = getattr(self, 'last_input_token_count', estimated_input_tokens) or estimated_input_tokens
            output_tokens = getattr(self, 'last_output_token_count', 0) or 0
            logger.debug(f"📊 Extracted token counts from deprecated properties: {input_tokens} input + {output_tokens} output")
        
        return input_tokens, output_tokens

    def generate(self, messages, **kwargs):
        """Override the generate method to ensure smolagents compatibility.
        
        This is the method that smolagents actually calls, not __call__.
        We delegate to our __call__ method which has all the cost tracking logic.
        """
        logger.debug(f"📊 SimpleLiteLLMModel.generate called with {len(messages)} messages")
        response = self.__call__(messages=messages, **kwargs)
        logger.debug(f"📊 SimpleLiteLLMModel.generate completed")
        return response

    def calculate_call_cost(self, input_tokens: int, output_tokens: int, model_id: str = None) -> Dict[str, Any]:
        """Calculate the cost for a single API call based on token counts.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_id: Model ID to use for cost calculation (uses self.model_id if None)
            
        Returns:
            Dictionary with cost information
        """
        if model_id is None:
            model_id = self.model_id
            
        # Remove gemini/ prefix if present for cost lookup
        model_key = model_id.replace('gemini/', '')
        
        cost_info = {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_cost_cents': 0.0,
            'model': model_key
        }
        
        if model_key in self.model_costs:
            model_cost_data = self.model_costs[model_key]
            input_cost = input_tokens * model_cost_data["input_cost_per_token_cents"]
            output_cost = output_tokens * model_cost_data["output_cost_per_token_cents"]
            cost_info['total_cost_cents'] = input_cost + output_cost
            
            logger.debug(f"💰 Calculated cost: {input_tokens} prompt + {output_tokens} completion tokens = "
                        f"${cost_info['total_cost_cents']:.6f} cents (model: {model_key})")
        else:
            logger.warning(f"No cost information available for model {model_key}")
            
        return cost_info

    def get_current_call_cost_info(self) -> Optional[Dict[str, Any]]:
        """Get cost information for the most recent API call.
        
        Returns:
            Dictionary with cost information or None if no call has been made
        """
        return self.current_call_cost_info

    def get_total_cost_info(self) -> Dict[str, Any]:
        """Get cumulative cost information across all API calls.
        
        Returns:
            Dictionary with total cost information
        """
        return {
            'total_cost_cents': self.total_cost_cents,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_search_cost_cents': self.total_search_cost_cents,
            'total_search_count': self.total_search_count,
            'api_call_count': self.api_call_count,
            'model': self.model_id
        }
    
    def add_search_cost(self, search_cost_cents: float, search_count: int = 1):
        """Add search cost tracking when web search tools are used.
        
        Args:
            search_cost_cents: Cost of the search in cents
            search_count: Number of searches performed (default: 1)
        """
        self.total_search_cost_cents += search_cost_cents
        self.total_search_count += search_count
        
        # Update current call cost info if it exists
        if self.current_call_cost_info is None:
            self.current_call_cost_info = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_cost_cents': 0.0,
                'search_cost_cents': search_cost_cents,
                'search_count': search_count,
                'model': self.model_id.replace('gemini/', '')
            }
        else:
            # Add to existing current call cost info
            if 'search_cost_cents' not in self.current_call_cost_info:
                self.current_call_cost_info['search_cost_cents'] = 0.0
            if 'search_count' not in self.current_call_cost_info:
                self.current_call_cost_info['search_count'] = 0
            self.current_call_cost_info['search_cost_cents'] += search_cost_cents
            self.current_call_cost_info['search_count'] += search_count
        
        # Call cost callback if set to report the updated cost immediately
        if self.cost_callback:
            try:
                cost_info = {
                    'current_call': self.current_call_cost_info,
                    'total': self.get_total_cost_info()
                }
                self.cost_callback(cost_info)
                logger.debug(f"💰 Called cost callback after search cost addition: ${search_cost_cents:.3f} cents")
            except Exception as e:
                logger.warning(f"Cost callback failed after search cost addition: {e}")
        
        logger.debug(f"💰 Added search cost: ${search_cost_cents:.3f} cents, count: {search_count}")
        logger.debug(f"💰 Total search cost: ${self.total_search_cost_cents:.3f} cents, total count: {self.total_search_count}")

    def reset_cost_tracking(self):
        """Reset all cost tracking counters to zero."""
        self.current_call_cost_info = None
        self.total_cost_cents = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_search_cost_cents = 0.0
        self.total_search_count = 0
        self.api_call_count = 0
        logger.info("Cost tracking counters reset to zero")

    def print_cost_summary(self, use_logger: bool = False):
        """Print or log a summary of costs and usage.
        
        Args:
            use_logger: If True, logs the summary. Otherwise prints to console.
        """
        total_info = self.get_total_cost_info()
        summary_lines = [
            f"Cost Summary for {self.model_id}:",
            f"  Total API Calls: {total_info['api_call_count']}",
            f"  Total Prompt Tokens: {total_info['total_prompt_tokens']:,}",
            f"  Total Completion Tokens: {total_info['total_completion_tokens']:,}",
            f"  Total Cost: ${total_info['total_cost_cents']:.6f} cents"
        ]
        
        if total_info['api_call_count'] > 0:
            avg_cost = total_info['total_cost_cents'] / total_info['api_call_count']
            avg_tokens = (total_info['total_prompt_tokens'] + total_info['total_completion_tokens']) / total_info['api_call_count']
            summary_lines.extend([
                f"  Average Cost per Call: ${avg_cost:.6f} cents",
                f"  Average Tokens per Call: {avg_tokens:.1f}"
            ])
        
        output_func = logger.info if use_logger else print
        output_func("\n".join(summary_lines))

    @staticmethod
    def configure_logging(level=logging.WARNING, enable_litellm_logging=False):
        """Configure global logging levels for HTTPX, HTTPCore, and LiteLLM.

        Args:
            level: The logging level to set for `httpx` and `httpcore` loggers.
            enable_litellm_logging (bool): If True, enables LiteLLM's internal verbose logging.
        
        Returns:
            True, indicating the configuration was applied.
        """
        logging.getLogger("httpx").setLevel(level)
        logging.getLogger("httpcore").setLevel(level)
        litellm.utils.logging_enabled = enable_litellm_logging
        os.environ["LITELLM_LOG_VERBOSE"] = str(enable_litellm_logging).lower()
        logging.getLogger("litellm").setLevel(logging.ERROR if not enable_litellm_logging else logging.INFO)
        logger.info(f"Configured logging: HTTP libs to {logging.getLevelName(level)}, "
                   f"LiteLLM logging {'enabled' if enable_litellm_logging else 'disabled'}")
        return True 