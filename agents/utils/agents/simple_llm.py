import json
import os
import logging
import time
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


class SimpleLiteLLMModel(LiteLLMModel):
    """A simplified wrapper around `smolagents.LiteLLMModel` with cost tracking and basic retry logic.

    Features:
    - Direct LiteLLM API calls without rate limiting
    - Exponential backoff retry logic for "model is overloaded" errors (up to 4 retries: 1s, 2s, 4s, 8s)
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
        """Initializes the simple LiteLLM model wrapper."""
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
        
        # Load comprehensive tiered pricing data
        if model_info_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_info_path = os.path.join(os.path.dirname(current_dir), "gemini", "gem_llm_info.json")
        
        from ..gemini.pricing_utils import load_comprehensive_pricing_from_json
        self.comprehensive_pricing = load_comprehensive_pricing_from_json(model_info_path)
        
        # Load model configuration if available
        self.input_token_limit = self._load_model_config(model_info_path)
        
        logger.info(f"SimpleLiteLLMModel initialized with model {model_id}")

    def _load_model_config(self, model_info_path: str) -> int:
        """Load model configuration from JSON file."""
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
        """Makes an API call using the current model with cost tracking and exponential backoff retry logic."""
        # Normalize messages to plain dicts and roles to supported strings
        def _normalize_role(role_value: Any) -> str:
            # Extract enum .value if present, then map to allowed set
            raw = getattr(role_value, "value", role_value)
            raw_str = str(raw).lower()
            # Handle cases like "messagerole.system" or "system"
            if "tool" in raw_str and ("response" in raw_str or "result" in raw_str):
                return "tool-response"
            if "tool" in raw_str and ("call" in raw_str or "invoke" in raw_str):
                return "tool-call"
            if "system" in raw_str:
                return "system"
            if "assistant" in raw_str:
                return "assistant"
            if "user" in raw_str:
                return "user"
            # Default to user
            return "user"

        def _to_message_dict(msg: Any) -> Dict[str, str]:
            # If it's already a dict, coerce values to strings as needed
            if isinstance(msg, dict):
                role_val = msg.get("role", "user")
                role = _normalize_role(role_val)
                content_val = msg.get("content", "")
                content = str(getattr(content_val, "content", content_val))
                return {"role": role, "content": content}
            # If it has attributes like a SimpleNamespace or Pydantic model
            role_attr = getattr(msg, "role", "user")
            role = _normalize_role(role_attr)
            content_attr = getattr(msg, "content", "")
            content = str(getattr(content_attr, "content", content_attr))
            return {"role": role, "content": content}

        messages_normalized: List[Dict[str, str]] = [_to_message_dict(m) for m in messages]

        # Estimate input tokens
        estimated_input_tokens = min(
            sum(len(m.get("content", "")) for m in messages_normalized) // 4,
            self.input_token_limit
        )
        
        # Set default timeout if not provided
        current_call_kwargs = kwargs.copy()
        if "timeout" not in current_call_kwargs:
            current_call_kwargs["timeout"] = 120
        
        # Retry logic for "model is overloaded" errors with exponential backoff
        max_retries = 4
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Make the API call using the parent's generate method directly - NO RATE LIMITING
                response = super().generate(messages=messages_normalized, **current_call_kwargs)
                
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
                
                if attempt > 0:
                    logger.info(f"API call successful with model {self.model_id} on attempt {attempt + 1}. "
                               f"Tokens: {input_tokens + output_tokens}, "
                               f"Cost: ${self.current_call_cost_info['total_cost_cents']/100:.6f} USD")
                else:
                    logger.info(f"API call successful with model {self.model_id}. "
                               f"Tokens: {input_tokens + output_tokens}, "
                               f"Cost: ${self.current_call_cost_info['total_cost_cents']/100:.6f} USD")
                
                return response
                
            except Exception as e:
                # Check if this is a "model is overloaded" error
                error_message = str(e).lower()
                is_overloaded_error = (
                    "model is overloaded" in error_message or
                    "overloaded" in error_message or
                    ("code\": 503" in str(e) and "unavailable" in error_message)
                )
                
                if is_overloaded_error and attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s
                    retry_delay = base_delay * (2 ** attempt)
                    logger.warning(f"Model overloaded error detected (attempt {attempt + 1}/{max_retries + 1}). "
                                  f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Either not an overloaded error, or we've exhausted retries
                    if attempt > 0:
                        logger.error(f"API call failed with model {self.model_id} after {attempt + 1} attempts: "
                                   f"{type(e).__name__}: {str(e)}")
                    else:
                        logger.error(f"API call failed with model {self.model_id}: {type(e).__name__}: {str(e)}")
                    raise

    def _extract_token_counts(self, response: Any, estimated_input_tokens: int) -> tuple[int, int]:
        """Extract token counts from the API response."""
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
        """Override the generate method to ensure smolagents compatibility."""
        logger.debug(f"📊 SimpleLiteLLMModel.generate called with {len(messages)} messages")
        response = self.__call__(messages=messages, **kwargs)
        logger.debug(f"📊 SimpleLiteLLMModel.generate completed")
        return response

    def calculate_call_cost(self, input_tokens: int, output_tokens: int, model_id: str = None) -> Dict[str, Any]:
        """Calculate the cost for a single API call using tiered pricing system."""
        from ..gemini.pricing_utils import calculate_tiered_cost
        
        if model_id is None:
            model_id = self.model_id
            
        # Remove gemini/ prefix if present for cost lookup
        model_key = model_id.replace('gemini/', '')
        
        cost_info = {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_cost_cents': 0.0,
            'model': model_key,
            'pricing_tier': 'unknown'
        }
        
        # Use comprehensive tiered pricing
        if model_key in self.comprehensive_pricing:
            model_pricing = self.comprehensive_pricing[model_key]
            total_cost_cents, tier_description = calculate_tiered_cost(
                model_pricing, input_tokens, output_tokens
            )
            cost_info['total_cost_cents'] = total_cost_cents
            cost_info['pricing_tier'] = tier_description
            
            logger.debug(f"💰 Calculated cost using {model_pricing.pricing_type} pricing "
                        f"(tier: {tier_description}): ${total_cost_cents/100:.6f} USD")
            
        else:
            logger.error(f"💰 No cost information available for model {model_key}. "
                        f"Available models: {list(self.comprehensive_pricing.keys())}")
            
        return cost_info

    def get_current_call_cost_info(self) -> Optional[Dict[str, Any]]:
        """Get cost information for the most recent API call."""
        return self.current_call_cost_info

    def get_total_cost_info(self) -> Dict[str, Any]:
        """Get cumulative cost information across all API calls."""
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
        """Add search cost tracking when web search tools are used."""
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
        """Print or log a summary of costs and usage."""
        total_info = self.get_total_cost_info()
        summary_lines = [
            f"Cost Summary for {self.model_id}:",
            f"  Total API Calls: {total_info['api_call_count']}",
            f"  Total Prompt Tokens: {total_info['total_prompt_tokens']:,}",
            f"  Total Completion Tokens: {total_info['total_completion_tokens']:,}",
            f"  Total Cost: ${total_info['total_cost_cents']/100:.6f} USD"
        ]
        
        if total_info['api_call_count'] > 0:
            avg_cost = total_info['total_cost_cents'] / total_info['api_call_count']
            avg_tokens = (total_info['total_prompt_tokens'] + total_info['total_completion_tokens']) / total_info['api_call_count']
            summary_lines.extend([
                f"  Average Cost per Call: ${avg_cost/100:.6f} USD",
                f"  Average Tokens per Call: {avg_tokens:.1f}"
            ])
        
        output_func = logger.info if use_logger else print
        output_func("\n".join(summary_lines))
        
    @staticmethod
    def configure_logging(level=logging.WARNING, enable_litellm_logging=False):
        """Configure global logging levels for HTTPX, HTTPCore, and LiteLLM."""
        logging.getLogger("httpx").setLevel(level)
        logging.getLogger("httpcore").setLevel(level)
        litellm.utils.logging_enabled = enable_litellm_logging
        os.environ["LITELLM_LOG_VERBOSE"] = str(enable_litellm_logging).lower()
        logging.getLogger("litellm").setLevel(logging.ERROR if not enable_litellm_logging else logging.INFO)
        logger.info(f"Configured logging: HTTP libs to {logging.getLevelName(level)}, "
                   f"LiteLLM logging {'enabled' if enable_litellm_logging else 'disabled'}")
        return True 


