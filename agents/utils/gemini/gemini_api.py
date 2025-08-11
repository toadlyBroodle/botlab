#!/usr/bin/env python3
"""Gemini API utilities."""

import os
import time
import traceback
import json
from google import genai
from google.genai import types
import google.api_core.exceptions
from typing import Optional, Union, List, Dict, Any, Callable
from dotenv import load_dotenv
from collections import deque
from datetime import datetime
from ...utils.logger_config import setup_logger
from .model_config import GEMINI_MODELS



# Maximum number of characters in a prompt
MAX_PROMPT_CHARS = 30000

def get_llm_postfix(model_name: str) -> str:
    """Get the postfix for the model name"""
    return f"\n*llm: {model_name}*" # this format ensures clients don't render this as a url

class GeminiAPI:
    def __init__(self, default_model: str = 'gemini-2.0-flash', cost_callback: Optional[Callable] = None):
        if default_model not in GEMINI_MODELS:
            raise ValueError(f"Invalid model name. Must be one of: {list(GEMINI_MODELS.keys())}")
            
        self.default_model = default_model
        self.input_token_limit = GEMINI_MODELS[default_model]['input_token_limit']
        self.cost_callback = cost_callback  # Cost callback for real-time cost reporting
        
        # Set up logging - respect console output setting
        disable_console = os.environ.get('DISABLE_CONSOLE_OUTPUT') == '1'
        self.logger = setup_logger('gemini_api', disable_console=disable_console)
        
        # Load environment variables from .env file
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("GEMINI_API_KEY environment variable is not set")
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Load comprehensive tiered pricing data
        json_path = os.path.join(os.path.dirname(__file__), 'gem_llm_info.json')
        
        from .pricing_utils import load_comprehensive_pricing_from_json
        self.comprehensive_pricing = load_comprehensive_pricing_from_json(json_path)
        
        # Cost tracking for cumulative totals
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_cents = 0.0
        self.total_search_cost_cents = 0.0
        self.total_search_count = 0
        self.current_call_cost_info = None
        
        # Add safety settings
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE"
            ),
        ]
        
        # Initialize function registry
        self.registered_functions: Dict[str, Dict[str, Any]] = {}
        
        # Add code execution tool
        self.code_execution_tool = types.Tool(
            code_execution=types.ToolCodeExecution()
        )
        
        # Initialize tools list with code execution
        self.tools = [self.code_execution_tool]
        
        # Rate limiting settings
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 120  # 2 minutes base delay
        self.WAIT_BETWEEN_CALLS = 2  # 2 seconds between calls
        self.RPM_LIMIT = 15  # 15 requests per minute
        self.TPM_LIMIT = 1_000_000  # 1 million tokens per minute
        self.RPD_LIMIT = 1_500  # 1,500 requests per day
        
        # Rate limit tracking
        self._last_call_time = 0
        self.request_times = deque(maxlen=self.RPM_LIMIT)
        self.token_usage = deque(maxlen=self.RPM_LIMIT)
        self.daily_requests = deque(maxlen=self.RPD_LIMIT)
        
        self.logger.info(f"GeminiAPI initialized with model {default_model}")

    def limit_input_tokens(self, text: str, model: Optional[str] = None) -> str:
        """Limit input text to model's token limit"""
        model_name = model or self.default_model
        if model_name in GEMINI_MODELS:
            max_tokens = GEMINI_MODELS[model_name]['input_token_limit']
            # Simple character-based estimation (4 chars ≈ 1 token)
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                self.logger.warning(f"Input text truncated from {len(text)} to {max_chars} characters")
                return text[:max_chars]
        return text

    def get_current_call_cost_info(self) -> Optional[Dict[str, Any]]:
        """Get cost information for the most recent API call"""
        return self.current_call_cost_info

    def get_total_cost_info(self) -> Dict[str, Any]:
        """Get cumulative cost information across all API calls"""
        return {
            'total_cost_cents': self.total_cost_cents,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_search_cost_cents': self.total_search_cost_cents,
            'total_search_count': self.total_search_count
        }

    def _call_cost_callback(self, cost_info: Dict[str, Any]):
        """Call the cost callback if one is set"""
        if self.cost_callback:
            try:
                comprehensive_cost_info = {
                    'current_call': cost_info,
                    'total': self.get_total_cost_info(),
                    'prompt_tokens': self.total_prompt_tokens,
                    'completion_tokens': self.total_completion_tokens,
                    'total_cost_cents': self.total_cost_cents
                }
                self.cost_callback(comprehensive_cost_info)
                self.logger.debug(f"💰 Cost callback called with: {comprehensive_cost_info}")
            except Exception as e:
                self.logger.warning(f"Cost callback failed: {e}")

    def _wait_between_calls(self):
        """Ensure minimum wait between calls"""
        if self._last_call_time > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.WAIT_BETWEEN_CALLS:
                wait_time = self.WAIT_BETWEEN_CALLS - elapsed
                time.sleep(wait_time)
        self._last_call_time = time.time()
        self.request_times.append(datetime.now())

    def _check_rate_limits(self):
        """Check and enforce rate limits"""
        current_time = datetime.now()
        
        # Clean up old entries
        while self.request_times and (current_time - self.request_times[0]).total_seconds() > 60:
            self.request_times.popleft()
            if self.token_usage:
                self.token_usage.popleft()
                
        while self.daily_requests and (current_time - self.daily_requests[0]).total_seconds() > 86400:
            self.daily_requests.popleft()
            
        # Check RPD limit
        if len(self.daily_requests) >= self.RPD_LIMIT:
            wait_time = 86400 - (current_time - self.daily_requests[0]).total_seconds()
            self.logger.warning(f"Daily request limit reached. Wait time: {wait_time/3600:.1f} hours")
            raise Exception("Daily request limit reached")
            
        # Check RPM
        if len(self.request_times) >= self.RPM_LIMIT:
            wait_time = 61 - (current_time - self.request_times[0]).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_times.clear()
                self.token_usage.clear()
                
        # Check TPM
        current_tpm = sum(self.token_usage)
        if current_tpm >= self.TPM_LIMIT:
            wait_time = 61 - (current_time - self.request_times[0]).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Token limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_times.clear()
                self.token_usage.clear()

    def count_tokens(self, text: Union[str, List], model: Optional[str] = None) -> int:
        """Get actual token count using Gemini API's count_tokens method"""
        try:
            contents = types.Part.from_text(text=text) if isinstance(text, str) else text
            response = self.client.models.count_tokens(
                model=(model or self.default_model),
                contents=contents
            )
            return response.total_tokens
        except Exception as e:
            self.logger.error(f"Error counting tokens: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            # Fallback to estimation if API call fails
            if isinstance(text, str):
                return len(text) // 4
            return 0

    def register_function(self, name: str, description: str, parameters: Dict[str, Any], func: Callable, **kwargs) -> None:
        """Register a function that can be called by the model.
        
        Args:
            name: The name of the function
            description: A description of what the function does
            parameters: A dictionary describing the parameters the function accepts
            func: The actual function to call
            **kwargs: Additional arguments to ignore (for compatibility)
        """
        try:
            # Create function schema
            function_schema = {
                "name": name,
                "description": description,
                "parameters": parameters
            }
            
            # Store function details
            self.registered_functions[name] = {
                "schema": function_schema,
                "function": func
            }
            
            # Create or update the function declarations tool
            if not hasattr(self, 'function_declarations_tool'):
                self.function_declarations_tool = types.Tool(
                    function_declarations=[function_schema]
                )
                self.tools.append(self.function_declarations_tool)
            else:
                # Add to existing declarations
                current_declarations = self.function_declarations_tool.function_declarations
                current_declarations.append(function_schema)
                # Update the tool with all declarations
                self.function_declarations_tool = types.Tool(
                    function_declarations=current_declarations
                )
                # Replace in tools list
                for i, tool in enumerate(self.tools):
                    if hasattr(tool, 'function_declarations'):
                        self.tools[i] = self.function_declarations_tool
                        break
                    
        except Exception as e:
            self.logger.error(f"Error registering function {name}: {str(e)}")
            raise

    def _handle_function_call(self, function_call) -> Optional[str]:
        """Handle a function call from the model.
        
        Args:
            function_call: The function call object from the model response
            
        Returns:
            The function response or None if there was an error
        """
        try:
            name = function_call.name
            args = function_call.args
            
            if name not in self.registered_functions:
                self.logger.error(f"Function {name} not found in registered functions")
                return None
                
            # Log the function call with its arguments
            self.logger.info(f"Calling tool: {name} with args: {args}")
                
            func = self.registered_functions[name]["function"]
            result = func(**args)
            
            # Process and format the result
            if result is None:
                self.logger.error(f"Function {name} returned None")
                return None
                
            # Handle different result types
            if isinstance(result, dict):
                # Extract specific fields based on function type
                if "current_time" in result:
                    self.logger.info(f"Tool {name} returned current time")
                    return result["current_time"]
                elif "calc_result" in result:
                    self.logger.info(f"Tool {name} returned calculation")
                    return result["calc_result"]
                elif "user_analysis" in result:
                    success = "error" not in result["user_analysis"].lower()
                    self.logger.info(f"Tool {name} {'succeeded' if success else 'failed'}")
                    return result["user_analysis"]
                elif "meme_url" in result:
                    self.logger.info(f"Tool {name} generated meme")
                    # Return just the URL string
                    return result["meme_url"]
                else:
                    # For other dict results, convert to string
                    self.logger.info(f"Tool {name} returned dictionary")
                    return str(result)
            elif isinstance(result, str):
                # For string results, return as is
                self.logger.info(f"Tool {name} returned: {result[:100]}...")
                return result
            else:
                # For any other type, convert to string
                str_result = str(result)
                self.logger.info(f"Tool {name} returned: {str_result[:100]}...")
                return str_result
            
        except Exception as e:
            self.logger.error(f"Error executing function {name}: {e}")
            return None

    def query(self, 
             prompt: str, 
             model: Optional[str] = None,
             temperature: float = 0.7,
             **kwargs) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Generate content using Gemini API
        
        Args:
            prompt: Input prompt for the model
            model: Optional model name to use (defaults to instance default_model)
            temperature: Temperature for generation (0.0 to 1.0)
            **kwargs: Additional arguments to pass to generate_content
            
        Returns:
            Tuple of (response_text, error_message, cost_info)
            cost_info contains: prompt_tokens, completion_tokens, total_cost_cents, search_cost_cents, search_count
        """
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt: prompt must be a non-empty string")
            return None, "Invalid prompt: prompt must be a non-empty string", None

        model_name = model or self.default_model
        if model_name not in GEMINI_MODELS:
            return None, f"Invalid model name. Must be one of: {list(GEMINI_MODELS.keys())}", None

        prompt = self.limit_input_tokens(prompt, model=model_name)
        input_tokens = self.count_tokens(prompt, model=model_name)

        for attempt in range(self.MAX_RETRIES):
            try:
                self._check_rate_limits()
                self._wait_between_calls()
                
                current_time = datetime.now()
                self.request_times.append(current_time)
                self.daily_requests.append(current_time)

                # Prepare generation config
                config_args = {
                    "temperature": temperature,
                    "safety_settings": self.safety_settings,
                    **kwargs
                }

                # Only include tools if we have registered functions AND the model supports function calling
                if self.registered_functions and GEMINI_MODELS[model_name]['supports_function_calling']:
                    config_args["tools"] = self.tools

                config = types.GenerateContentConfig(**config_args)

                # Generate content
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=types.Part.from_text(text=prompt),
                    config=config
                )
                
                # Initialize cost tracking variables
                prompt_tokens = 0
                completion_tokens = 0
                total_cost_cents = 0
                search_cost_cents = 0
                search_count = 0
                
                # Extract token counts and calculate costs from response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) or getattr(usage, 'promptTokenCount', 0)
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'candidatesTokenCount', 0)
                    
                    # Calculate cost using tiered pricing
                    if model_name in self.comprehensive_pricing:
                        from .pricing_utils import calculate_tiered_cost
                        model_pricing = self.comprehensive_pricing[model_name]
                        total_cost_cents, tier_description = calculate_tiered_cost(
                            model_pricing, prompt_tokens, completion_tokens
                        )
                        
                        self.logger.debug(f"💰 API Response cost using {model_pricing.pricing_type} pricing "
                                        f"(tier: {tier_description}): {prompt_tokens} prompt + {completion_tokens} completion tokens = ${total_cost_cents/100:.6f} USD")
                    else:
                        self.logger.warning(f"💰 No pricing information available for model {model_name}")
                        total_cost_cents = 0.0
                    
                    # Extract search context cost if present
                    if hasattr(usage, 'search_context_cost_per_query'):
                        search_cost_cents = getattr(usage, 'search_context_cost_per_query', 0)
                    
                    # Update tracking for actual tokens used
                    actual_tokens = getattr(usage, 'total_token_count', 0) or getattr(usage, 'totalTokenCount', 0)
                    if actual_tokens > 0:
                        if self.token_usage:
                            self.token_usage.pop()
                        self.token_usage.append(actual_tokens)
                
                # Prepare cost info to return
                cost_info = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_cost_cents': total_cost_cents,
                    'search_cost_cents': search_cost_cents,
                    'search_count': search_count,
                    'model': model_name
                } if prompt_tokens > 0 or completion_tokens > 0 or total_cost_cents > 0 else None
                
                # Update cumulative totals
                if cost_info:
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_cost_cents += total_cost_cents
                    self.total_search_cost_cents += search_cost_cents
                    self.total_search_count += search_count
                    self.current_call_cost_info = cost_info
                    
                    # Call cost callback if set
                    self._call_cost_callback(cost_info)
                
                if hasattr(response, 'usage_metadata'):
                    actual_tokens = response.usage_metadata.total_token_count
                    if self.token_usage:
                        self.token_usage.pop()
                        self.token_usage.append(actual_tokens)

                # Handle responses
                if hasattr(response, 'candidates') and response.candidates:
                    result_parts = []
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    if part.text is not None:
                                        result_parts.append(part.text)
                                elif hasattr(part, 'executable_code') and part.executable_code:
                                    code = part.executable_code.code if hasattr(part.executable_code, 'code') else ''
                                    language = part.executable_code.language if hasattr(part.executable_code, 'language') else 'python'
                                    if code:
                                        result_parts.append(f"```{language}\n{code}\n```")
                                elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                                    if hasattr(part.code_execution_result, 'output') and part.code_execution_result.output:
                                        result_parts.append(f"`{part.code_execution_result.output}`")
                                    if hasattr(part.code_execution_result, 'error') and part.code_execution_result.error:
                                        error_msg = part.code_execution_result.error
                                        self.logger.error(f"Code execution error: {error_msg}")
                                        return None, "Error: Code execution failed", cost_info
                                    if hasattr(part.code_execution_result, 'outcome'):
                                        self.logger.debug(f"\nOutcome: {part.code_execution_result.outcome}")
                                elif hasattr(part, 'function_call') and part.function_call:
                                    # Handle function calls
                                    function_response = self._handle_function_call(part.function_call)
                                    if function_response is not None:
                                        result_parts.append(str(function_response))
                                    else:
                                        return None, "Error: Function call failed", cost_info
                                elif hasattr(part, 'function_response') and part.function_response:
                                    if hasattr(part.function_response, 'output') and part.function_response.output is not None:
                                        result_parts.append(f"`{part.function_response.output}`")
                                    elif hasattr(part.function_response, 'response') and part.function_response.response is not None:
                                        result_parts.append(f"`{part.function_response.response}`")

                    response_text = "\n".join(result_parts) if result_parts else None
                    if response_text:
                        return response_text, None, cost_info
                    else:
                        return None, "No response content found", cost_info

                return None, "No valid response from model", cost_info

            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    error_msg = f"API call failed after {self.MAX_RETRIES} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    return None, error_msg, None
                else:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)

        return None, "Max retries exceeded", None
    
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
                'model': self.default_model
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
                # For GeminiAPI, we need to structure the cost info similar to what's expected
                cost_info = {
                    'current_call': self.current_call_cost_info,
                    'total': {
                        'total_cost_cents': self.total_cost_cents,
                        'total_prompt_tokens': self.total_prompt_tokens,
                        'total_completion_tokens': self.total_completion_tokens,
                        'total_search_cost_cents': self.total_search_cost_cents,
                        'total_search_count': self.total_search_count
                    }
                }
                self.cost_callback(cost_info)
                self.logger.debug(f"💰 Called cost callback after search cost addition: ${search_cost_cents:.3f} cents")
            except Exception as e:
                self.logger.warning(f"Cost callback failed after search cost addition: {e}")
        
        self.logger.debug(f"💰 Added search cost: ${search_cost_cents:.3f} cents, count: {search_count}")
        self.logger.debug(f"💰 Total search cost: ${self.total_search_cost_cents:.3f} cents, total count: {self.total_search_count}")
    
    def get_model_cost_info(self):
        """Get comprehensive cost information from the GeminiAPI model.
        
        Returns:
            Dictionary with current and total cost information
        """
        return {
            'current_call': self.current_call_cost_info,
            'total': {
                'total_cost_cents': self.total_cost_cents,
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_search_cost_cents': self.total_search_cost_cents,
                'total_search_count': self.total_search_count
            },
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_cost_cents': self.total_cost_cents,
            'search_cost_cents': self.total_search_cost_cents,
            'search_count': self.total_search_count
        }