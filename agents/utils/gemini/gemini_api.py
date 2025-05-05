#!/usr/bin/env python3
"""Gemini API utilities."""

import os
import time
import traceback
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
    def __init__(self, default_model: str = 'gemini-2.0-flash'):
        if default_model not in GEMINI_MODELS:
            raise ValueError(f"Invalid model name. Must be one of: {list(GEMINI_MODELS.keys())}")
            
        self.default_model = default_model
        self.input_token_limit = GEMINI_MODELS[default_model]['input_token_limit']
        
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
             **kwargs) -> tuple[Optional[str], Optional[str]]:
        """Generate content using Gemini API
        
        Args:
            prompt: Input prompt for the model
            model: Optional model name to use (defaults to instance default_model)
            temperature: Temperature for generation (0.0 to 1.0)
            **kwargs: Additional arguments to pass to generate_content
            
        Returns:
            Tuple of (response_text, error_message)
        """
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt: prompt must be a non-empty string")
            return None, "Invalid prompt: prompt must be a non-empty string"

        model_name = model or self.default_model
        if model_name not in GEMINI_MODELS:
            return None, f"Invalid model name. Must be one of: {list(GEMINI_MODELS.keys())}"

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
                                        return None, "Error: Code execution failed"
                                    if hasattr(part.code_execution_result, 'outcome'):
                                        self.logger.debug(f"\nOutcome: {part.code_execution_result.outcome}")
                                elif hasattr(part, 'function_call') and part.function_call:
                                    # Handle function calls
                                    function_response = self._handle_function_call(part.function_call)
                                    if function_response is not None:
                                        result_parts.append(str(function_response))
                                    else:
                                        return None, "Error: Function call failed"
                                elif hasattr(part, 'function_response') and part.function_response:
                                    if hasattr(part.function_response, 'output') and part.function_response.output is not None:
                                        result_parts.append(f"`{part.function_response.output}`")
                                    elif hasattr(part.function_response, 'response') and part.function_response.response is not None:
                                        result_parts.append(f"`{part.function_response.response}`")
                                    if hasattr(part.function_response, 'error') and part.function_response.error:
                                        error_msg = part.function_response.error
                                        self.logger.error(f"Function execution error: {error_msg}")
                                        return None, "Error: Function execution failed"
                    
                    if result_parts:
                        result = "\n".join(filter(None, result_parts))
                        if result.strip():
                            return f"{result}\n{get_llm_postfix(model_name)}", None

                # Handle regular text response
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_msg = f"Content blocked: {response.prompt_feedback.block_reason}"
                    self.logger.error(error_msg)
                    return None, "Error: Content blocked"

                # Extract text from parts if direct text access fails
                try:
                    if hasattr(response, 'text'):
                        text = response.text
                        if text and text.strip():
                            return f"{text.strip()}\n{get_llm_postfix(model_name)}", None
                            
                    if hasattr(response, 'parts'):
                        text_parts = []
                        for part in response.parts:
                            if hasattr(part, 'text') and part.text is not None:
                                text_parts.append(part.text)
                        if text_parts:
                            text = ' '.join(filter(None, text_parts)).strip()
                            if text:
                                return f"{text}\n{get_llm_postfix(model_name)}", None
                    
                    # If we get here and have no content, return an error
                    return None, "Error: No valid content in response"
                    
                except AttributeError as e:
                    self.logger.error(f"Failed to extract text from response: {str(e)}")
                    return None, "Error: Failed to extract text from response"

            except google.api_core.exceptions.ResourceExhausted as e:
                self.logger.warning(f"Rate limit exceeded: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                
                # If not already using gemlite, try falling back to it
                if model_name != "gemini-2.0-flash-lite":
                    self.logger.info("Attempting fallback to gemlite model...")
                    try:
                        return self.query(prompt=prompt, model="gemini-2.0-flash-lite", temperature=temperature, **kwargs)
                    except Exception as fallback_e:
                        self.logger.error(f"Fallback to gemlite failed: {fallback_e}")
                
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (attempt + 1)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return None, "Error: Rate limit exceeded"
                    
            except google.api_core.exceptions.InvalidArgument as e:
                self.logger.error(f"Invalid argument error: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                return None, "Error: Invalid argument"
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (attempt + 1)
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return None, "Error: Unexpected error occurred"
        
        return None, "Error: Max retries exceeded"

    def limit_input_tokens(self, text: str, max_tokens: Optional[int] = None, model: Optional[str] = None) -> str:
        """Limit input text to stay within token limits"""
        if model and model in GEMINI_MODELS:
            token_limit = GEMINI_MODELS[model]['input_token_limit']
        else:
            token_limit = self.input_token_limit
            
        if max_tokens is not None:
            token_limit = min(token_limit, max_tokens)
            
        current_tokens = self.count_tokens(text, model=model)
        
        if current_tokens <= token_limit:
            return text
            
        keep_ratio = token_limit / current_tokens
        char_limit = int(len(text) * keep_ratio)
        
        truncated = text[:char_limit]
        last_period = truncated.rfind('.')
        
        if last_period > 0:
            truncated = truncated[:last_period + 1]
        
        self.logger.debug(f"Truncated text from {current_tokens} to ~{self.count_tokens(truncated, model=model)} tokens")
        return truncated
