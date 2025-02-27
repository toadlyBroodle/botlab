import time
import random
import json
import os
from collections import deque
from datetime import datetime, timedelta
from smolagents import LiteLLMModel
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from typing import List, Dict, Optional, Any, Tuple, Union
import threading
import logging
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
litellm.utils.logging_enabled = False  # Disable default logging
os.environ["LITELLM_LOG_VERBOSE"] = "False"  # Disable verbose logging

# Add imports for additional error types
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
import litellm.exceptions
import traceback

logger = logging.getLogger(__name__)

class SharedRateLimitTracker:
    """Singleton class to track rate limits across all model instances"""
    _instance = None
    _lock = threading.RLock()  # Reentrant lock for thread safety
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedRateLimitTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if self._initialized:
                return
                
            # Initialize tracking dictionaries
            self._model_limits = {}  # Stores limits for each model_id
            self._request_timestamps = {}  # Timestamps of requests for RPM/RPD tracking
            self._token_usage = {}  # Token usage for TPM tracking
            self._initialized = True
            logger.info("Initialized SharedRateLimitTracker")
    
    def initialize_model(self, model_id: str, rpm_limit: int, tpm_limit: int, rpd_limit: int):
        """Initialize tracking for a specific model
        
        Args:
            model_id: The model identifier
            rpm_limit: Requests per minute limit
            tpm_limit: Tokens per minute limit
            rpd_limit: Requests per day limit
        """
        with self._lock:
            # Store the limits for this model
            self._model_limits[model_id] = {
                'rpm': rpm_limit,
                'tpm': tpm_limit,
                'rpd': rpd_limit
            }
            
            # Initialize tracking for this model if not already present
            if model_id not in self._request_timestamps:
                self._request_timestamps[model_id] = []
            
            if model_id not in self._token_usage:
                self._token_usage[model_id] = []
                
            logger.info(f"Initialized rate limit tracking for model {model_id} with limits: "
                       f"RPM={rpm_limit}, TPM={tpm_limit}, RPD={rpd_limit}")
    
    def update_tracking(self, model_id: str, input_tokens: int, output_tokens: int):
        """Update tracking after an API call
        
        Args:
            model_id: The model identifier
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        with self._lock:
            current_time = time.time()
            
            # Ensure model is initialized
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker")
                return
                
            # Update request timestamps
            self._request_timestamps[model_id].append(current_time)
            
            # Update token usage
            total_tokens = input_tokens + output_tokens
            self._token_usage[model_id].append((current_time, total_tokens))
            
            # Clean up old data
            self._clean_old_data(model_id)
            
            logger.debug(f"Updated rate limit tracking for {model_id}: "
                        f"+{total_tokens} tokens ({input_tokens} in, {output_tokens} out)")
    
    def _clean_old_data(self, model_id: str):
        """Clean up data older than tracking windows
        
        Args:
            model_id: The model identifier
        """
        current_time = time.time()
        minute_ago = current_time - 60
        day_ago = current_time - 86400  # 24 hours in seconds
        
        # Clean up request timestamps
        self._request_timestamps[model_id] = [
            ts for ts in self._request_timestamps[model_id] if ts > day_ago
        ]
        
        # Clean up token usage
        self._token_usage[model_id] = [
            (ts, tokens) for ts, tokens in self._token_usage[model_id] if ts > minute_ago
        ]
    
    def check_rate_limits(self, model_id: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if any rate limits would be exceeded
        
        Args:
            model_id: The model identifier
            
        Returns:
            Tuple of (would_exceed, limit_type, wait_time)
        """
        with self._lock:
            # Ensure model is initialized
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker")
                return False, None, None
                
            current_time = time.time()
            minute_ago = current_time - 60
            day_ago = current_time - 86400  # 24 hours in seconds
            
            # Check RPM limit
            rpm_limit = self._model_limits[model_id]['rpm']
            rpm_count = sum(1 for ts in self._request_timestamps[model_id] if ts > minute_ago)
            
            if rpm_count >= rpm_limit:
                # Calculate wait time - find oldest timestamp within the minute window
                minute_timestamps = [ts for ts in self._request_timestamps[model_id] if ts > minute_ago]
                if minute_timestamps:
                    oldest_ts = min(minute_timestamps)
                    wait_time = 60 - (current_time - oldest_ts) + 0.1  # Add a small buffer
                else:
                    wait_time = 0.5  # Fallback
                return True, "rpm", wait_time
            
            # Check TPM limit
            tpm_limit = self._model_limits[model_id]['tpm']
            tpm_count = sum(tokens for ts, tokens in self._token_usage[model_id] if ts > minute_ago)
            
            if tpm_count >= tpm_limit:
                # Calculate wait time based on token usage timestamps
                minute_token_usage = [(ts, tokens) for ts, tokens in self._token_usage[model_id] if ts > minute_ago]
                if minute_token_usage:
                    oldest_ts = min(ts for ts, _ in minute_token_usage)
                    wait_time = 60 - (current_time - oldest_ts) + 0.1  # Add a small buffer
                else:
                    wait_time = 0.5  # Fallback
                return True, "tpm", wait_time
            
            # Check RPD limit
            rpd_limit = self._model_limits[model_id]['rpd']
            rpd_count = len([ts for ts in self._request_timestamps[model_id] if ts > day_ago])
            
            if rpd_count >= rpd_limit:
                # Calculate wait time based on daily usage
                day_timestamps = [ts for ts in self._request_timestamps[model_id] if ts > day_ago]
                if day_timestamps:
                    oldest_ts = min(day_timestamps)
                    wait_time = 86400 - (current_time - oldest_ts) + 0.1  # Add a small buffer
                else:
                    wait_time = 0.5  # Fallback
                return True, "rpd", wait_time
            
            return False, None, None
    
    def get_rate_limit_status(self, model_id: str) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get the current rate limit status for a model
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dictionary with rate limit status information
        """
        with self._lock:
            # Ensure model is initialized
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker")
                return {
                    'rpm': {'limit': 0, 'usage': 0, 'percentage': 0},
                    'tpm': {'limit': 0, 'usage': 0, 'percentage': 0},
                    'rpd': {'limit': 0, 'usage': 0, 'percentage': 0}
                }
                
            current_time = time.time()
            minute_ago = current_time - 60
            day_ago = current_time - 86400  # 24 hours in seconds
            
            # Calculate current usage
            rpm_limit = self._model_limits[model_id]['rpm']
            rpm_usage = sum(1 for ts in self._request_timestamps[model_id] if ts > minute_ago)
            rpm_percentage = (rpm_usage / rpm_limit * 100) if rpm_limit > 0 else 0
            
            tpm_limit = self._model_limits[model_id]['tpm']
            tpm_usage = sum(tokens for ts, tokens in self._token_usage[model_id] if ts > minute_ago)
            tpm_percentage = (tpm_usage / tpm_limit * 100) if tpm_limit > 0 else 0
            
            rpd_limit = self._model_limits[model_id]['rpd']
            rpd_usage = sum(1 for ts in self._request_timestamps[model_id] if ts > day_ago)
            rpd_percentage = (rpd_usage / rpd_limit * 100) if rpd_limit > 0 else 0
            
            return {
                'rpm': {'limit': rpm_limit, 'usage': rpm_usage, 'percentage': rpm_percentage},
                'tpm': {'limit': tpm_limit, 'usage': tpm_usage, 'percentage': tpm_percentage},
                'rpd': {'limit': rpd_limit, 'usage': rpd_usage, 'percentage': rpd_percentage}
            }

class RateLimitedLiteLLMModel(LiteLLMModel):
    """A wrapper around LiteLLMModel that adds rate limiting functionality."""
    
    # get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    parent_dir = os.path.dirname(current_dir)
    # get the gem_llm_info.json file
    gem_llm_info_path = os.path.join(current_dir, "gem_llm_info.json")

    def __init__(
        self,
        model_id: str,
        model_info_path: str = gem_llm_info_path,
        base_wait_time: float = 1.0,
        max_retries: int = 3,
        jitter_factor: float = 0.1,
        **kwargs
    ):
        """Initialize the rate-limited model.
        
        Args:
            model_id: The model identifier to use with LiteLLM
            model_info_path: Optional path to the JSON file containing model information
            base_wait_time: Optional base wait time in seconds for rate limit backoff
            max_retries: Optional maximum number of retries for rate limit errors
            jitter_factor: Optional factor to apply random jitter to wait times (0-1)
            **kwargs: Additional arguments to pass to LiteLLMModel
        """
        # Ensure LiteLLM logging is disabled before initializing
        litellm.utils.logging_enabled = False
        
        # Initialize the parent class
        super().__init__(model_id=model_id, **kwargs)
        
        self.model_id = model_id
        self.base_wait_time = base_wait_time
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        
        # Add counter for tracking API calls
        self.api_call_count = 0
        
        # Load model information from JSON file
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            # Get model-specific information
            model_key = model_id.split('/')[-1]  # Extract model name from model_id
            if model_key not in model_info:
                logger.warning(f"Model {model_key} not found in model info file. Using default limits.")
                model_key = "default"
                
            model_data = model_info.get(model_key, model_info.get("default", {}))
            
            # Set rate limits
            self.rpm_limit = model_data.get("rpm_limit", 60)
            self.tpm_limit = model_data.get("tpm_limit", 60000)
            self.rpd_limit = model_data.get("rpd_limit", 10000)
            self.input_token_limit = model_data.get("input_token_limit", 32000)
            
            logger.info(f"Initialized rate-limited model {model_id} with limits: "
                       f"RPM={self.rpm_limit}, TPM={self.tpm_limit}, RPD={self.rpd_limit}")
            
        except Exception as e:
            logger.error(f"Error loading model info from {model_info_path}: {e}")
            # Set default limits
            self.rpm_limit = 60
            self.tpm_limit = 60000
            self.rpd_limit = 10000
            self.input_token_limit = 32000
        
        # Initialize shared rate limit tracker
        self.shared_tracker = SharedRateLimitTracker()
        self.shared_tracker.initialize_model(
            model_id=self.model_id,
            rpm_limit=self.rpm_limit,
            tpm_limit=self.tpm_limit,
            rpd_limit=self.rpd_limit
        )
        
        # For tracking the last API call
        self.last_api_call_time = 0
        self.last_wait_time = 0
        self.last_error = None
        
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is related to rate limiting
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error is related to rate limiting, False otherwise
        """
        # Log the error type and message for debugging
        error_type = type(error).__name__
        error_str = str(error)
        
        # Check for LiteLLM's RateLimitError
        if isinstance(error, RateLimitError):
            return True
            
        # Check for ServiceUnavailableError which can sometimes be rate limiting
        if isinstance(error, ServiceUnavailableError):
            return True
            
        # Check for Google API specific exceptions
        if isinstance(error, ResourceExhausted):
            return True
        if isinstance(error, ServiceUnavailable):
            return True
        if isinstance(error, GoogleAPIError):
            error_str_lower = error_str.lower()
            if any(term in error_str_lower for term in ["quota", "rate", "limit", "429"]):
                return True
        
        # Check for LiteLLM API errors
        if isinstance(error, litellm.exceptions.APIError):
            error_str_lower = error_str.lower()
            if any(term in error_str_lower for term in ["rate", "limit", "quota", "429"]):
                return True
                
        # Check specifically for VertexAIException pattern
        if "VertexAIException" in error_str:
            if "RESOURCE_EXHAUSTED" in error_str:
                return True
            if "code\": 429" in error_str or "'code': 429" in error_str:
                return True
            if "Resource has been exhausted" in error_str:
                return True
        
        # Check for the exact error pattern from the logs
        if "HTTP/1.1 429 Too Many Requests" in error_str:
            return True
            
        # Check error message for rate limit related terms
        error_str_lower = error_str.lower()
        rate_limit_terms = ["rate limit", "quota", "too many requests", "429", "resource exhausted", "resource has been exhausted"]
        for term in rate_limit_terms:
            if term in error_str_lower:
                return True
            
        # Check for nested status code in exception
        try:
            if hasattr(error, 'status_code') and error.status_code == 429:
                return True
            if hasattr(error, 'code') and error.code == 429:
                return True
            if hasattr(error, 'response') and hasattr(error.response, 'status_code') and error.response.status_code == 429:
                return True
            
            # Check for VertexAIException structure
            if hasattr(error, 'error') and isinstance(error.error, dict):
                if error.error.get('code') == 429:
                    return True
                if error.error.get('status') == 'RESOURCE_EXHAUSTED':
                    return True
        except Exception as e:
            logger.debug(f"Exception while checking error attributes: {e}")
            pass
            
        return False
        
    def _apply_rate_limit(self, estimated_tokens: int = 0) -> Tuple[bool, float]:
        """Check rate limits and apply waiting if needed
        
        Args:
            estimated_tokens: Estimated number of tokens for the request
            
        Returns:
            Tuple of (waited, wait_time)
        """
        # Check if we would exceed any rate limits
        would_exceed, limit_type, wait_time = self.shared_tracker.check_rate_limits(self.model_id)
        
        if would_exceed:
            logger.info(f"Rate limit ({limit_type}) would be exceeded. Waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            self.last_wait_time = wait_time
            return True, wait_time
            
        return False, 0
        
    def _update_rate_limit_tracking(self, input_tokens: int, output_tokens: int):
        """Update rate limit tracking after a successful API call
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        self.shared_tracker.update_tracking(
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self.last_api_call_time = time.time()
        
    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Override the __call__ method to add rate limiting
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The model response
        """
        # Make a simple token estimate for pre-call rate limiting
        # This is just a rough estimate to prevent obvious rate limit errors
        estimated_input_tokens = min(
            sum(len(m.get("content", "")) for m in messages) // 4,
            self.input_token_limit
        )
        
        # Apply rate limiting before the call
        waited, wait_time = self._apply_rate_limit(estimated_input_tokens)
        if waited:
            logger.info(f"Waited {wait_time:.2f}s before API call due to rate limit")
            
        # Try the API call with retries for rate limit errors
        retries = 0
        backoff_time = self.base_wait_time
        
        while True:
            try:
                # Make the API call
                response = super().__call__(messages=messages, **kwargs)
                
                # Get actual token counts from the model
                token_counts = self.get_token_counts()
                input_tokens = token_counts.get('input_token_count', estimated_input_tokens)
                output_tokens = token_counts.get('output_token_count', 0)
                total_tokens = input_tokens + output_tokens
                
                # Update rate limit tracking
                self._update_rate_limit_tracking(input_tokens, output_tokens)
                
                # Increment API call counter
                self.api_call_count += 1
                
                # Print rate limit status every 3 calls
                if self.api_call_count % 3 == 0:
                    logger.info(f"Automatic rate limit status check (call #{self.api_call_count}):")
                    self.print_rate_limit_status(use_logger=True)
                
                # Log token usage
                logger.info(f"API call successful. Used {total_tokens} tokens "
                           f"({input_tokens} input, {output_tokens} output)")
                
                return response
                
            except Exception as e:
                self.last_error = e
                
                # Log the full error for debugging
                logger.debug(f"API call error: {type(e).__name__}: {str(e)}")
                
                # Check if this is a rate limit error
                is_rate_limit = self._is_rate_limit_error(e)
                if is_rate_limit:
                    retries += 1
                    
                    if retries > self.max_retries:
                        logger.error(f"Maximum retries ({self.max_retries}) exceeded for rate limit error: {e}")
                        raise
                        
                    # Calculate backoff time with jitter
                    jitter = self.jitter_factor * backoff_time * (2 * (0.5 - random.random()))
                    wait_time = backoff_time + jitter
                    
                    logger.warning(f"Rate limit error: {e}. Retry {retries}/{self.max_retries} "
                                  f"after {wait_time:.2f}s backoff")
                    
                    time.sleep(wait_time)
                    backoff_time *= 2  # Exponential backoff
                    
                    # Check rate limits again after waiting
                    waited, additional_wait = self._apply_rate_limit(estimated_input_tokens)
                    if waited:
                        logger.info(f"Additional wait of {additional_wait:.2f}s after backoff")
                else:
                    # Not a rate limit error, re-raise
                    logger.error(f"API call failed with non-rate-limit error: {e}")
                    raise
    
    def get_rate_limit_status(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get the current rate limit status
        
        Returns:
            Dictionary with rate limit status information
        """
        return self.shared_tracker.get_rate_limit_status(self.model_id)
        
    def print_rate_limit_status(self, use_logger=False):
        """Print the current rate limit status to the console or log
        
        Args:
            use_logger: If True, output to logger instead of console
        """
        status = self.get_rate_limit_status()
        
        # Format the status message
        status_lines = [
            "Rate Limit Status:",
            f"  RPM: {status['rpm']['usage']}/{status['rpm']['limit']} ({status['rpm']['percentage']:.1f}%)",
            f"  TPM: {status['tpm']['usage']}/{status['tpm']['limit']} ({status['tpm']['percentage']:.1f}%)",
            f"  RPD: {status['rpd']['usage']}/{status['rpd']['limit']} ({status['rpd']['percentage']:.1f}%)"
        ]
        
        # Add warning indicators for high usage
        if status['rpm']['percentage'] > 80:
            status_lines.append("  ⚠️ RPM usage is high!")
        if status['tpm']['percentage'] > 80:
            status_lines.append("  ⚠️ TPM usage is high!")
        if status['rpd']['percentage'] > 80:
            status_lines.append("  ⚠️ RPD usage is high!")
        
        # Output to logger or console
        if use_logger:
            logger.info("\n".join(status_lines))
        else:
            print("\n" + "\n".join(status_lines))
            
    @staticmethod
    def configure_logging(level=logging.WARNING, enable_litellm_logging=False):
        """Configure logging levels for HTTP libraries and LiteLLM
        
        Args:
            level: Logging level for HTTP libraries (default: WARNING to hide API keys)
            enable_litellm_logging: Whether to enable LiteLLM's internal logging
        """
        # Configure HTTP library logging
        logging.getLogger("httpx").setLevel(level)
        logging.getLogger("httpcore").setLevel(level)
        
        # Configure LiteLLM logging
        litellm.utils.logging_enabled = enable_litellm_logging
        os.environ["LITELLM_LOG_VERBOSE"] = str(enable_litellm_logging).lower()
        
        logger.info(f"Configured logging: HTTP libraries set to {logging.getLevelName(level)}, "
                   f"LiteLLM logging {'enabled' if enable_litellm_logging else 'disabled'}")
                   
        return True