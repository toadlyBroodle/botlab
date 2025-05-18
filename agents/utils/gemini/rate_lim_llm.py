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

# Set conservative buffer factor - only use 60% of actual limits to be safe
# This gives us a safety margin for imprecise tracking and concurrent requests
SAFETY_BUFFER_FACTOR = 0.6
# Additional delay between API calls to avoid burst requests
MIN_DELAY_BETWEEN_CALLS = 2.0  # seconds
# Default cooldown period after hitting limits
DEFAULT_COOLDOWN_PERIOD = 60.0  # seconds

# Define fallback model chains for different model families
GEMINI_FALLBACK_CHAIN = {
    "gemini-2.5-pro-preview-03-25": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "gemini-2.0-flash": ["gemini-2.0-flash-lite"],
    "gemini-2.0-flash-lite": []  # No fallbacks for the lowest tier
}

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
            self._cooldown_until = {}  # Timestamps for cooldown periods
            self._last_request_time = {}  # Last request time per model
            self._consecutive_errors = {}  # Track consecutive errors per model
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
            # Apply safety buffer to all limits
            safe_rpm = max(1, int(rpm_limit * SAFETY_BUFFER_FACTOR))
            safe_tpm = max(1, int(tpm_limit * SAFETY_BUFFER_FACTOR))
            safe_rpd = max(1, int(rpd_limit * SAFETY_BUFFER_FACTOR))
            
            # Store the limits for this model
            self._model_limits[model_id] = {
                'rpm': safe_rpm,  # Apply safety buffer
                'tpm': safe_tpm,  # Apply safety buffer
                'rpd': safe_rpd,  # Apply safety buffer
                'original_rpm': rpm_limit,  # Store original for reference
                'original_tpm': tpm_limit,
                'original_rpd': rpd_limit
            }
            
            # Initialize tracking for this model if not already present
            if model_id not in self._request_timestamps:
                self._request_timestamps[model_id] = []
            
            if model_id not in self._token_usage:
                self._token_usage[model_id] = []
            
            if model_id not in self._cooldown_until:
                self._cooldown_until[model_id] = 0.0
                
            if model_id not in self._last_request_time:
                self._last_request_time[model_id] = 0.0
                
            if model_id not in self._consecutive_errors:
                self._consecutive_errors[model_id] = 0
                
            logger.info(f"Initialized rate limit tracking for model {model_id} with limits: "
                       f"RPM={rpm_limit} (safe: {safe_rpm}), TPM={tpm_limit} (safe: {safe_tpm}), "
                       f"RPD={rpd_limit} (safe: {safe_rpd})")
    
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
            
            # Update last request time
            self._last_request_time[model_id] = current_time
            
            # Reset consecutive errors on successful call
            self._consecutive_errors[model_id] = 0
            
            # Clean up old data
            self._clean_old_data(model_id)
            
            logger.debug(f"Updated rate limit tracking for {model_id}: "
                        f"+{total_tokens} tokens ({input_tokens} in, {output_tokens} out)")
    
    def record_error(self, model_id: str, is_rate_limit: bool = True):
        """Record an error occurrence for a model
        
        Args:
            model_id: The model identifier
            is_rate_limit: Whether the error was a rate limit error
        """
        with self._lock:
            if model_id not in self._consecutive_errors:
                self._consecutive_errors[model_id] = 0
                
            # Increment error count
            self._consecutive_errors[model_id] += 1
            
            # Set a cooldown period that increases with consecutive errors
            if is_rate_limit:
                # Exponential cooldown based on consecutive errors
                cooldown_time = DEFAULT_COOLDOWN_PERIOD * (2 ** min(self._consecutive_errors[model_id] - 1, 5))
                self._cooldown_until[model_id] = time.time() + cooldown_time
                logger.warning(f"Set cooldown for {model_id} until {datetime.fromtimestamp(self._cooldown_until[model_id]).strftime('%H:%M:%S')} "
                             f"({cooldown_time:.1f}s) after error #{self._consecutive_errors[model_id]}")
    
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
                
            # Check if we're in a cooldown period
            if self._cooldown_until.get(model_id, 0) > current_time:
                cooldown_remaining = self._cooldown_until[model_id] - current_time
                return True, "cooldown", cooldown_remaining
                
            # Check minimum delay between calls
            if self._last_request_time.get(model_id, 0) > 0:
                time_since_last_call = current_time - self._last_request_time[model_id]
                if time_since_last_call < MIN_DELAY_BETWEEN_CALLS:
                    wait_time = MIN_DELAY_BETWEEN_CALLS - time_since_last_call
                    return True, "delay", wait_time
            
            minute_ago = current_time - 60
            day_ago = current_time - 86400  # 24 hours in seconds
            
            # Check RPM limit
            rpm_limit = self._model_limits[model_id]['rpm']
            rpm_count = sum(1 for ts in self._request_timestamps[model_id] if ts > minute_ago)
            
            rpm_percentage = (rpm_count / rpm_limit * 100) if rpm_limit > 0 else 0
            logger.debug(f"Current RPM usage for {model_id}: {rpm_count}/{rpm_limit} ({rpm_percentage:.1f}%)")
            
            # Lower threshold for Google Gemini models (extra cautious)
            rpm_threshold = 0.8 * rpm_limit if 'gemini' in model_id.lower() else 0.9 * rpm_limit
            
            if rpm_count >= rpm_threshold:
                # Calculate wait time - find oldest timestamp within the minute window
                minute_timestamps = [ts for ts in self._request_timestamps[model_id] if ts > minute_ago]
                if minute_timestamps:
                    oldest_ts = min(minute_timestamps)
                    wait_time = 61 - (current_time - oldest_ts)  # Add extra second as buffer
                else:
                    wait_time = MIN_DELAY_BETWEEN_CALLS
                return True, "rpm", wait_time
            
            # Check TPM limit
            tpm_limit = self._model_limits[model_id]['tpm']
            tpm_count = sum(tokens for ts, tokens in self._token_usage[model_id] if ts > minute_ago)
            
            tpm_percentage = (tpm_count / tpm_limit * 100) if tpm_limit > 0 else 0
            logger.debug(f"Current TPM usage for {model_id}: {tpm_count}/{tpm_limit} ({tpm_percentage:.1f}%)")
            
            # Lower threshold for Google Gemini models
            tpm_threshold = 0.8 * tpm_limit if 'gemini' in model_id.lower() else 0.9 * tpm_limit
            
            if tpm_count >= tpm_threshold:
                # Calculate wait time based on token usage timestamps
                minute_token_usage = [(ts, tokens) for ts, tokens in self._token_usage[model_id] if ts > minute_ago]
                if minute_token_usage:
                    oldest_ts = min(ts for ts, _ in minute_token_usage)
                    wait_time = 61 - (current_time - oldest_ts)  # Add extra second as buffer
                else:
                    wait_time = MIN_DELAY_BETWEEN_CALLS
                return True, "tpm", wait_time
            
            # Check RPD limit
            rpd_limit = self._model_limits[model_id]['rpd']
            rpd_count = len([ts for ts in self._request_timestamps[model_id] if ts > day_ago])
            
            rpd_percentage = (rpd_count / rpd_limit * 100) if rpd_limit > 0 else 0
            
            # Only log RPD percentage occasionally to avoid log spam
            if rpd_percentage > 50 and (current_time % 60) < 1:
                logger.info(f"Current RPD usage for {model_id}: {rpd_count}/{rpd_limit} ({rpd_percentage:.1f}%)")
            
            # Lower threshold for Google Gemini models
            rpd_threshold = 0.8 * rpd_limit if 'gemini' in model_id.lower() else 0.9 * rpd_limit
            
            if rpd_count >= rpd_threshold:
                # Calculate wait time based on daily usage
                day_timestamps = [ts for ts in self._request_timestamps[model_id] if ts > day_ago]
                if day_timestamps:
                    oldest_ts = min(day_timestamps)
                    wait_time = 86400 - (current_time - oldest_ts) + 1  # Add extra second as buffer
                else:
                    wait_time = MIN_DELAY_BETWEEN_CALLS
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
                    'rpm': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0},
                    'tpm': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0},
                    'rpd': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0}
                }
                
            current_time = time.time()
            minute_ago = current_time - 60
            day_ago = current_time - 86400  # 24 hours in seconds
            
            # Get original limits
            original_rpm = self._model_limits[model_id]['original_rpm']
            original_tpm = self._model_limits[model_id]['original_tpm']
            original_rpd = self._model_limits[model_id]['original_rpd']
            
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
            
            # Include cooldown status if applicable
            cooldown_remaining = max(0, self._cooldown_until.get(model_id, 0) - current_time)
            
            return {
                'rpm': {
                    'limit': rpm_limit, 
                    'usage': rpm_usage, 
                    'percentage': rpm_percentage,
                    'original': original_rpm
                },
                'tpm': {
                    'limit': tpm_limit, 
                    'usage': tpm_usage, 
                    'percentage': tpm_percentage,
                    'original': original_tpm
                },
                'rpd': {
                    'limit': rpd_limit, 
                    'usage': rpd_usage, 
                    'percentage': rpd_percentage,
                    'original': original_rpd
                },
                'cooldown': {
                    'active': cooldown_remaining > 0,
                    'remaining': cooldown_remaining,
                    'consecutive_errors': self._consecutive_errors.get(model_id, 0)
                }
            }

    def check_model_availability(self, model_id: str, threshold: float = 0.7) -> bool:
        """Check if a model is available based on current usage
        
        Args:
            model_id: The model identifier
            threshold: Usage threshold to consider (0-1)
            
        Returns:
            True if model is available, False if it's approaching limits
        """
        with self._lock:
            # Ensure model is initialized
            if model_id not in self._model_limits:
                # If model isn't tracked, assume it's available
                return True
                
            current_time = time.time()
                
            # Check if we're in a cooldown period
            if self._cooldown_until.get(model_id, 0) > current_time:
                return False
                
            minute_ago = current_time - 60
            
            # Calculate percentages of limits
            rpm_limit = self._model_limits[model_id]['rpm']
            rpm_count = sum(1 for ts in self._request_timestamps[model_id] if ts > minute_ago)
            rpm_percentage = (rpm_count / rpm_limit) if rpm_limit > 0 else 0
            
            tpm_limit = self._model_limits[model_id]['tpm']
            tpm_count = sum(tokens for ts, tokens in self._token_usage[model_id] if ts > minute_ago)
            tpm_percentage = (tpm_count / tpm_limit) if tpm_limit > 0 else 0
            
            # Model is considered unavailable if either limit is above threshold
            if rpm_percentage > threshold or tpm_percentage > threshold:
                return False
                
            return True

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
        base_wait_time: float = 2.0,  # Increased default wait time
        max_retries: int = 5,  # Increased max retries
        jitter_factor: float = 0.2,  # Increased jitter
        enable_fallback: bool = False,  # Enable fallback to lower-tier models
        **kwargs
    ):
        """Initialize the rate-limited model.
        
        Args:
            model_id: The model identifier to use with LiteLLM
            model_info_path: Optional path to the JSON file containing model information
            base_wait_time: Optional base wait time in seconds for rate limit backoff
            max_retries: Optional maximum number of retries for rate limit errors
            jitter_factor: Optional factor to apply random jitter to wait times (0-1)
            enable_fallback: Whether to enable automatic fallback to lower-tier models
            **kwargs: Additional arguments to pass to LiteLLMModel
        """
        # Ensure LiteLLM logging is disabled before initializing
        litellm.utils.logging_enabled = False
        
        # Store the original model ID for reference
        self.original_model_id = model_id
        
        # Initialize the parent class
        super().__init__(model_id=model_id, **kwargs)
        
        self.model_id = model_id
        self.base_wait_time = base_wait_time
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        self.enable_fallback = enable_fallback
        
        # Add counter for tracking API calls
        self.api_call_count = 0
        
        # Track fallback information
        self.current_fallback_level = 0
        self.fallback_history = []
        
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
            
            # Apply extra caution for Google Gemini models as they have stricter enforcement
            if 'gemini' in model_id.lower():
                # If this is a Gemini model, be very conservative with limits
                logger.info(f"Applying stricter limits for Gemini model {model_id}")
                
                # Apply special overrides for Gemini-specific models
                if model_key == "gemini-2.0-flash":
                    # Hard-override RPM limit to 12 for gemini-2.0-flash
                    self.rpm_limit = min(self.rpm_limit, 12)  # Never exceed 12 RPM for free tier
                elif model_key == "gemini-2.0-flash-lite":
                    # Hard-override RPM limit to 20 for gemini-2.0-flash-lite
                    self.rpm_limit = min(self.rpm_limit, 20)  # Never exceed 20 RPM for free tier lite
                elif model_key == "gemini-2.5-pro-preview-03-25":
                    # Hard-override RPM limit to 12 for gemini-2.5-pro
                    self.rpm_limit = min(self.rpm_limit, 12)  # Never exceed 12 RPM for free tier pro
                    
            # Log fallback status
            if self.enable_fallback and 'gemini' in model_id.lower():
                model_base = model_id.split('/')[-1]
                fallback_options = GEMINI_FALLBACK_CHAIN.get(model_base, [])
                if fallback_options:
                    fallback_str = " -> ".join(fallback_options)
                    logger.info(f"Model fallback enabled: {model_base} -> {fallback_str}")
                else:
                    logger.info(f"Model fallback enabled but no fallbacks available for {model_base}")
                    
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
        
    def _get_fallback_model(self, current_model_id: str) -> Optional[str]:
        """Get the next fallback model in the chain
        
        Args:
            current_model_id: The current model identifier
            
        Returns:
            The next fallback model or None if no fallback is available
        """
        if not self.enable_fallback:
            return None
            
        # Extract the base model name from the full model ID
        provider_prefix = ""
        if "/" in current_model_id:
            provider_prefix = current_model_id.split("/")[0] + "/"
            model_base = current_model_id.split("/")[-1]
        else:
            model_base = current_model_id
            
        # Check if this model has fallbacks
        if model_base not in GEMINI_FALLBACK_CHAIN:
            logger.info(f"No fallback chain defined for {model_base}")
            return None
            
        fallback_options = GEMINI_FALLBACK_CHAIN[model_base]
        if not fallback_options:
            logger.info(f"No fallback models available for {model_base}")
            return None
            
        # Try each fallback model in order
        for fallback_model_base in fallback_options:
            fallback_model_id = f"{provider_prefix}{fallback_model_base}"
            
            # Check if this fallback model is available (not at its rate limits)
            if self.shared_tracker.check_model_availability(fallback_model_id, threshold=0.7):
                logger.info(f"Selected fallback model: {fallback_model_id}")
                return fallback_model_id
            else:
                logger.info(f"Fallback model {fallback_model_id} is unavailable due to rate limits")
                
        logger.warning(f"All fallback models for {model_base} are unavailable")
        return None
        
    def reset_to_original_model(self) -> bool:
        """Reset to the original model if it's available
        
        Returns:
            True if reset was successful, False otherwise
        """
        if self.model_id == self.original_model_id:
            return True
            
        if self.shared_tracker.check_model_availability(self.original_model_id, threshold=0.6):
            logger.info(f"Resetting from fallback {self.model_id} to original model {self.original_model_id}")
            self.model_id = self.original_model_id
            self.current_fallback_level = 0
            return True
            
        return False
        
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
        
        # Check for timeout errors, which can be related to Gemini rate limits
        if isinstance(error, litellm.exceptions.APIConnectionError):
            error_str_lower = error_str.lower()
            if "timeout" in error_str_lower or "exceeded the maximum execution time" in error_str_lower:
                logger.warning(f"Treating timeout error as rate limit related: {error_str}")
                return True
        
        # Check for execution timeout errors from smolagents
        if "ExecutionTimeoutError" in error_type or "ExecutionTimeoutError" in error_str:
            logger.warning(f"Treating execution timeout error as rate limit related: {error_str}")
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
        rate_limit_terms = [
            "rate limit", "quota", "too many requests", "429", "resource exhausted", 
            "resource has been exhausted", "exceeded your current quota", 
            "quota exhausted", "requests per minute", "requests per day"
        ]
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
                if 'quota' in str(error.error).lower():
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
            
            # Add jitter to wait time to avoid thundering herd
            jitter = random.uniform(-0.1, 0.1) * wait_time
            adjusted_wait_time = max(0.1, wait_time + jitter)
            
            time.sleep(adjusted_wait_time)
            self.last_wait_time = adjusted_wait_time
            return True, adjusted_wait_time
            
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
        
    def _create_temp_model(self, model_id: str) -> 'RateLimitedLiteLLMModel':
        """Create a temporary RateLimitedLiteLLMModel instance with proper settings
        
        Args:
            model_id: The model ID to use
            
        Returns:
            A properly configured RateLimitedLiteLLMModel instance that shares the rate limiter
        """
        # Create a new RateLimitedLiteLLMModel instance with the fallback model ID
        # We need to ensure it's properly connected to the same rate limiter
        temp_model = RateLimitedLiteLLMModel(
            model_id=model_id,
            max_retries=self.max_retries,
            base_wait_time=self.base_wait_time,
            jitter_factor=self.jitter_factor,
            enable_fallback=False,  # Disable cascading fallbacks
            model_info_path=self.gem_llm_info_path  # Share the same model info
        )
        
        # Critical: Use the same shared tracker instance to ensure coordinated rate limiting
        temp_model.shared_tracker = self.shared_tracker
        
        # Set proper timeouts
        temp_model._request_timeout = 60  # Increase request timeout
        
        logger.info(f"Created rate-limited model for fallback: {model_id} (sharing rate tracker)")
        return temp_model

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Override the __call__ method to add rate limiting
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            The model response
        """
        # Try to reset to original model if possible (periodically)
        if self.current_fallback_level > 0 and self.api_call_count % 5 == 0:
            self.reset_to_original_model()
            
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
        current_model = self.model_id
        
        # Set default timeout in kwargs if not present
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60  # 60 seconds timeout
        
        while True:
            try:
                # Check if we need to use a fallback model
                if current_model != self.model_id:
                    # Create a temporary instance of LiteLLMModel with the fallback model
                    # This avoids the conflict with the 'model' parameter
                    temp_model = self._create_temp_model(current_model)
                    response = temp_model(messages=messages, **kwargs)
                else:
                    # Use our current instance since we're on the primary model
                    response = super().__call__(messages=messages, **kwargs)
                
                # Get actual token counts from the model
                token_counts = self.get_token_counts()
                input_tokens = token_counts.get('input_token_count', estimated_input_tokens)
                output_tokens = token_counts.get('output_token_count', 0)
                total_tokens = input_tokens + output_tokens
                
                # Update rate limit tracking
                self._update_rate_limit_tracking(input_tokens, output_tokens)
                
                # Record successful model use
                if current_model != self.model_id:
                    self.fallback_history.append((datetime.now(), self.model_id, current_model))
                    logger.info(f"Successfully used fallback model: {current_model}")
                    
                # Increment API call counter
                self.api_call_count += 1
                
                # Print rate limit status every 3 calls
                if self.api_call_count % 3 == 0:
                    logger.info(f"Automatic rate limit status check (call #{self.api_call_count}):")
                    self.print_rate_limit_status(use_logger=True)
                
                # Log token usage
                logger.info(f"API call successful with model {current_model}. Used {total_tokens} tokens "
                           f"({input_tokens} input, {output_tokens} output)")
                
                return response
                
            except Exception as e:
                self.last_error = e
                
                # Log the full error for debugging
                logger.debug(f"API call error with model {current_model}: {type(e).__name__}: {str(e)}")
                
                # Check if this is a rate limit error
                is_rate_limit = self._is_rate_limit_error(e)
                if is_rate_limit:
                    # Record the rate limit error in the tracker
                    self.shared_tracker.record_error(current_model, is_rate_limit=True)
                    
                    # Try a fallback model if enabled
                    if self.enable_fallback:
                        fallback_model = self._get_fallback_model(current_model)
                        if fallback_model and fallback_model != current_model:
                            self.current_fallback_level += 1
                            logger.warning(f"Rate limit hit. Switching from {current_model} to fallback model {fallback_model}")
                            current_model = fallback_model
                            # Reset retry counter when switching models
                            retries = 0
                            continue
                    
                    retries += 1
                    
                    if retries > self.max_retries:
                        logger.error(f"Maximum retries ({self.max_retries}) exceeded for rate limit error: {e}")
                        raise
                        
                    # Calculate backoff time with jitter
                    jitter = self.jitter_factor * backoff_time * (2 * random.random() - 1)
                    wait_time = backoff_time + jitter
                    
                    # Double minimum wait time on each retry
                    if 'gemini' in current_model.lower():
                        # For Gemini models, use even more aggressive backoff
                        wait_time = max(wait_time, 5.0 * (2 ** (retries - 1)))
                    
                    logger.warning(f"Rate limit error with model {current_model}: {e}. Retry {retries}/{self.max_retries} "
                                  f"after {wait_time:.2f}s backoff")
                    
                    # Add full error info to logs for debugging
                    logger.debug(f"Full error traceback: {traceback.format_exc()}")
                    
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
        status = self.shared_tracker.get_rate_limit_status(self.model_id)
        
        # Add fallback status information if enabled
        if self.enable_fallback:
            # Add current fallback level
            status['fallback'] = {
                'enabled': True,
                'current_level': self.current_fallback_level,
                'original_model': self.original_model_id,
                'current_model': self.model_id,
                'fallback_count': len(self.fallback_history)
            }
        else:
            status['fallback'] = {
                'enabled': False
            }
            
        return status
        
    def print_rate_limit_status(self, use_logger=False):
        """Print the current rate limit status to the console or log
        
        Args:
            use_logger: If True, output to logger instead of console
        """
        status = self.get_rate_limit_status()
        
        # Format the status message
        status_lines = [
            "Rate Limit Status:",
            f"  RPM: {status['rpm']['usage']}/{status['rpm']['limit']} ({status['rpm']['percentage']:.1f}%) [original: {status['rpm']['original']}]",
            f"  TPM: {status['tpm']['usage']}/{status['tpm']['limit']} ({status['tpm']['percentage']:.1f}%) [original: {status['tpm']['original']}]",
            f"  RPD: {status['rpd']['usage']}/{status['rpd']['limit']} ({status['rpd']['percentage']:.1f}%) [original: {status['rpd']['original']}]"
        ]
        
        # Add fallback information if enabled
        if status['fallback']['enabled']:
            if status['fallback']['current_level'] > 0:
                status_lines.append(f"  Model Fallback: Active (Level {status['fallback']['current_level']})")
                status_lines.append(f"  Current Model: {status['fallback']['current_model']} (Original: {status['fallback']['original_model']})")
                status_lines.append(f"  Fallback Count: {status['fallback']['fallback_count']}")
            else:
                status_lines.append(f"  Model Fallback: Enabled but not active")
        else:
            status_lines.append(f"  Model Fallback: Disabled")
        
        # Add cooldown status if applicable
        if 'cooldown' in status and status['cooldown']['active']:
            status_lines.append(f"  ⚠️ Cooldown period active: {status['cooldown']['remaining']:.1f}s remaining")
            status_lines.append(f"  Consecutive errors: {status['cooldown']['consecutive_errors']}")
        
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
        
        # Explicitly set LiteLLM logger level to ERROR to suppress INFO logs
        logging.getLogger("litellm").setLevel(logging.ERROR)
        
        logger.info(f"Configured logging: HTTP libraries set to {logging.getLevelName(level)}, "
                   f"LiteLLM logging {'enabled' if enable_litellm_logging else 'disabled'}")
                   
        return True