import time
import random
import json
import os
from collections import deque
from datetime import datetime, timedelta
from smolagents import LiteLLMModel
from litellm.exceptions import RateLimitError

class RateLimitedLiteLLMModel(LiteLLMModel):
    """A wrapper around LiteLLMModel that adds rate limiting to handle API rate limits.
    
    This class implements model-specific rate limiting based on the Gemini API limits:
    - RPM (Requests Per Minute)
    - TPM (Tokens Per Minute)
    - RPD (Requests Per Day)
    
    It also implements exponential backoff with jitter to handle rate limits gracefully.
    When rate limits are encountered, it will automatically retry with increasing wait times.
    """
    
    # get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    parent_dir = os.path.dirname(current_dir)
    # get the gem_llm_info.json file
    gem_llm_info_path = os.path.join(current_dir, "gem_llm_info.json")

    def __init__(
        self, 
        model_id: str, 
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        jitter_factor: float = 0.2,
        model_info_path: str = gem_llm_info_path,
        **kwargs
    ):
        """Initialize the rate-limited model.
        
        Args:
            model_id: The model ID to use with LiteLLM
            base_wait_time: Base wait time in seconds (default: 2.0)
            max_retries: Maximum number of retry attempts (default: 3)
            jitter_factor: Random jitter factor to add to wait times (default: 0.2)
            model_info_path: Path to the model info JSON file (default: None)
            **kwargs: Additional arguments to pass to LiteLLMModel
        """
        super().__init__(model_id=model_id, **kwargs)
        self.base_wait_time = base_wait_time
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        self.consecutive_failures = 0
        self.last_call_time = 0
        
        # Extract the base model name from the model_id (e.g., "gemini/gemini-2.0-flash" -> "gemini-2.0-flash")
        self.base_model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Load model rate limit information
        self.model_info = self._load_model_info(model_info_path)
        
        # Initialize rate limit tracking
        self.request_timestamps = deque(maxlen=1000)  # Store timestamps of recent requests
        self.token_usage = deque(maxlen=1000)  # Store token usage with timestamps
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now() + timedelta(days=1)
        
        # Log initialization
        print(f"Initialized rate-limited model for {self.base_model_name}")
        if self.base_model_name in self.model_info:
            limits = self.model_info[self.base_model_name]
            print(f"Rate limits: {limits['RPM']} RPM, {limits['TPM']} TPM, {limits['RPD']} RPD")
        else:
            print(f"Warning: No rate limit information found for {self.base_model_name}")
    
    def _load_model_info(self, model_info_path: str = None) -> dict:
        """Load model information from the JSON file.
        
        Args:
            model_info_path: Path to the model info JSON file
            
        Returns:
            Dictionary containing model information
        """
        # Default paths to check
        paths_to_check = [
            model_info_path,
            "agents/utils/gemini/gem_llm_info.json",
            "utils/gemini/gem_llm_info.json",
            "gem_llm_info.json"
        ]
        
        # Filter out None values
        paths_to_check = [p for p in paths_to_check if p is not None]
        
        # Try to load from each path
        for path in paths_to_check:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model info from {path}: {e}")
        
        # Default fallback values if no file is found
        return {
            "gemini-2.0-flash": {
                "RPM": 15,
                "TPM": 1000000,
                "RPD": 1500,
                "api_info": {
                    "name": "models/gemini-2.0-flash",
                    "base_model_id": "",
                    "version": "2.0",
                    "display_name": "Gemini 2.0 Flash",
                    "description": "Gemini 2.0 Flash",
                    "input_token_limit": 1048576,
                    "output_token_limit": 8192,
                    "supported_generation_methods": [
                        "generateContent",
                        "countTokens"
                    ],
                    "temperature": 1.0,
                    "max_temperature": 2.0,
                    "top_p": 0.95,
                    "top_k": 40
                },
                "model_family": "gemini"
            },
            "gemini-2.0-flash-lite": {
                "RPM": 30,
                "TPM": 1000000,
                "RPD": 1500,
                "api_info": {
                    "name": "models/gemini-2.0-flash-lite",
                    "base_model_id": "",
                    "version": "2.0",
                    "display_name": "Gemini 2.0 Flash-Lite",
                    "description": "Gemini 2.0 Flash-Lite",
                    "input_token_limit": 1048576,
                    "output_token_limit": 8192,
                    "supported_generation_methods": [
                        "generateContent",
                        "countTokens"
                    ],
                    "temperature": 1.0,
                    "max_temperature": 2.0,
                    "top_p": 0.95,
                    "top_k": 40
                },
                "model_family": "gemini"
            },
            "gemini-2.0-pro-experimental-02-05": {
                "RPM": 2,
                "TPM": 1000000,
                "RPD": 50
            },
            "gemini-2.0-flash-thinking-experimental-01-21": {
                "RPM": 10,
                "TPM": 4000000,
                "RPD": 1500
            },
            "gemini-1.5-flash": {
                "RPM": 15,
                "TPM": 1000000,
                "RPD": 1500,
                "api_info": {
                    "name": "models/gemini-1.5-flash",
                    "base_model_id": "",
                    "version": "001",
                    "display_name": "Gemini 1.5 Flash",
                    "description": "Alias that points to the most recent stable version of Gemini 1.5 Flash, our fast and versatile multimodal model for scaling across diverse tasks.",
                    "input_token_limit": 1000000,
                    "output_token_limit": 8192,
                    "supported_generation_methods": [
                        "generateContent",
                        "countTokens"
                    ],
                    "temperature": 1.0,
                    "max_temperature": 2.0,
                    "top_p": 0.95,
                    "top_k": 40
                },
                "model_family": "gemini"
            },
            "gemini-1.5-flash-8b": {
                "RPM": 15,
                "TPM": 1000000,
                "RPD": 1500,
                "api_info": {
                    "name": "models/gemini-1.5-flash-8b",
                    "base_model_id": "",
                    "version": "001",
                    "display_name": "Gemini 1.5 Flash-8B",
                    "description": "Stable version of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model, released in October of 2024.",
                    "input_token_limit": 1000000,
                    "output_token_limit": 8192,
                    "supported_generation_methods": [
                        "createCachedContent",
                        "generateContent",
                        "countTokens"
                    ],
                    "temperature": 1.0,
                    "max_temperature": 2.0,
                    "top_p": 0.95,
                    "top_k": 40
                },
                "model_family": "gemini"
            },
            "gemini-1.5-pro": {
                "RPM": 2,
                "TPM": 32000,
                "RPD": 50,
                "api_info": {
                    "name": "models/gemini-1.5-pro",
                    "base_model_id": "",
                    "version": "001",
                    "display_name": "Gemini 1.5 Pro",
                    "description": "Stable version of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens, released in May of 2024.",
                    "input_token_limit": 2000000,
                    "output_token_limit": 8192,
                    "supported_generation_methods": [
                        "generateContent",
                        "countTokens"
                    ],
                    "temperature": 1.0,
                    "max_temperature": 2.0,
                    "top_p": 0.95,
                    "top_k": 40
                },
                "model_family": "gemini"
            }
        }
    
    def _add_jitter(self, wait_time: float) -> float:
        """Add random jitter to wait time to avoid synchronized requests.
        
        Args:
            wait_time: Base wait time in seconds
            
        Returns:
            Wait time with jitter added
        """
        jitter = wait_time * self.jitter_factor
        return wait_time * (1 - self.jitter_factor) + jitter * random.random() * 2
    
    def _calculate_wait_time(self) -> float:
        """Calculate wait time with exponential backoff based on consecutive failures.
        
        Returns:
            Wait time in seconds
        """
        wait_time = self.base_wait_time * (2 ** self.consecutive_failures)
        return self._add_jitter(wait_time)
    
    def _check_rpm_limit(self) -> float:
        """Check if we're approaching the RPM limit and calculate wait time if needed.
        
        Returns:
            Wait time in seconds (0 if no waiting is needed)
        """
        # Get the RPM limit for this model
        rpm_limit = self.model_info.get(self.base_model_name, {}).get("RPM", 10)  # Default to 10 RPM
        
        # Clean up old timestamps (older than 1 minute)
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Count requests in the last minute
        recent_requests = [ts for ts in self.request_timestamps if ts > one_minute_ago]
        
        # If we're approaching the limit, calculate wait time
        if len(recent_requests) >= rpm_limit - 1:  # Leave room for 1 more request
            # Calculate time until the oldest request is more than 1 minute old
            if recent_requests:
                oldest_recent = min(recent_requests)
                time_until_slot_available = (oldest_recent + timedelta(minutes=1) - current_time).total_seconds()
                return max(0, time_until_slot_available)
        
        return 0
    
    def _check_tpm_limit(self, estimated_tokens: int = 1000) -> float:
        """Check if we're approaching the TPM limit and calculate wait time if needed.
        
        Args:
            estimated_tokens: Estimated token usage for the next request
            
        Returns:
            Wait time in seconds (0 if no waiting is needed)
        """
        # Get the TPM limit for this model
        tpm_limit = self.model_info.get(self.base_model_name, {}).get("TPM", 100000)  # Default to 100K TPM
        
        # Clean up old token usage (older than 1 minute)
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Calculate tokens used in the last minute
        recent_tokens = [(tokens, ts) for tokens, ts in self.token_usage if ts > one_minute_ago]
        tokens_used_last_minute = sum(tokens for tokens, _ in recent_tokens)
        
        # If adding estimated_tokens would exceed the limit, calculate wait time
        if tokens_used_last_minute + estimated_tokens > tpm_limit:
            # Calculate time until enough tokens are available
            if recent_tokens:
                # Sort by timestamp (oldest first)
                recent_tokens.sort(key=lambda x: x[1])
                
                # Calculate how many tokens we need to free up
                tokens_to_free = (tokens_used_last_minute + estimated_tokens) - tpm_limit
                tokens_freed = 0
                
                for tokens, timestamp in recent_tokens:
                    tokens_freed += tokens
                    if tokens_freed >= tokens_to_free:
                        # Calculate time until this timestamp is more than 1 minute old
                        time_until_tokens_available = (timestamp + timedelta(minutes=1) - current_time).total_seconds()
                        return max(0, time_until_tokens_available)
        
        return 0
    
    def _check_rpd_limit(self) -> float:
        """Check if we're approaching the RPD limit and calculate wait time if needed.
        
        Returns:
            Wait time in seconds (0 if no waiting is needed)
        """
        # Get the RPD limit for this model
        rpd_limit = self.model_info.get(self.base_model_name, {}).get("RPD", 1000)  # Default to 1000 RPD
        
        # Check if we need to reset the daily counter
        current_time = datetime.now()
        if current_time >= self.daily_reset_time:
            self.daily_request_count = 0
            self.daily_reset_time = current_time + timedelta(days=1)
            # Reset at midnight
            self.daily_reset_time = self.daily_reset_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # If we're at the limit, calculate wait time until reset
        if self.daily_request_count >= rpd_limit:
            time_until_reset = (self.daily_reset_time - current_time).total_seconds()
            return time_until_reset
        
        return 0
    
    def _apply_rate_limit(self, estimated_tokens: int = 1000):
        """Apply rate limiting by waiting if needed.
        
        Args:
            estimated_tokens: Estimated token usage for the next request
        """
        # Check all rate limits and get the maximum wait time
        rpm_wait = self._check_rpm_limit()
        tpm_wait = self._check_tpm_limit(estimated_tokens)
        rpd_wait = self._check_rpd_limit()
        
        # Get the maximum wait time
        wait_time = max(rpm_wait, tpm_wait, rpd_wait)
        
        # Add exponential backoff if we've had failures
        if self.consecutive_failures > 0:
            backoff_wait = self._calculate_wait_time()
            wait_time = max(wait_time, backoff_wait)
        
        # If we need to wait, sleep
        if wait_time > 0:
            reason = "rate limiting"
            if rpm_wait == wait_time:
                reason = "RPM limit"
            elif tpm_wait == wait_time:
                reason = "TPM limit"
            elif rpd_wait == wait_time:
                reason = "RPD limit"
            elif self.consecutive_failures > 0:
                reason = "backoff after failures"
                
            print(f"Rate limiting ({reason}): Waiting {wait_time:.2f} seconds before next API call")
            time.sleep(wait_time)
    
    def _update_rate_limit_tracking(self, tokens_used: int = 1000):
        """Update rate limit tracking after a successful API call.
        
        Args:
            tokens_used: Actual token usage for the request
        """
        current_time = datetime.now()
        
        # Update request timestamps
        self.request_timestamps.append(current_time)
        
        # Update token usage
        self.token_usage.append((tokens_used, current_time))
        
        # Update daily request count
        self.daily_request_count += 1
    
    def completion(self, *args, **kwargs):
        """Override completion method to add rate limiting and retries.
        
        Returns:
            The completion result from the underlying model
            
        Raises:
            RateLimitError: If max retries are exceeded
        """
        retries = 0
        
        # Estimate token usage (can be refined based on actual input)
        estimated_tokens = 1000  # Default estimate
        
        while retries <= self.max_retries:
            try:
                # Apply rate limiting before making the call
                self._apply_rate_limit(estimated_tokens)
                
                # Make the API call - use the correct method from LiteLLMModel
                # Instead of using super().completion(), call the appropriate method
                result = self.generate(*args, **kwargs)
                
                # Update rate limit tracking with actual token usage if available
                actual_tokens = estimated_tokens
                if hasattr(result, 'usage') and result.usage:
                    if hasattr(result.usage, 'total_tokens'):
                        actual_tokens = result.usage.total_tokens
                
                self._update_rate_limit_tracking(actual_tokens)
                
                # Success - reset consecutive failures
                self.consecutive_failures = 0
                
                return result
                
            except RateLimitError as e:
                self.consecutive_failures += 1
                retries += 1
                
                if retries <= self.max_retries:
                    # Calculate backoff time
                    backoff_time = self._calculate_wait_time()
                    
                    print(f"Rate limit hit. Retrying in {backoff_time:.2f} seconds (attempt {retries}/{self.max_retries})")
                    time.sleep(backoff_time)
                else:
                    # Max retries exceeded
                    print(f"Rate limit error after {self.max_retries} retries. Giving up.")
                    raise
            except Exception as e:
                # For other exceptions, just raise them
                raise