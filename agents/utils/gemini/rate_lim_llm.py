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

# Set conservative buffer factor - operate at 90% of actual limits
SAFETY_BUFFER_FACTOR = 0.9
# Default minimum delay between API calls if model-specific delay can't be calculated
DEFAULT_MIN_DELAY_BETWEEN_CALLS = 4.0  # seconds
# Default cooldown period after hitting limits
DEFAULT_COOLDOWN_PERIOD = 60.0  # seconds

# Define fallback model chains for different model families
GEMINI_FALLBACK_CHAIN = {
    "gemini-2.5-pro-preview-03-25": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "gemini-2.0-flash": ["gemini-2.0-flash-lite", "gemini-1.5-flash"],
    "gemini-2.0-flash-lite": ["gemini-1.5-flash"],
    "gemini-1.5-flash": []  # No fallbacks for the lowest tier
}

class SharedRateLimitTracker:
    """Singleton class to track rate limits across all model instances.
    
    This class ensures that all instances of RateLimitedLiteLLMModel share the same
    rate limit counters, cooldown states, and minimum delay settings for each model ID.
    It applies a safety buffer to the configured limits and calculates a dynamic minimum
    delay between calls based on the original RPM for each model. It also manages
    cooldown periods for models that hit rate limits and tracks consecutive errors.
    """
    _instance = None
    _lock = threading.RLock()  # Reentrant lock for thread safety
    
    def __new__(cls):
        """Ensures only one instance of SharedRateLimitTracker is created (Singleton pattern)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedRateLimitTracker, cls).__new__(cls)
                cls._instance._initialized = False # Initialize _initialized flag here
            return cls._instance
    
    def __init__(self):
        """Initializes the shared tracking dictionaries if not already initialized.
        
        This constructor is called only once due to the singleton pattern.
        It sets up dictionaries to store model-specific limits, request timestamps,
        token usage, cooldown end times, last request times, and consecutive error counts.

        Internal Attributes (initialized if not present):
            _model_limits (dict): Stores configured limits (buffered RPM, TPM, RPD,
                                  original limits, and calculated min_delay) for each model_id.
            _request_timestamps (dict): A list of timestamps for recent requests for each model_id,
                                      used for RPM and RPD calculations.
            _token_usage (dict): A list of (timestamp, tokens) tuples for recent requests
                               for each model_id, used for TPM calculations.
            _cooldown_until (dict): Timestamp until which a model_id is in a cooldown period
                                  due to hitting rate limits.
            _last_request_time (dict): Timestamp of the last request allowed to proceed for
                                     each model_id, used for enforcing min_delay.
            _consecutive_errors (dict): Count of consecutive rate limit errors encountered
                                      for each model_id, used for exponential backoff on cooldowns.
        """
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            self._model_limits = {}  
            self._request_timestamps = {}
            self._token_usage = {}  
            self._cooldown_until = {} 
            self._last_request_time = {} 
            self._consecutive_errors = {} 
            self._initialized = True
            logger.info("Initialized SharedRateLimitTracker")
    
    def initialize_model(self, model_id: str, rpm_limit: int, tpm_limit: int, rpd_limit: int):
        """Initializes or updates rate limit tracking parameters for a specific model ID.
        
        This method applies the `SAFETY_BUFFER_FACTOR` to the provided API limits to
        calculate 'safe' operating thresholds. It also calculates a `min_delay`
        between consecutive calls for the model based on its original RPM limit.
        If a model is re-initialized (e.g., due to fallback), its tracking data
        structures are ensured to exist.

        Args:
            model_id (str): The unique identifier for the model (e.g., "gemini/gemini-1.5-flash").
            rpm_limit (int): The original Requests Per Minute limit from the API/configuration.
            tpm_limit (int): The original Tokens Per Minute limit from the API/configuration.
            rpd_limit (int): The original Requests Per Day limit from the API/configuration.
        """
        with self._lock:
            safe_rpm = max(1, int(rpm_limit * SAFETY_BUFFER_FACTOR))
            safe_tpm = max(1, int(tpm_limit * SAFETY_BUFFER_FACTOR))
            safe_rpd = max(1, int(rpd_limit * SAFETY_BUFFER_FACTOR))
            
            model_min_delay = (60.0 / rpm_limit) if rpm_limit > 0 else DEFAULT_MIN_DELAY_BETWEEN_CALLS
            
            self._model_limits[model_id] = {
                'rpm': safe_rpm,
                'tpm': safe_tpm,
                'rpd': safe_rpd,
                'original_rpm': rpm_limit,
                'original_tpm': tpm_limit,
                'original_rpd': rpd_limit,
                'min_delay': model_min_delay
            }
            
            if model_id not in self._request_timestamps: self._request_timestamps[model_id] = []
            if model_id not in self._token_usage: self._token_usage[model_id] = []
            if model_id not in self._cooldown_until: self._cooldown_until[model_id] = 0.0
            if model_id not in self._last_request_time: self._last_request_time[model_id] = 0.0
            if model_id not in self._consecutive_errors: self._consecutive_errors[model_id] = 0
                
            logger.info(f"Initialized rate limit tracking for model {model_id} with limits: "
                       f"Original RPM={rpm_limit} (Safe RPM={safe_rpm}, Min Delay={model_min_delay:.2f}s), "
                       f"Original TPM={tpm_limit} (Safe TPM={safe_tpm}), "
                       f"Original RPD={rpd_limit} (Safe RPD={safe_rpd})")
    
    def update_tracking(self, model_id: str, input_tokens: int, output_tokens: int):
        """Updates request counts and token usage after an API call has been made.
        
        This method should be called after a request has been successfully processed
        by the API, or if a rate limit error occurs that is still counted by the API.
        It records the current timestamp for RPM/RPD and the token count for TPM.
        Successfully tracked calls also reset the consecutive error count for the model.
        Old tracking data (older than 1 day for requests, 1 minute for tokens) is cleaned up.

        Args:
            model_id (str): The identifier of the model for which the call was made.
            input_tokens (int): Number of input tokens used in the API call.
            output_tokens (int): Number of output tokens received from the API call.
        """
        with self._lock:
            current_time = time.time()
            
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker for update_tracking")
                return
                
            self._request_timestamps[model_id].append(current_time)
            total_tokens = input_tokens + output_tokens
            self._token_usage[model_id].append((current_time, total_tokens))
            
            # _last_request_time is updated pre-emptively in check_rate_limits if the call proceeds.
            # No need to update it here again unless we want to mark the *end* of the call.
            
            self._consecutive_errors[model_id] = 0 # Reset on any counted call
            self._clean_old_data(model_id)
            
            logger.debug(f"Updated rate limit tracking for {model_id}: "
                        f"+{total_tokens} tokens ({input_tokens} in, {output_tokens} out)")
    
    def record_error(self, model_id: str, is_rate_limit: bool = True):
        """Records an error for a model, increments its consecutive error count, and applies a cooldown if it's a rate limit.
        
        If `is_rate_limit` is True, an exponential backoff strategy is used to determine
        the cooldown duration, increasing with the number of consecutive errors, capped at 2^5 factor.
        The cooldown prevents further calls to the model until the period expires.

        Args:
            model_id (str): The model identifier for which the error occurred.
            is_rate_limit (bool): If True, the error is treated as a rate limit violation,
                                  triggering a cooldown. Defaults to True.
        """
        with self._lock:
            if model_id not in self._consecutive_errors: self._consecutive_errors[model_id] = 0 # Should be initialized but safe
            self._consecutive_errors[model_id] += 1
            
            if is_rate_limit:
                cooldown_time = DEFAULT_COOLDOWN_PERIOD * (2 ** min(self._consecutive_errors[model_id] - 1, 5)) # Exponential backoff for cooldown
                self._cooldown_until[model_id] = time.time() + cooldown_time
                logger.warning(f"Set cooldown for {model_id} until {datetime.fromtimestamp(self._cooldown_until[model_id]).strftime('%H:%M:%S')} "
                             f"({cooldown_time:.1f}s) after error #{self._consecutive_errors[model_id]}")
    
    def _clean_old_data(self, model_id: str):
        """Removes outdated request timestamps and token usage data for the given model ID.
        
        Request timestamps older than 24 hours (for RPD) and token usage entries
        older than 1 minute (for TPM) are removed to keep the tracking data relevant
        and prevent unbounded growth. RPM also uses data from the last minute.

        Args:
            model_id (str): The model identifier whose old data needs cleaning.
        """
        current_time = time.time()
        minute_ago = current_time - 60
        day_ago = current_time - 86400
        
        if model_id in self._request_timestamps:
            self._request_timestamps[model_id] = [ts for ts in self._request_timestamps.get(model_id, []) if ts > day_ago]
        if model_id in self._token_usage:
            self._token_usage[model_id] = [(ts, tokens) for ts, tokens in self._token_usage.get(model_id, []) if ts > minute_ago]
    
    def check_rate_limits(self, model_id: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Checks if an API call for the specified model would violate current rate limits or cooldown status.
        
        This comprehensive check verifies:
        1. If the model is currently in a cooldown period.
        2. If the time since the last request for this model respects the `min_delay`.
        3. If the RPM (Requests Per Minute) would exceed the safe limit.
        4. If the TPM (Tokens Per Minute) would exceed the safe limit (Note: this check is based on past usage,
           actual token count for the current call is not known at this stage).
        5. If the RPD (Requests Per Day) would exceed the safe limit.
        
        If all checks pass, this method pre-emptively updates `_last_request_time` for the model
        to the current time. This helps throttle immediate subsequent concurrent requests by making them
        respect the `min_delay` relative to this optimistic update.

        Args:
            model_id (str): The model identifier to check limits for.

        Returns:
            Tuple[bool, Optional[str], Optional[float]]: 
                - `would_exceed` (bool): True if a limit is hit or cooldown is active, False otherwise.
                - `limit_type` (Optional[str]): A string indicating the type of limit hit
                  (e.g., "rpm", "tpm", "rpd", "cooldown", "delay", "uninitialized") if `would_exceed` is True.
                - `wait_time_seconds` (Optional[float]): The suggested time in seconds to wait before
                  retrying if a limit is hit. This can be the remaining cooldown time, time to respect
                  min_delay, or time until a minute/day window potentially clears up.
        """
        with self._lock:
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker for check_rate_limits")
                return True, "uninitialized", 0 
                
            current_time = time.time()
                
            if self._cooldown_until.get(model_id, 0) > current_time:
                cooldown_remaining = self._cooldown_until[model_id] - current_time
                return True, "cooldown", cooldown_remaining
                
            model_min_delay = self._model_limits[model_id].get('min_delay', DEFAULT_MIN_DELAY_BETWEEN_CALLS)
            last_call_ts = self._last_request_time.get(model_id, 0)
            
            if last_call_ts > 0:
                time_since_last_call = current_time - last_call_ts
                if time_since_last_call < model_min_delay:
                    wait_time = model_min_delay - time_since_last_call
                    return True, "delay", wait_time + 0.01 
            
            minute_ago = current_time - 60
            day_ago = current_time - 86400
            
            safe_rpm_limit = self._model_limits[model_id]['rpm']
            rpm_count = sum(1 for ts in self._request_timestamps.get(model_id, []) if ts > minute_ago)
            if rpm_count >= safe_rpm_limit:
                minute_timestamps = [ts for ts in self._request_timestamps.get(model_id, []) if ts > minute_ago]
                wait_time = (60.1 - (current_time - min(minute_timestamps))) if minute_timestamps else model_min_delay
                return True, "rpm", wait_time

            safe_tpm_limit = self._model_limits[model_id]['tpm']
            tpm_count = sum(tokens for ts, tokens in self._token_usage.get(model_id, []) if ts > minute_ago)
            if tpm_count >= safe_tpm_limit:
                minute_token_usage = [(ts, tokens) for ts, tokens in self._token_usage.get(model_id, []) if ts > minute_ago]
                wait_time = (60.1 - (current_time - min(ts for ts, _ in minute_token_usage))) if minute_token_usage else model_min_delay
                return True, "tpm", wait_time
            
            safe_rpd_limit = self._model_limits[model_id]['rpd']
            rpd_count = len([ts for ts in self._request_timestamps.get(model_id, []) if ts > day_ago])
            if rpd_count >= safe_rpd_limit:
                day_timestamps = [ts for ts in self._request_timestamps.get(model_id, []) if ts > day_ago]
                wait_time = (86400.1 - (current_time - min(day_timestamps))) if day_timestamps else model_min_delay
                return True, "rpd", wait_time
            
            self._last_request_time[model_id] = current_time # Eager update
            return False, None, None 
    
    def get_rate_limit_status(self, model_id: str) -> Dict[str, Any]:
        """Retrieves the current rate limit usage and status for the specified model ID.
        
        Provides a snapshot of RPM, TPM, and RPD usage against their safe operating limits
        and original API limits. Also includes current cooldown status, consecutive error count,
        and the configured minimum delay between calls for the model.

        Args:
            model_id (str): The model identifier.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'rpm': {'limit', 'usage', 'percentage', 'original'}
                - 'tpm': {'limit', 'usage', 'percentage', 'original'}
                - 'rpd': {'limit', 'usage', 'percentage', 'original'}
                - 'cooldown': {'active', 'remaining', 'consecutive_errors'}
                - 'min_delay': The minimum delay (seconds) enforced between calls for this model.
        """
        with self._lock:
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized in rate limit tracker for get_rate_limit_status")
                return {
                    'rpm': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0},
                    'tpm': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0},
                    'rpd': {'limit': 0, 'usage': 0, 'percentage': 0, 'original': 0},
                    'cooldown': {'active': False, 'remaining': 0, 'consecutive_errors': 0},
                    'min_delay': DEFAULT_MIN_DELAY_BETWEEN_CALLS
                }
                
            current_time = time.time()
            minute_ago = current_time - 60
            day_ago = current_time - 86400
            limits = self._model_limits[model_id]
            rpm_usage = sum(1 for ts in self._request_timestamps.get(model_id, []) if ts > minute_ago)
            rpm_percentage = (rpm_usage / limits['rpm'] * 100) if limits['rpm'] > 0 else 0
            tpm_usage = sum(tokens for ts, tokens in self._token_usage.get(model_id, []) if ts > minute_ago)
            tpm_percentage = (tpm_usage / limits['tpm'] * 100) if limits['tpm'] > 0 else 0
            rpd_usage = sum(1 for ts in self._request_timestamps.get(model_id, []) if ts > day_ago)
            rpd_percentage = (rpd_usage / limits['rpd'] * 100) if limits['rpd'] > 0 else 0
            cooldown_remaining = max(0, self._cooldown_until.get(model_id, 0) - current_time)
            
            return {
                'rpm': {'limit': limits['rpm'], 'usage': rpm_usage, 'percentage': rpm_percentage, 'original': limits['original_rpm']},
                'tpm': {'limit': limits['tpm'], 'usage': tpm_usage, 'percentage': tpm_percentage, 'original': limits['original_tpm']},
                'rpd': {'limit': limits['rpd'], 'usage': rpd_usage, 'percentage': rpd_percentage, 'original': limits['original_rpd']},
                'cooldown': {'active': cooldown_remaining > 0, 'remaining': cooldown_remaining, 'consecutive_errors': self._consecutive_errors.get(model_id, 0)},
                'min_delay': limits.get('min_delay', DEFAULT_MIN_DELAY_BETWEEN_CALLS)
            }

    def check_model_availability(self, model_id: str, threshold: float = 0.85) -> bool:
        """Checks if a model is considered 'available' for fallback or proactive switching.
        
        A model is considered available if:
        1. It's not currently in a cooldown period.
        2. Its current RPM usage is below the specified `threshold` of its safe limit.
        3. Its current TPM usage is below the specified `threshold` of its safe limit.
        
        This is primarily used to determine if a fallback model is a viable candidate.

        Args:
            model_id (str): The model identifier to check.
            threshold (float): The usage threshold (0.0 to 1.0, default 0.85) for RPM and TPM.
                               If usage of either exceeds this proportion of the safe limit,
                               the model is considered unavailable.

        Returns:
            bool: True if the model is available according to the criteria, False otherwise.
        """
        with self._lock:
            if model_id not in self._model_limits:
                logger.warning(f"Model {model_id} not initialized for check_model_availability, assuming available.")
                return True 
                
            current_time = time.time()
            if self._cooldown_until.get(model_id, 0) > current_time: return False 
                
            minute_ago = current_time - 60
            limits = self._model_limits[model_id]
            rpm_usage = sum(1 for ts in self._request_timestamps.get(model_id, []) if ts > minute_ago)
            if limits['rpm'] > 0 and (rpm_usage / limits['rpm']) > threshold: return False
            tpm_usage = sum(tokens for ts, tokens in self._token_usage.get(model_id, []) if ts > minute_ago)
            if limits['tpm'] > 0 and (tpm_usage / limits['tpm']) > threshold: return False
            return True

class RateLimitedLiteLLMModel(LiteLLMModel):
    """A wrapper around `smolagents.LiteLLMModel` that adds robust, shared rate limiting,
    automatic retries with exponential backoff and jitter, and optional model fallback.

    This class uses a singleton `SharedRateLimitTracker` to manage API call rates
    globally across all instances and model IDs. It aims to operate within a
    configurable safety buffer (e.g., 90%) of actual API limits and enforces
    dynamically calculated minimum delays between calls for each model.

    Features:
    - Shared, thread-safe rate limit tracking (RPM, TPM, RPD).
    - Operation below API limits using a `SAFETY_BUFFER_FACTOR`.
    - Dynamic minimum delay between calls based on model RPM.
    - Exponential backoff with jitter for retries on rate limit errors.
    - Cooldown periods for models hitting rate limits, with increasing duration
      for consecutive errors.
    - Optional, configurable model fallback (e.g., Gemini Pro -> Flash) when rate limits
      are encountered on the primary model.
    - Automatic re-initialization of limits for fallback models.
    - Periodic attempts to revert to the original model if fallback is active.
    - Enhanced error detection for various rate-limit related exceptions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gem_llm_info_path = os.path.join(current_dir, "gem_llm_info.json")

    def __init__(
        self,
        model_id: str,
        model_info_path: str = gem_llm_info_path,
        base_wait_time: float = 2.0,
        max_retries: int = 5,
        jitter_factor: float = 0.2,
        enable_fallback: bool = False,
        **kwargs
    ):
        """Initializes the rate-limited LiteLLM model wrapper.

        Loads model-specific rate limit information (RPM, TPM, RPD, input token limit)
        from a JSON file. Initializes the shared rate limit tracker for the model.
        Sets up parameters for retry logic (base wait time, max retries, jitter)
        and enables model fallback if specified.

        Args:
            model_id (str): The primary model identifier for LiteLLM (e.g., "gemini/gemini-1.5-flash").
                            This will be the `original_model_id`.
            model_info_path (str): Path to the JSON file containing model rate limit configurations.
                                   Defaults to "gem_llm_info.json" in the same directory.
            base_wait_time (float): Base wait time in seconds for exponential backoff on retries.
            max_retries (int): Maximum number of retries for rate limit errors or designated retryable errors.
            jitter_factor (float): Factor for adding random jitter to wait times (0 to 1 range typical).
                                   Jitter is calculated as `jitter_factor * backoff_time * random_value_in_[-1,1]`.
            enable_fallback (bool): If True, enables automatic fallback to lower-tier models
                                    (defined in `GEMINI_FALLBACK_CHAIN`) upon persistent rate limits.
            **kwargs: Additional arguments passed to the underlying `LiteLLMModel` constructor.
        """
        litellm.utils.logging_enabled = False
        self.original_model_id = model_id
        super().__init__(model_id=model_id, **kwargs)
        self.model_id = model_id 
        self.base_wait_time = base_wait_time
        self.max_retries = max_retries
        self.jitter_factor = jitter_factor
        self.enable_fallback = enable_fallback
        self.api_call_count = 0
        self.current_fallback_level = 0
        self.fallback_history = []
        self.model_info_path_for_fallback = model_info_path

        try:
            with open(model_info_path, 'r') as f: model_info_json = json.load(f)
            model_key_parts = model_id.split('/')
            model_key = model_key_parts[-1] if len(model_key_parts) > 0 else model_id
            if model_key not in model_info_json: 
                logger.warning(f"Model key '{model_key}' (from '{model_id}') not found in {model_info_path}. Using default limits.")
                model_data = model_info_json.get("default", {}) 
            else: model_data = model_info_json[model_key]
            if not isinstance(model_data, dict): 
                logger.error(f"Model data for '{model_key}' is not a dictionary. Using empty defaults.")
                model_data = {}
            self.rpm_limit = model_data.get("rpm_limit", 15) 
            self.tpm_limit = model_data.get("tpm_limit", 1000000)
            self.rpd_limit = model_data.get("rpd_limit", 1500)
            self.input_token_limit = model_data.get("input_token_limit", 32000)
            if 'gemini' in model_id.lower():
                logger.info(f"Applying Gemini-specific RPM considerations for {model_id}")
                if model_key == "gemini-2.0-flash" or model_key == "gemini-1.5-flash": self.rpm_limit = min(self.rpm_limit, 15) 
                elif model_key == "gemini-2.0-flash-lite": self.rpm_limit = min(self.rpm_limit, 30) 
                elif model_key == "gemini-2.5-pro-preview-03-25": self.rpm_limit = min(self.rpm_limit, 15)
        except Exception as e:
            logger.error(f"Error loading model info from {model_info_path}: {e}. Using fallback default limits.")
            self.rpm_limit, self.tpm_limit, self.rpd_limit, self.input_token_limit = 15, 1000000, 1500, 32000
        
        self.shared_tracker = SharedRateLimitTracker()
        self.shared_tracker.initialize_model(self.model_id, self.rpm_limit, self.tpm_limit, self.rpd_limit)
        
        if self.enable_fallback and 'gemini' in self.original_model_id.lower():
            model_base = self.original_model_id.split('/')[-1]
            fallback_options = GEMINI_FALLBACK_CHAIN.get(model_base, [])
            logger.info(f"Model fallback for {self.original_model_id} {'enabled: ' + model_base + ' -> ' + ' -> '.join(fallback_options) if fallback_options else 'enabled but no fallbacks defined'}")

        self.last_api_call_time = 0 
        self.last_wait_time = 0
        self.last_error = None
        self.last_wait_type = "None" # For logging which limit caused wait in _apply_rate_limit
        
    def _get_fallback_model(self, current_model_id_for_fallback: str) -> Optional[str]:
        """Determines the next available fallback model from a predefined chain.

        This method is called when `enable_fallback` is True and the `current_model_id_for_fallback`
        encounters a rate limit. It consults `GEMINI_FALLBACK_CHAIN` for potential
        fallback options. A fallback model is considered "available" if it's not in
        cooldown and its RPM/TPM usage is below 85% of its safe limits, as checked by
        `shared_tracker.check_model_availability`.

        Args:
            current_model_id_for_fallback (str): The model ID that just encountered an issue
                                                 and needs a fallback.

        Returns:
            Optional[str]: The model ID of the next suitable and available fallback model.
                           Returns None if no fallback chain is defined for the current model,
                           if the chain is exhausted, or if no defined fallbacks are currently available.
        """
        if not self.enable_fallback: return None
        provider_prefix = current_model_id_for_fallback.split("/")[0] + "/" if "/" in current_model_id_for_fallback else ""
        model_base_for_fallback = current_model_id_for_fallback.split("/")[-1] if "/" in current_model_id_for_fallback else current_model_id_for_fallback

        if model_base_for_fallback not in GEMINI_FALLBACK_CHAIN:
            logger.info(f"No fallback chain defined for base model {model_base_for_fallback}")
            return None
        fallback_chain = GEMINI_FALLBACK_CHAIN[model_base_for_fallback]
        if not fallback_chain:
            logger.info(f"No fallback models available in chain for {model_base_for_fallback}")
            return None
            
        for next_fallback_base in fallback_chain:
            next_fallback_id = f"{provider_prefix}{next_fallback_base}"
            if self.shared_tracker.check_model_availability(next_fallback_id, threshold=0.85):
                logger.info(f"Selected available fallback model: {next_fallback_id}")
                return next_fallback_id
            else: logger.info(f"Fallback model {next_fallback_id} is unavailable (usage > 85% or cooldown)")
        logger.warning(f"All fallback models for {model_base_for_fallback} are currently unavailable.")
        return None
        
    def reset_to_original_model(self) -> bool:
        """Attempts to revert the active model to the `original_model_id`.

        This method is called periodically if the model is currently using a fallback.
        It checks the availability of the `original_model_id` using
        `shared_tracker.check_model_availability` (with a 70% usage threshold).
        If available, it resets `self.model_id` to `original_model_id`, re-initializes
        its limits in the `shared_tracker`, and resets the fallback level.

        Returns:
            bool: True if the model was successfully reset to the original, or if it was
                  already using the original model. False if the original model is still
                  not considered freely available and the instance remains on a fallback.
        """
        if self.model_id == self.original_model_id: return True
        if self.shared_tracker.check_model_availability(self.original_model_id, threshold=0.7):
            logger.info(f"Attempting to reset from fallback {self.model_id} to original model {self.original_model_id}")
            self.model_id = self.original_model_id
            # Re-fetch original model's limits for re-initialization, as self.rpm_limit might reflect a fallback model's limits.
            try:
                with open(self.model_info_path_for_fallback, 'r') as f: model_info_json = json.load(f)
                model_key = self.original_model_id.split('/')[-1]
                model_data = model_info_json.get(model_key, model_info_json.get("default", {}))
                if not isinstance(model_data, dict): model_data = {}
                original_rpm = model_data.get("rpm_limit", 15)
                original_tpm = model_data.get("tpm_limit", 1000000)
                original_rpd = model_data.get("rpd_limit", 1500)
                if 'gemini' in self.original_model_id.lower(): # Re-apply gemini specific overrides for original
                    if model_key == "gemini-2.0-flash" or model_key == "gemini-1.5-flash": original_rpm = min(original_rpm, 15)
                    elif model_key == "gemini-2.0-flash-lite": original_rpm = min(original_rpm, 30)
                    elif model_key == "gemini-2.5-pro-preview-03-25": original_rpm = min(original_rpm, 15)
                self.shared_tracker.initialize_model(self.model_id, original_rpm, original_tpm, original_rpd)
            except Exception as e: logger.error(f"Error re-initializing original model limits for {self.original_model_id}: {e}")
            self.current_fallback_level = 0
            logger.info(f"Successfully reset to original model: {self.original_model_id}")
            return True
        logger.info(f"Original model {self.original_model_id} still not freely available. Staying on {self.model_id}.")
        return False

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Determines if the given exception is likely a rate-limit related error.
        
        Checks for specific exception types (RateLimitError, ResourceExhausted,
        ServiceUnavailable from LiteLLM and Google API core) and keywords in
        the error message (e.g., "rate limit", "quota", "429", "resource_exhausted").
        Also treats certain timeout errors (`APIConnectionError` with "timeout" or
        "exceeded the maximum execution time", `smolagents.ExecutionTimeoutError`)
        as potentially rate-limit related to trigger fallbacks or retries.

        Args:
            error (Exception): The exception instance to check.

        Returns:
            bool: True if the error is identified as rate-limit related, False otherwise.
        """
        error_type_name = type(error).__name__.lower()
        error_str = str(error).lower()
        if isinstance(error, (RateLimitError, ResourceExhausted, ServiceUnavailable)): return True
        rate_limit_keywords = ["rate limit", "quota", "429", "resource_exhausted", "resource exhausted", "too many requests", "exceeded your current quota"]
        if any(keyword in error_str for keyword in rate_limit_keywords): return True
        if "timeout" in error_str or "exceeded the maximum execution time" in error_str:
            logger.warning(f"Treating timeout error as potentially rate-limit related: {error_str[:200]}")
            return True
        if "executiontimeouterror" in error_type_name: 
            logger.warning(f"Treating ExecutionTimeoutError as potentially rate-limit related: {error_str[:200]}")
            return True
        if isinstance(error, GoogleAPIError) and any(keyword in error_str for keyword in ["quota", "rate", "limit", "429"]): return True
        if isinstance(error, litellm.exceptions.APIError) and any(keyword in error_str for keyword in ["rate", "limit", "quota", "429"]): return True
        if "vertexaiexception" in error_str and ("resource_exhausted" in error_str or "429" in error_str): return True
        return False
        
    def _create_temp_model(self, fallback_model_id: str) -> 'RateLimitedLiteLLMModel':
        """Creates a new RateLimitedLiteLLMModel instance, typically for a fallback scenario.
        
        Note: This method is not directly used in the current primary fallback logic of the `__call__`
        method, which instead modifies `self.model_id` and reuses the existing instance.
        This helper might be intended for alternative fallback strategies or future use.

        The new instance will use the specified `fallback_model_id` and inherit relevant
        configurations like retry behavior from the current instance. Crucially, it should
        use the same singleton `SharedRateLimitTracker` to ensure coordinated rate limiting.
        Fallback is typically disabled for such temporary/fallback instances to prevent
        cascading fallbacks from the temporary instance itself.

        Args:
            fallback_model_id (str): The model ID for the new fallback instance.

        Returns:
            RateLimitedLiteLLMModel: A new `RateLimitedLiteLLMModel` instance configured
                                     for the fallback model.
        """
        logger.info(f"Creating temporary RateLimitedLiteLLMModel for fallback: {fallback_model_id}")
        temp_model = RateLimitedLiteLLMModel(
            model_id=fallback_model_id,
            model_info_path=self.model_info_path_for_fallback, 
            base_wait_time=self.base_wait_time,
            max_retries=self.max_retries, 
            jitter_factor=self.jitter_factor,
            enable_fallback=False, 
        )
        return temp_model

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Makes an API call using the current model, handling rate limits, retries, and fallbacks.

        This is the primary method for interacting with the LLM. It performs the following steps:
        1. Periodically attempts to reset to the `original_model_id` if currently on a fallback.
        2. Estimates input tokens for the request.
        3. Applies rate limiting for the `active_model_for_this_attempt` (which is `self.model_id`)
           by calling `_apply_rate_limit`, potentially waiting if limits are close.
        4. Enters a retry loop:
            a. Calls `super().__call__` (i.e., `LiteLLMModel.__call__`) using `self.model_id`
               (which reflects the current original or fallback model).
            b. If successful, updates tracking in `SharedRateLimitTracker`, logs success,
               and returns the response.
            c. If an exception occurs:
                i. Detects if it's a rate limit error using `_is_rate_limit_error`.
                ii. If it's a rate limit error:
                    - Records the error in `SharedRateLimitTracker` for the current model.
                    - If `enable_fallback` is True, attempts to switch to a `new_fallback_candidate`
                      using `_get_fallback_model`. If a valid fallback is found, `self.model_id`
                      is updated, its limits are initialized in the tracker, and the loop continues
                      with the new model.
                    - If no fallback is attempted or successful, increments retry count.
                      If max retries are exceeded, raises the error.
                    - Otherwise, waits for an exponentially backed-off duration (with jitter)
                      and specific Gemini model considerations, then retries.
                      Applies rate limits again before the next attempt in the loop.
                iii. If not a rate limit error, re-raises the exception.

        Args:
            messages (List[Dict[str, str]]): The list of messages for the chat completion,
                                              conforming to the LiteLLM input format.
            **kwargs: Additional keyword arguments to pass to the `LiteLLMModel.completion` call
                      (e.g., temperature, max_tokens). A default timeout of 120s is added if not present.

        Returns:
            Any: The response from the LiteLLM completion call (typically a `ModelResponse` object).

        Raises:
            Exception: If the API call ultimately fails after all retries and fallback attempts,
                       or if a non-retryable, non-rate-limit error occurs.
        """
        if self.current_fallback_level > 0 and self.api_call_count % 5 == 0: 
            self.reset_to_original_model()
            
        estimated_input_tokens = min(sum(len(m.get("content", "")) for m in messages) // 4, self.input_token_limit)
        
        # self.model_id reflects the current primary model for this instance (original or a stabilized fallback)
        # _apply_rate_limit will check against this model_id's limits in the shared_tracker.
        waited, wait_time = self._apply_rate_limit(self.model_id, estimated_input_tokens)
        if waited:
            logger.info(f"Waited {wait_time:.2f}s before API call for {self.model_id} due to rate limit ({self.last_wait_type})")
            
        retries = 0
        backoff_time = self.base_wait_time
        current_call_kwargs = kwargs.copy()
        if "timeout" not in current_call_kwargs: current_call_kwargs["timeout"] = 120
        
        active_model_for_this_attempt = self.model_id # Start with the instance's current primary model

        while True:
            try:
                # current_call_kwargs['model'] = active_model_for_this_attempt # Ensure litellm uses this model
                response = super().__call__(messages=messages, **current_call_kwargs)
                
                token_counts = self.get_token_counts()
                input_tokens, output_tokens = token_counts.get('input_token_count', estimated_input_tokens), token_counts.get('output_token_count', 0)
                self.shared_tracker.update_tracking(active_model_for_this_attempt, input_tokens, output_tokens)
                
                if active_model_for_this_attempt != self.original_model_id and active_model_for_this_attempt == self.model_id:
                    self.fallback_history.append((datetime.now().isoformat(), self.original_model_id, active_model_for_this_attempt, "success_stabilized"))
                
                self.api_call_count += 1
                if self.api_call_count % 3 == 0: self.print_rate_limit_status(use_logger=True)
                logger.info(f"API call successful with model {active_model_for_this_attempt}. Tokens: {input_tokens+output_tokens}")
                return response
                
            except Exception as e:
                self.last_error = e
                logger.debug(f"API call error with model {active_model_for_this_attempt}: {type(e).__name__}: {str(e)[:500]}")
                is_rl_error = self._is_rate_limit_error(e)
                if is_rl_error:
                    self.shared_tracker.record_error(active_model_for_this_attempt, is_rate_limit=True)
                    if self.enable_fallback:
                        new_fallback_candidate = self._get_fallback_model(active_model_for_this_attempt)
                        if new_fallback_candidate and new_fallback_candidate != active_model_for_this_attempt:
                            logger.warning(f"Rate limit for {active_model_for_this_attempt}. Switching to fallback: {new_fallback_candidate}")
                            self.fallback_history.append((datetime.now().isoformat(), self.original_model_id, active_model_for_this_attempt, f"failed_switching_to_{new_fallback_candidate}"))
                            self.model_id = new_fallback_candidate # Update instance's primary model ID
                            active_model_for_this_attempt = new_fallback_candidate # Use new fallback for next attempt in loop
                            try:
                                with open(self.model_info_path_for_fallback, 'r') as f_info: fb_model_info_json = json.load(f_info)
                                fb_model_key = new_fallback_candidate.split('/')[-1]
                                fb_model_data = fb_model_info_json.get(fb_model_key, fb_model_info_json.get("default",{}))
                                if not isinstance(fb_model_data, dict): fb_model_data = {}
                                fb_rpm, fb_tpm, fb_rpd = fb_model_data.get("rpm_limit",15), fb_model_data.get("tpm_limit",1000000), fb_model_data.get("rpd_limit",1500)
                                if 'gemini' in new_fallback_candidate.lower():
                                    if fb_model_key == "gemini-2.0-flash" or fb_model_key == "gemini-1.5-flash": fb_rpm=min(fb_rpm,15)
                                    elif fb_model_key == "gemini-2.0-flash-lite": fb_rpm=min(fb_rpm,30)
                                    elif fb_model_key == "gemini-2.5-pro-preview-03-25": fb_rpm=min(fb_rpm,15)
                                self.shared_tracker.initialize_model(new_fallback_candidate, fb_rpm, fb_tpm, fb_rpd)
                            except Exception as init_e: logger.error(f"Failed to re-initialize tracker for new fallback {new_fallback_candidate}: {init_e}")
                            self.current_fallback_level += 1
                            retries, backoff_time = 0, self.base_wait_time
                            waited_fb, wait_time_fb = self._apply_rate_limit(active_model_for_this_attempt, estimated_input_tokens)
                            if waited_fb: logger.info(f"Waited {wait_time_fb:.2f}s for new fallback {active_model_for_this_attempt} due to its rate limits ({self.last_wait_type})")
                            continue 
                    retries += 1
                    if retries > self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for {active_model_for_this_attempt} after rate limit error: {str(e)[:500]}")
                        raise
                    jitter = self.jitter_factor * backoff_time * (2 * random.random() - 1)
                    current_wait_time = backoff_time + jitter
                    if 'gemini' in active_model_for_this_attempt.lower(): current_wait_time = max(current_wait_time, 5.0 * (1.5 ** (retries -1)))
                    logger.warning(f"Rate limit error for {active_model_for_this_attempt}. Retry {retries}/{self.max_retries} after {current_wait_time:.2f}s. Error: {str(e)[:200]}")
                    logger.debug(f"Full error traceback: {traceback.format_exc()}")
                    time.sleep(current_wait_time)
                    backoff_time *= 1.5 
                    waited_retry, wait_time_retry = self._apply_rate_limit(active_model_for_this_attempt, estimated_input_tokens)
                    if waited_retry: logger.info(f"Additional wait of {wait_time_retry:.2f}s for {active_model_for_this_attempt} after backoff ({self.last_wait_type})")
                else: 
                    logger.error(f"API call failed with non-rate-limit error for {active_model_for_this_attempt}: {str(e)[:500]}")
                    raise
    
    def _apply_rate_limit(self, model_id_to_check: str, estimated_tokens: int = 0) -> Tuple[bool, float]:
        """Checks rate limits via `SharedRateLimitTracker` and sleeps if necessary.
        
        This helper method is called before each API call attempt. It uses
        `shared_tracker.check_rate_limits` to determine if a wait is needed
        due to RPM, TPM, RPD, min_delay, or cooldowns for the `model_id_to_check`.
        If a wait is required, this method sleeps for the suggested duration
        (with a small jitter) before returning.

        Args:
            model_id_to_check (str): The model ID whose limits are being checked.
            estimated_tokens (int): Estimated tokens for the upcoming call. Currently, this
                                    is not directly used by `check_rate_limits` for pre-emptive
                                    TPM blocking, as TPM is based on past usage.

        Returns:
            Tuple[bool, float]: 
                - `waited` (bool): True if a sleep occurred, False otherwise.
                - `wait_duration_seconds` (float): The actual duration slept in seconds.
                                                   Returns 0 if no wait occurred.
        """
        self.last_wait_type = "None" 
        would_exceed, limit_type, wait_time = self.shared_tracker.check_rate_limits(model_id_to_check)
        if would_exceed and wait_time is not None: 
            self.last_wait_type = limit_type or "unknown"
            logger.info(f"Rate limit ({limit_type}) for {model_id_to_check} would be exceeded. Suggested wait: {wait_time:.2f}s")
            jitter = random.uniform(-0.1, 0.1) * wait_time
            adjusted_wait_time = max(0.1, wait_time + jitter)
            time.sleep(adjusted_wait_time)
            self.last_wait_time = adjusted_wait_time 
            return True, adjusted_wait_time
        return False, 0
        
    def print_rate_limit_status(self, use_logger=False):
        """Prints or logs the current detailed rate limit status for the instance's active model.
        
        Fetches status from `SharedRateLimitTracker.get_rate_limit_status` for `self.model_id`.
        Displays current RPM, TPM, RPD usage against both safe (buffered) and original API limits.
        Also shows percentages, min_delay for the model, fallback status (if enabled and active),
        and any active cooldown period with remaining time and consecutive error count.
        Highlights if RPM/TPM/RPD usage exceeds 85% of safe limits.

        Args:
            use_logger (bool): If True, logs the status using `logger.info`.
                               Otherwise, prints to the console. Defaults to False.
        """
        status = self.shared_tracker.get_rate_limit_status(self.model_id) 
        status_lines = [
            f"Rate Limit Status for {self.model_id}:",
            f"  RPM: {status['rpm']['usage']}/{status['rpm']['limit']} ({status['rpm']['percentage']:.1f}%) [Original API Limit: {status['rpm']['original']}]",
            f"  TPM: {status['tpm']['usage']}/{status['tpm']['limit']} ({status['tpm']['percentage']:.1f}%) [Original API Limit: {status['tpm']['original']}]",
            f"  RPD: {status['rpd']['usage']}/{status['rpd']['limit']} ({status['rpd']['percentage']:.1f}%) [Original API Limit: {status['rpd']['original']}]",
            f"  Min Delay between calls for this model: {status.get('min_delay', 'N/A'):.2f}s"
        ]
        if self.enable_fallback:
            current_model_for_fallback_status = self.model_id
            original_model_for_fallback_status = self.original_model_id
            if self.current_fallback_level > 0 or current_model_for_fallback_status != original_model_for_fallback_status:
                status_lines.append(f"  Model Fallback: Active (Level {self.current_fallback_level})")
                status_lines.append(f"  Currently Using: {current_model_for_fallback_status} (Originally: {original_model_for_fallback_status})")
                status_lines.append(f"  Total Fallbacks This Session: {len(self.fallback_history)}")
            else: status_lines.append(f"  Model Fallback: Enabled but not active (Using original: {original_model_for_fallback_status})")
        else: status_lines.append(f"  Model Fallback: Disabled")
        cooldown_info = status.get('cooldown', {})
        if cooldown_info.get('active', False):
            status_lines.append(f"  ⚠️ Cooldown period active for {self.model_id}: {cooldown_info.get('remaining', 0):.1f}s remaining")
            status_lines.append(f"  Consecutive errors for {self.model_id}: {cooldown_info.get('consecutive_errors',0)}")
        if status['rpm']['percentage'] > 85: status_lines.append(f"  ⚠️ RPM usage for {self.model_id} is high!")
        if status['tpm']['percentage'] > 85: status_lines.append(f"  ⚠️ TPM usage for {self.model_id} is high!")
        if status['rpd']['percentage'] > 85: status_lines.append(f"  ⚠️ RPD usage for {self.model_id} is high!")
        output_func = logger.info if use_logger else print
        output_func("\n".join(status_lines))
            
    @staticmethod
    def configure_logging(level=logging.WARNING, enable_litellm_logging=False):
        """Configures global logging levels for HTTPX, HTTPCore, and LiteLLM.

        This allows quieting down verbose logging from underlying HTTP libraries
        and LiteLLM itself, which can be useful to prevent sensitive information
        (like API keys in headers, though LiteLLM aims to prevent this) from
        being logged or to reduce log noise.

        Args:
            level: The logging level to set for `httpx` and `httpcore` loggers
                   (e.g., `logging.WARNING`, `logging.ERROR`). Defaults to `logging.WARNING`.
            enable_litellm_logging (bool): If True, enables LiteLLM's internal verbose logging.
                                           If False, attempts to disable it and sets LiteLLM's
                                           logger to ERROR. Defaults to False.
        
        Returns:
            True, indicating the configuration was applied.
        """
        logging.getLogger("httpx").setLevel(level)
        logging.getLogger("httpcore").setLevel(level)
        litellm.utils.logging_enabled = enable_litellm_logging
        os.environ["LITELLM_LOG_VERBOSE"] = str(enable_litellm_logging).lower()
        logging.getLogger("litellm").setLevel(logging.ERROR if not enable_litellm_logging else logging.INFO)
        logger.info(f"Configured logging: HTTP libs to {logging.getLevelName(level)}, LiteLLM logging {'enabled' if enable_litellm_logging else 'disabled'}")
        return True