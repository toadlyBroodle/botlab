#!/usr/bin/env python3
"""Gemini model configuration module."""

import os
import json
from typing import Dict, Any

# Path to the gem_llm_info.json file
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gem_llm_info.json")

def _load_model_info() -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    try:
        with open(_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading Gemini model config: {e}")
        # Return fallback config with basic models
        return {
            "gemini-2.5-pro-preview-03-25": {
                "rpm_limit": 15,
                "tpm_limit": 1000000,
                "rpd_limit": 25,
                "input_token_limit": 1048576,
                "output_token_limit": 8192,
                "supports_function_calling": True
            },
            "gemini-2.0-flash": {
                "rpm_limit": 15,
                "tpm_limit": 1000000,
                "rpd_limit": 1500,
                "input_token_limit": 1048576,
                "output_token_limit": 8192,
                "supports_function_calling": True
            },
            "gemini-2.0-flash-lite": {
                "rpm_limit": 30,
                "tpm_limit": 1000000,
                "rpd_limit": 1500,
                "input_token_limit": 1048576,
                "output_token_limit": 8192,
                "supports_function_calling": False
            }
        }

# Load the model information
GEMINI_MODELS = _load_model_info()

# Add missing fields to model configs if needed
for model_name, config in GEMINI_MODELS.items():
    # Extract API info if available
    api_info = config.get("api_info", {})
    
    # Set default input and output token limits if not present
    if "input_token_limit" not in config:
        config["input_token_limit"] = api_info.get("input_token_limit", 32000)
    
    if "output_token_limit" not in config:
        config["output_token_limit"] = api_info.get("output_token_limit", 8192)
    
    # Set default function calling support if not present
    if "supports_function_calling" not in config:
        config["supports_function_calling"] = True 