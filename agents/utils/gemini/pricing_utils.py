#!/usr/bin/env python3
"""Comprehensive pricing utilities for Gemini models with support for tiered token pricing."""

import json
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PricingTier:
    """Represents a pricing tier with token thresholds and costs."""
    threshold: Optional[int]  # None for unlimited/highest tier
    input_cost_per_token_cents: float
    output_cost_per_token_cents: float
    tier_name: str

@dataclass 
class ModelPricing:
    """Complete pricing information for a model including all tiers."""
    model_name: str
    pricing_type: str  # 'flat', 'tiered', 'multimodal'
    tiers: list[PricingTier]
    audio_input_cost_per_token_cents: Optional[float] = None  # For multimodal models

def load_comprehensive_pricing_from_json(model_info_path: str) -> Dict[str, ModelPricing]:
    """Load comprehensive model pricing information with all tiers from gem_llm_info.json.
    
    Args:
        model_info_path: Path to the gem_llm_info.json file
        
    Returns:
        Dictionary mapping model names to ModelPricing objects with all tier information
    """
    pricing_data = {}
    
    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        for model_name, model_data in model_info.items():
            if 'cost_info' not in model_data:
                continue
                
            cost_info = model_data['cost_info']
            tiers = []
            
            # Handle different pricing structures
            if "input_cost_per_token_cents" in cost_info:
                # Simple flat pricing
                tiers.append(PricingTier(
                    threshold=None,
                    input_cost_per_token_cents=cost_info["input_cost_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_per_token_cents"],
                    tier_name="flat"
                ))
                pricing_data[model_name] = ModelPricing(
                    model_name=model_name,
                    pricing_type="flat",
                    tiers=tiers
                )
                
            elif "input_cost_le_200k_per_token_cents" in cost_info:
                # 200k threshold pricing (like gemini-2.5-pro)
                tiers.append(PricingTier(
                    threshold=200000,
                    input_cost_per_token_cents=cost_info["input_cost_le_200k_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_le_200k_per_token_cents"],
                    tier_name="≤200k"
                ))
                tiers.append(PricingTier(
                    threshold=None,
                    input_cost_per_token_cents=cost_info["input_cost_gt_200k_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_gt_200k_per_token_cents"],
                    tier_name=">200k"
                ))
                pricing_data[model_name] = ModelPricing(
                    model_name=model_name,
                    pricing_type="tiered",
                    tiers=tiers
                )
                
            elif "input_cost_le_128k_per_token_cents" in cost_info:
                # 128k threshold pricing (like gemini-1.5-flash)
                tiers.append(PricingTier(
                    threshold=128000,
                    input_cost_per_token_cents=cost_info["input_cost_le_128k_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_le_128k_per_token_cents"],
                    tier_name="≤128k"
                ))
                tiers.append(PricingTier(
                    threshold=None,
                    input_cost_per_token_cents=cost_info["input_cost_gt_128k_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_gt_128k_per_token_cents"],
                    tier_name=">128k"
                ))
                pricing_data[model_name] = ModelPricing(
                    model_name=model_name,
                    pricing_type="tiered",
                    tiers=tiers
                )
                
            elif "input_cost_text_image_video_per_token_cents" in cost_info:
                # Multi-modal pricing (like gemini-2.5-flash)
                tiers.append(PricingTier(
                    threshold=None,
                    input_cost_per_token_cents=cost_info["input_cost_text_image_video_per_token_cents"],
                    output_cost_per_token_cents=cost_info["output_cost_per_token_cents"],
                    tier_name="text/image/video"
                ))
                pricing_data[model_name] = ModelPricing(
                    model_name=model_name,
                    pricing_type="multimodal",
                    tiers=tiers,
                    audio_input_cost_per_token_cents=cost_info.get("input_cost_audio_per_token_cents")
                )
                
            logger.debug(f"Loaded pricing for {model_name}: {pricing_data[model_name].pricing_type} with {len(tiers)} tiers")
            
    except Exception as e:
        logger.error(f"Failed to load comprehensive pricing from {model_info_path}: {e}")
        
    return pricing_data

def calculate_tiered_cost(model_pricing: ModelPricing, input_tokens: int, output_tokens: int, 
                         is_audio_input: bool = False) -> Tuple[float, str]:
    """Calculate cost using appropriate pricing tier based on token counts.
    
    Args:
        model_pricing: ModelPricing object with tier information
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        is_audio_input: Whether input tokens are audio (for multimodal models)
        
    Returns:
        Tuple of (total_cost_cents, tier_description)
    """
    if not model_pricing.tiers:
        logger.warning(f"No pricing tiers available for model {model_pricing.model_name}")
        return 0.0, "no_pricing"
        
    total_tokens = input_tokens + output_tokens
    
    # For flat pricing, use the single tier
    if model_pricing.pricing_type == "flat":
        tier = model_pricing.tiers[0]
        input_cost = input_tokens * tier.input_cost_per_token_cents
        output_cost = output_tokens * tier.output_cost_per_token_cents
        return input_cost + output_cost, tier.tier_name
        
    # For multimodal pricing, check audio vs text/image/video
    elif model_pricing.pricing_type == "multimodal":
        tier = model_pricing.tiers[0]  # Only one tier for multimodal
        
        if is_audio_input and model_pricing.audio_input_cost_per_token_cents is not None:
            input_cost = input_tokens * model_pricing.audio_input_cost_per_token_cents
            tier_name = "audio"
        else:
            input_cost = input_tokens * tier.input_cost_per_token_cents
            tier_name = tier.tier_name
            
        output_cost = output_tokens * tier.output_cost_per_token_cents
        return input_cost + output_cost, tier_name
        
    # For tiered pricing, find appropriate tier based on total tokens
    elif model_pricing.pricing_type == "tiered":
        # Sort tiers by threshold (None threshold should be last)
        sorted_tiers = sorted(model_pricing.tiers, key=lambda t: t.threshold or float('inf'))
        
        selected_tier = None
        for tier in sorted_tiers:
            if tier.threshold is None or total_tokens <= tier.threshold:
                selected_tier = tier
                break
                
        if selected_tier is None:
            # Fallback to last tier if no match found
            selected_tier = sorted_tiers[-1]
            
        input_cost = input_tokens * selected_tier.input_cost_per_token_cents
        output_cost = output_tokens * selected_tier.output_cost_per_token_cents
        
        logger.debug(f"Model {model_pricing.model_name}: {total_tokens} total tokens using tier '{selected_tier.tier_name}' "
                    f"(input: ${input_cost/100:.6f} USD, output: ${output_cost/100:.6f} USD)")
        
        return input_cost + output_cost, selected_tier.tier_name
        
    else:
        logger.warning(f"Unknown pricing type {model_pricing.pricing_type} for model {model_pricing.model_name}")
        return 0.0, "unknown"

def get_model_pricing_summary(model_pricing: ModelPricing) -> str:
    """Get a human-readable summary of model pricing tiers.
    
    Args:
        model_pricing: ModelPricing object
        
    Returns:
        String describing the pricing structure
    """
    if model_pricing.pricing_type == "flat":
        tier = model_pricing.tiers[0]
        avg_cost = (tier.input_cost_per_token_cents + tier.output_cost_per_token_cents) / 2
        return f"~${avg_cost * 1000:.5f}/1K tokens avg"
        
    elif model_pricing.pricing_type == "multimodal":
        tier = model_pricing.tiers[0]
        return f"${tier.input_cost_per_token_cents * 1000:.5f}/1K input • ${tier.output_cost_per_token_cents * 1000:.5f}/1K output"
        
    elif model_pricing.pricing_type == "tiered":
        tier_descriptions = []
        for tier in sorted(model_pricing.tiers, key=lambda t: t.threshold or float('inf')):
            avg_cost = (tier.input_cost_per_token_cents + tier.output_cost_per_token_cents) / 2
            tier_descriptions.append(f"{tier.tier_name}: ${avg_cost * 1000:.5f}/1K avg")
        return " | ".join(tier_descriptions)
        
    return "Unknown pricing"

def get_legacy_pricing_for_compatibility(model_pricing: ModelPricing) -> Dict[str, float]:
    """Get legacy-format pricing for backward compatibility with existing code.
    
    Returns the lowest tier pricing in the old format for models that have tiered pricing.
    This ensures existing code continues to work while we transition to tiered pricing.
    
    Args:
        model_pricing: ModelPricing object
        
    Returns:
        Dictionary in old format with input_cost_per_token_cents and output_cost_per_token_cents
    """
    if not model_pricing.tiers:
        return {
            "input_cost_per_token_cents": 0.0,
            "output_cost_per_token_cents": 0.0
        }
        
    # For flat and multimodal, use the single tier
    if model_pricing.pricing_type in ["flat", "multimodal"]:
        tier = model_pricing.tiers[0]
        return {
            "input_cost_per_token_cents": tier.input_cost_per_token_cents,
            "output_cost_per_token_cents": tier.output_cost_per_token_cents
        }
        
    # For tiered, use the lowest tier (best price) for compatibility
    elif model_pricing.pricing_type == "tiered":
        # Find the tier with the lowest threshold (best pricing)
        lowest_tier = min(model_pricing.tiers, key=lambda t: t.threshold or float('inf'))
        return {
            "input_cost_per_token_cents": lowest_tier.input_cost_per_token_cents,
            "output_cost_per_token_cents": lowest_tier.output_cost_per_token_cents
        }
        
    return {
        "input_cost_per_token_cents": 0.0,
        "output_cost_per_token_cents": 0.0
    } 