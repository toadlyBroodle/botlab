"""Tools for the QAQC agent."""

from typing import Dict, Any
from smolagents import tool

@tool
def select_best_output(output_number: int, output_text: str) -> Dict[str, Any]:
    """Select the best output from the comparison.
    
    Args:
        output_number: The number of the selected output (1 or 2)
        output_text: The complete text of the selected output
        
    Returns:
        A dictionary with the selection result
    """
    if output_number not in [1, 2]:
        return {
            "success": False,
            "error": f"Invalid output number: {output_number}. Must be 1 or 2."
        }
    
    return {
        "success": True,
        "selected_output_number": output_number,
        "selected_output_text": output_text
    }

# QAQC agent doesn't need special tools for now
# This file is included for consistency with other agent directories 