from typing import Tuple, Optional
from ..utils.agents.rate_lim_llm import RateLimitedLiteLLMModel
import re

def llm_judge(
    artifact_content_A: str,
    artifact_content_B: str,
    goal_description: str,
    llm_model: RateLimitedLiteLLMModel,
) -> Tuple[Optional[str], str]:
    """Judges which artifact is superior based on the goal using an LLM.

    Args:
        artifact_content_A: Content of the first artifact.
        artifact_content_B: Content of the second artifact.
        goal_description: The goal to evaluate against.
        llm_model: The RateLimitedLiteLLMModel instance for making the call.

    Returns:
        A tuple containing the winner ('A', 'B', or 'Equal') and the justification.
        Returns (None, "Error message") if LLM call fails or parsing is unsuccessful.
    """
    prompt = f"""Given the primary goal of '{goal_description}', which of the following two document versions is superior?

Document A:
'''
{artifact_content_A}
'''

Document B:
'''
{artifact_content_B}
'''

Consider factors like clarity, completeness, relevance to the goal, and overall quality.
Respond with only "Document A is superior", "Document B is superior", or "Both are of equal quality".
Provide a brief one-sentence justification after your choice."""

    try:
        # Create a message in the format expected by the LiteLLMModel
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Call the model directly (which uses __call__)
        response = llm_model(messages, max_tokens=150)
        
        # Extract the response text from the model output
        if isinstance(response, dict) and "choices" in response:
            response_text = response["choices"][0]["message"]["content"].strip()
        else:
            response_text = str(response).strip()

        # Extract winner
        winner = None
        if "Document A is superior".lower() in response_text.lower():
            winner = 'A'
        elif "Document B is superior".lower() in response_text.lower():
            winner = 'B'
        elif "Both are of equal quality".lower() in response_text.lower():
            winner = 'Equal'

        # Extract justification (everything after the winner phrase)
        justification = "No justification provided."
        phrases = [
            "Document A is superior",
            "Document B is superior",
            "Both are of equal quality"
        ]
        
        # Attempt to find the phrase and get text after it
        # Case-insensitive search for the phrase
        found_phrase = None
        for phrase in phrases:
            match = re.search(re.escape(phrase) + r"[.\s]*", response_text, re.IGNORECASE)
            if match:
                found_phrase = phrase
                justification_start_index = match.end()
                justification_candidate = response_text[justification_start_index:].strip()
                if justification_candidate: # If there's something after the phrase
                    justification = justification_candidate
                break # Stop after finding the first matching phrase
        
        if winner is None:
            # Fallback if structured response not found
            justification = f"Could not determine winner from LLM response: {response_text}"
            print(f"LLM Judge Warning: {justification}")
            return None, justification 
            
        return winner, justification

    except Exception as e:
        error_message = f"LLM call failed in llm_judge: {str(e)}"
        print(f"LLM Judge Error: {error_message}")
        return None, error_message 