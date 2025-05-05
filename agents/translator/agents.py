from typing import Dict, List, Optional, Union
from agents.utils.gemini.gemini_api import GeminiAPI

class TranslatorAgent:
    """Agent for translating text between languages using Gemini 2.5."""

    def __init__(
        self,
        model: Optional[str] = None,
        agent_instance: Optional[GeminiAPI] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ):
        """Initialize the translator agent.

        Args:
            model: The Gemini model to use. If None, defaults to gemini-2.5-pro-preview-03-25.
            agent_instance: An existing GeminiAPI instance to use. If provided, model is ignored.
            system_prompt: Custom system prompt to use. If None, a default is provided.
            temperature: Temperature for generation (0.0 to 1.0).
        """
        self._agent_instance = agent_instance
        self._model = model or "gemini-2.5-pro-preview-03-25"
        self._temperature = temperature
        
        default_system_prompt = """You are a language translation expert. 
Your task is to accurately translate text between languages while preserving:
- Original meaning and context
- Tone and style
- Cultural nuances and idioms
- Technical terminology (when present)
- Formatting and structure
- EXACT subtitle syntax (e.g. "00:00:00,000 --> 00:00:00,000") if present

Provide only the translated text without explanations unless specifically requested.
"""
        self._system_prompt = system_prompt or default_system_prompt

    @property
    def agent(self) -> GeminiAPI:
        """Get the underlying GeminiAPI instance."""
        if self._agent_instance is None:
            self._agent_instance = GeminiAPI(default_model=self._model)
        return self._agent_instance

    def translate(
        self, 
        text: str, 
        source_language: str,
        target_language: str,
        preserve_formatting: bool = True,
        context: Optional[str] = None,
    ) -> str:
        """Translate text from source language to target language using Gemini.
        
        Args:
            text: The text to translate
            source_language: The source language (e.g., "Japanese", "Spanish")
            target_language: The target language (e.g., "English", "French")
            preserve_formatting: Whether to preserve the original formatting
            context: Optional context about the text to improve translation quality
            
        Returns:
            The translated text
        """
        prompt = f"{self._system_prompt}\n\nTranslate the following text from {source_language} to {target_language}.\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
            
        if preserve_formatting:
            prompt += "Please preserve the original formatting, including paragraphs, bullet points, and special characters.\n\n"
            
        prompt += f"Text to translate:\n{text}"
        
        response, error = self.agent.query(
            prompt=prompt,
            model=self._model,
            temperature=self._temperature
        )
        
        if error:
            raise Exception(f"Translation error: {error}")
            
        return response.strip()
        
    def detect_language(self, text: str) -> str:
        """Detect the language of the provided text using Gemini.
        
        Args:
            text: The text to analyze
            
        Returns:
            The detected language
        """
        prompt = f"{self._system_prompt}\n\nIdentify the language of the following text. Respond with just the language name.\n\n"
        prompt += f"Text: {text}"
        
        response, error = self.agent.query(
            prompt=prompt,
            model=self._model,
            temperature=0.1  # Lower temperature for more deterministic response
        )
        
        if error:
            raise Exception(f"Language detection error: {error}")
            
        return response.strip() 