#!/usr/bin/env python
# coding=utf-8

"""Example usage of the TranslatorAgent with Gemini 2.5."""

import sys
import os

# Add the project root directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.translator.agents import TranslatorAgent
from agents.utils.file_manager import FileManager
from agents.utils.agents.tools import save_final_answer


def main():
    """Run examples using the Gemini-powered translator agent."""
    # Create the translator agent with Gemini 2.5
    translator = TranslatorAgent(model="gemini-2.5-pro-preview-03-25")
    
    # Example 1: Simple translation
    print("Example 1: English to Spanish translation with Gemini 2.5")
    english_text = "Hello world! This is a test of the translator agent using Gemini 2.5."
    translated = translator.translate(
        text=english_text,
        source_language="English",
        target_language="Spanish"
    )
    print(f"Original: {english_text}")
    print(f"Translated: {translated}")
    print("\n" + "-" * 50 + "\n")
    
    # Example 2: Translation with context
    print("Example 2: Translation with context")
    technical_text = "The model leverages transformer architecture with attention mechanisms for natural language processing."
    technical_translation = translator.translate(
        text=technical_text,
        source_language="English",
        target_language="French",
        context="This is about machine learning and AI technology."
    )
    print(f"Original: {technical_text}")
    print(f"Translated (with context): {technical_translation}")
    print("\n" + "-" * 50 + "\n")
    
    # Example 3: Language detection
    print("Example 3: Language detection")
    unknown_text = "こんにちは、世界！これはGemini 2.5を使用したテストです。"
    detected_language = translator.detect_language(unknown_text)
    print(f"Text: {unknown_text}")
    print(f"Detected language: {detected_language}")
    
    # Save the results
    file_manager = FileManager()
    results = {
        "examples": [
            {"original": english_text, "translated": translated, "from": "English", "to": "Spanish"},
            {"original": technical_text, "translated": technical_translation, "from": "English", "to": "French"},
            {"unknown_text": unknown_text, "detected_language": detected_language}
        ],
        "model_used": "gemini-2.5-pro-preview-03-25"
    }
    save_final_answer(translator, results, "gemini_translation_examples", "translator", use_daily_master=True)
    print(f"\nResults saved to the translator/data directory.")


if __name__ == "__main__":
    main() 