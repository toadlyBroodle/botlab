import json
import os
import re
import argparse
import google.generativeai as genai
from dotenv import load_dotenv

def format_scientific_notation(json_str):
    """Convert scientific notation in JSON string to decimal format"""
    def replace_scientific(match):
        number = float(match.group(0))
        # Format with enough precision and remove trailing zeros
        formatted = f"{number:.12f}".rstrip('0').rstrip('.')
        return formatted
    
    # Pattern to match scientific notation (e.g., 1.5e-05, 3e-05)
    pattern = r'\d+\.?\d*e[+-]\d+'
    return re.sub(pattern, replace_scientific, json_str)

def configure_genai():
    """Configures the Gemini API."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables. Please set it.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error: Could not configure Gemini API: {e}")
        return False

def list_available_models():
    """Lists all available models from the Gemini API."""
    if not configure_genai():
        return

    print("Fetching available models...")
    try:
        for m in genai.list_models():
            print(f"Name: {m.name}")
            print(f"Display Name: {m.display_name}")
            print(f"Description: {m.description}")
            print("-" * 20)
    except Exception as e:
        print(f"Error listing models: {e}")

def process_gemini_data(input_file="agents/utils/gemini/gem_llm_rates.json", output_file="agents/utils/gemini/gem_llm_info.json"):
    if not configure_genai():
        return

    try:
        with open(input_file, 'r') as f:
            gemini_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{input_file}'.")
        return

    processed_data = {}
    for model_name, model_info in gemini_data.items():
        try:
            # Extract just the model name part (e.g., "gemini-1.5-flash") for the API call
            api_model_name = model_name
            if "models/" not in api_model_name:
                api_model_name = "models/" + model_name

            model = genai.get_model(api_model_name) # this is where the model data is fetched.

            # Add the Gemini API model information to the existing dictionary
            model_info['api_info'] = {
                'name': model.name,
                'base_model_id': model.base_model_id,
                'version': model.version,
                'display_name': model.display_name,
                'description': model.description,
                'input_token_limit': model.input_token_limit,
                'output_token_limit': model.output_token_limit,
                'supported_generation_methods': model.supported_generation_methods,
                'temperature': model.temperature,
                'max_temperature': model.max_temperature,
                'top_p': model.top_p,
                'top_k': model.top_k
            }

            model_info['model_family'] = model_name.split('-')[0]
            processed_data[model_name] = model_info

        except Exception as e:
            print(f"Error fetching API info for '{model_name}': {e}")
            processed_data[model_name] = model_info # Store the partial data
            # continue  # Keep going to the next model even if one fails
            # or
            # return # stops on first error.

    try:
        # First write the JSON normally
        json_string = json.dumps(processed_data, indent=4)
        # Apply post-processing to format scientific notation
        formatted_json = format_scientific_notation(json_string)
        # Write the formatted JSON to file
        with open(output_file, 'w') as f:
            f.write(formatted_json)
        print(f"Successfully saved processed data to '{output_file}'.")
    except IOError:
        print(f"Error: Could not write to output file '{output_file}'.")
        return

def main(args):
    if args.list_models:
        list_available_models()
    else:
        process_gemini_data()

if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Process Gemini LLM info or list available models.")
    parser.add_argument("--list-models", action="store_true", help="List all available Gemini models.")
    main(parser.parse_args())
