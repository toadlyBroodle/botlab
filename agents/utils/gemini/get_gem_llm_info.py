import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

def process_gemini_data(input_file="utils/gemini/gem_rate_lims.json", output_file="utils/gemini/gem_llm_info.json"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables. Please set it.")
        return

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error: Could not configure Gemini API: {e}")
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
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=4)
        print(f"Successfully saved processed data to '{output_file}'.")
    except IOError:
        print(f"Error: Could not write to output file '{output_file}'.")
        return

if __name__ == "__main__":
    process_gemini_data()