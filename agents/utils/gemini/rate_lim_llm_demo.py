#!/usr/bin/env python3
"""
Example usage of RateLimitedLiteLLMModel with different models.

This script demonstrates how to use the rate-limited LLM client with different models,
including how to specify rate limits and handle retries.

Usage:
    poetry run python -m agents.utils.gemini.rate_lim_llm_demo
"""

import os
import time
import threading
import logging
from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent
from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from agents.utils.agent_utils import apply_agent_specific_templates
from agents.utils.telemetry import suppress_litellm_logs

# Suppress LiteLLM logs
suppress_litellm_logs()

# Configure logging using the static method
RateLimitedLiteLLMModel.configure_logging(level=logging.WARNING, enable_litellm_logging=False)

def create_demo_agent(model: LiteLLMModel) -> ToolCallingAgent:
    """Creates a simple agent for demonstration purposes
    
    Args:
        model: The LiteLLM model to use for the agent
        
    Returns:
        A configured demo agent
    """
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="demo_agent",
        description='A simple agent that demonstrates the rate-limited model.',
    )
    
    # Apply agent-specific templates
    apply_agent_specific_templates(agent)
    
    return agent

def test_direct_model_call(model: RateLimitedLiteLLMModel):
    """Test the model directly without using an agent
    
    Args:
        model: The rate-limited model to test
    """
    print("\n" + "="*80)
    print("Testing direct model call")
    print("="*80)
    
    # Display initial rate limit status
    model.print_rate_limit_status()
    
    # Create a simple message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what rate limiting is in one sentence."}
    ]
    
    # Call the model directly
    print("\nCalling model directly...")
    start_time = time.time()
    try:
        response = model(messages=messages)
        duration = time.time() - start_time
        print(f"Response received in {duration:.2f} seconds:")
        print(f"Content: {response.content}")
        
        # Demonstrate getting token counts directly
        token_counts = model.get_token_counts()
        print("\nToken counts from get_token_counts():")
        print(f"Input tokens: {token_counts['input_token_count']}")
        print(f"Output tokens: {token_counts['output_token_count']}")
        print(f"Total tokens: {token_counts['input_token_count'] + token_counts['output_token_count']}")
        
        # Show current API call count
        print(f"\nCurrent API call count: {model.api_call_count}")
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error after {duration:.2f} seconds: {e}")
    
    # Display rate limit status after the call
    model.print_rate_limit_status()
    
def run_demo(agent: ToolCallingAgent, model: RateLimitedLiteLLMModel, 
             prompt: str = "Explain the concept of rate limiting in API calls in 3 sentences.", 
             num_requests: int = 16):
    """Runs multiple requests to demonstrate rate limiting
    
    Args:
        agent: The agent to use for the demo
        model: The rate-limited model to monitor
        prompt: The prompt to send to the model
        num_requests: Number of requests to make
    Returns:
        None, prints results to console
    """
    print("\n" + "="*80)
    print(f"Starting demo with {num_requests} requests")
    print(f"Prompt: '{prompt}'")
    print("="*80)
    
    # Display initial rate limit status
    model.print_rate_limit_status()
    print(f"Current API call count: {model.api_call_count}")

    successful_requests = 0
    failed_requests = 0
    start_time = time.time()
    total_tokens = 0

    for i in range(num_requests):
        print(f"\n{'-'*40}")
        print(f"Request {i+1}/{num_requests}:")
        
        # Note: We don't need to manually show status every 3 requests anymore
        # as the model will automatically do this
            
        request_start = time.time()
        try:
            # Make the API call using the agent's run method
            result = agent.run(prompt)
            request_duration = time.time() - request_start
            successful_requests += 1
            
            # Get token counts
            token_counts = model.get_token_counts()
            request_tokens = token_counts['input_token_count'] + token_counts['output_token_count']
            total_tokens += request_tokens
            
            print(f"Request completed in {request_duration:.2f} seconds")
            print(f"Tokens used: {request_tokens} (input: {token_counts['input_token_count']}, output: {token_counts['output_token_count']})")
            print(f"Current API call count: {model.api_call_count}")
            
        except Exception as e:
            request_duration = time.time() - request_start
            failed_requests += 1
            print(f"Error after {request_duration:.2f} seconds: {e}")
    
    total_duration = time.time() - start_time
    
    # Display final rate limit status
    model.print_rate_limit_status()
    
    print("\n" + "="*80)
    print("Demo Summary:")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per request: {total_duration/num_requests:.2f} seconds")
    if successful_requests > 0:
        print(f"Total tokens used: {total_tokens}")
        print(f"Average tokens per request: {total_tokens/successful_requests:.1f}")
    print(f"Final API call count: {model.api_call_count}")
    print("="*80)

def test_parallel_model_calls(model_id="gemini/gemini-2.0-flash", num_models=3, num_requests=3):
    """Test multiple model instances running in parallel to demonstrate shared rate limiting
    
    Args:
        model_id: The model ID to use
        num_models: Number of model instances to create
        num_requests: Number of requests per model
    """
    print("\n" + "="*80)
    print(f"Testing shared rate limiting with {num_models} parallel model instances")
    print("="*80)
    
    # Create multiple model instances
    models = []
    for i in range(num_models):
        model = RateLimitedLiteLLMModel(
            model_id=model_id,
            model_info_path="agents/utils/gemini/gem_llm_info.json",
            base_wait_time=1.0,
            max_retries=3,
            jitter_factor=0.2,
        )
        models.append(model)
        print(f"Created model instance {i+1}")
    
    # Display initial rate limit status
    print("\nInitial rate limit status:")
    models[0].print_rate_limit_status()
    
    # Create a simple message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what rate limiting is in one sentence."}
    ]
    
    # Function to run requests on a model
    def run_model_requests(model_idx):
        model = models[model_idx]
        print(f"\nStarting requests for model instance {model_idx+1}")
        
        for i in range(num_requests):
            try:
                print(f"Model {model_idx+1}, Request {i+1}: Starting")
                start_time = time.time()
                response = model(messages=messages)
                duration = time.time() - start_time
                
                # Get token counts
                token_counts = model.get_token_counts()
                request_tokens = token_counts['input_token_count'] + token_counts['output_token_count']
                
                print(f"Model {model_idx+1}, Request {i+1}: Completed in {duration:.2f}s, {request_tokens} tokens")
                print(f"Model {model_idx+1} API call count: {model.api_call_count}")
                
                # Note: The automatic status printing will happen every 3 calls
                
            except Exception as e:
                print(f"Model {model_idx+1}, Request {i+1}: Error - {e}")
    
    # Create and start threads for each model
    threads = []
    for i in range(num_models):
        thread = threading.Thread(target=run_model_requests, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Display final rate limit status
    print("\nFinal rate limit status after parallel requests:")
    models[0].print_rate_limit_status()
    
    # Show API call counts for each model
    print("\nAPI call counts:")
    for i, model in enumerate(models):
        print(f"Model {i+1}: {model.api_call_count} calls")
    
    print("\n" + "="*80)
    print("Parallel test complete")
    print("="*80)

def main():
    """Main entry point when run directly"""

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Make sure you have a .env file with GEMINI_API_KEY set.")

    # Configure logging to suppress API key display
    print("Configuring secure logging...")
    RateLimitedLiteLLMModel.configure_logging()
    
    print("Creating rate-limited model...")
    # Create a rate-limited version of the model
    rate_limited_model = RateLimitedLiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        model_info_path="agents/utils/gemini/gem_llm_info.json",
        base_wait_time=0,  # Start with a 1 second base wait time
        max_retries=2,
        jitter_factor=0.2,
    )
    
    # First test the model directly
    test_direct_model_call(rate_limited_model)
    
    # Test parallel model instances with shared rate limiting
    test_parallel_model_calls(num_models=3, num_requests=2)
    
    # Create agents with the model
    print("\nCreating demo agent...")
    rate_limited_agent = create_demo_agent(rate_limited_model)

    # Run the demo with a moderate number of requests to demonstrate rate limiting
    run_demo(rate_limited_agent, rate_limited_model, num_requests=4)
    
    # Show how to use the rate limit status directly
    print("\nYou can also get the rate limit status as a dictionary:")
    status = rate_limited_model.get_rate_limit_status()
    print(f"RPM usage: {status['rpm']['percentage']:.1f}%")
    print(f"TPM usage: {status['tpm']['percentage']:.1f}%")
    print(f"RPD usage: {status['rpd']['percentage']:.1f}%")

if __name__ == "__main__":
    main()