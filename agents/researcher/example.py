import time
import sys
import researcher.main as researcher_main

# Example usage
# From agents directory: python -m researcher.example "Your search query here"
# Or run without arguments to use the default query

def main(query=None, telemetry=False, max_steps=8, base_wait_time=3.0, max_retries=5):
    # Enable telemetry for tracing
    run_query = researcher_main.initialize(
        enable_telemetry=telemetry, 
        max_steps=max_steps,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )

    # Use provided query or fall back to default
    if query is None:
        query = """How many people live in the Canada as of 2025?"""
    
    print(f"\nProcessing query: {query}")
    
    # Time the query execution
    start_time = time.time()
    
    # Run the query
    result = run_query(query)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("\nResult:")
    print(result)

if __name__ == "__main__":
    # Get query from command line arguments if provided
    query = sys.argv[1] if len(sys.argv) > 1 else None
    main(query) 