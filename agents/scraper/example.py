import time
import scraper.main as scraper_main

# Example usage
# poetry run python -m scraper.example

def main():
    # Enable telemetry for tracing
    run_query = scraper_main.initialize(enable_telemetry=False)

    # Example query
    query = """How many people live in Canada?"""
    
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
    main() 