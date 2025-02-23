from .main import setup_agents, run_query

def main():
    # Replace with your Gemini API key
    GEMINI_API_KEY = "your_gemini_api_key_here"
    
    # Setup the agent system
    manager = setup_agents(GEMINI_API_KEY)
    
    # Example query
    query = """If LLM training continues to scale up at the current rhythm until 2030, 
    what would be the electric power in GW required to power the biggest training runs by 2030? 
    What would that correspond to, compared to some countries?"""
    
    # Run the query
    result = run_query(manager, query)
    print(result)

if __name__ == "__main__":
    main() 