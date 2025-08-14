#!/usr/bin/env python3
"""
Example demonstrating how to use the Tor manager module in other scrapers.

This shows the basic pattern for integrating Tor functionality into any scraper.
"""

import requests
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.tor_manager import (
    configure_session_for_tor, 
    test_api_connection,
    renew_circuit_and_reinitialize_session,
    setup_tor_cleanup
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_session_example(use_tor: bool = False, max_retries: int = 3) -> bool:
    """
    Example session initialization function that your scraper would implement.
    This function should contain any site-specific setup (cookies, headers, etc.)
    
    Args:
        use_tor: Whether Tor is being used (affects timeout)
        max_retries: Maximum retry attempts
    
    Returns:
        bool: True if initialization successful
    """
    # Your scraper-specific session initialization logic here
    # For example: visit login pages, set cookies, configure headers, etc.
    logging.info("Initializing session for example scraper...")
    
    # Example: Make a test request to verify session is working
    try:
        response = session.get("https://httpbin.org/ip", timeout=30 if not use_tor else 60)
        if response.status_code == 200:
            logging.info("Session initialization successful")
            return True
    except Exception as e:
        logging.error(f"Session initialization failed: {e}")
        return False
    
    return False


def example_scraper_with_tor():
    """Example showing how to integrate Tor into a scraper."""
    
    # Create a session for your scraper
    global session
    session = requests.Session()
    
    # Set up headers specific to your target site
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Configure Tor (if needed)
    use_tor = True  # Set based on your requirements
    
    if use_tor:
        try:
            configure_session_for_tor(session, use_tor=True, auto_start=True)
            setup_tor_cleanup()  # Ensure Tor is stopped on exit
            logging.info("Tor configured successfully")
        except ConnectionError as e:
            logging.error(f"Failed to configure Tor: {e}")
            return False
    else:
        configure_session_for_tor(session, use_tor=False)
    
    # Initialize session with your scraper-specific logic
    if not renew_circuit_and_reinitialize_session(
        session, 
        initialize_session_example,
        use_tor=use_tor, 
        reason="initial setup",
        max_attempts=5
    ):
        logging.error("Failed to initialize session")
        return False
    
    # Test API connectivity
    test_params = {'format': 'json'}  # Example parameters
    if not test_api_connection(
        session, 
        "https://httpbin.org/ip", 
        test_params,
        use_tor=use_tor,
        timeout=30,
        max_retries=3
    ):
        logging.error("API connectivity test failed")
        return False
    
    # Your scraping logic here...
    consecutive_failures = 0
    circuit_renewal_threshold = 3
    
    for i in range(10):  # Example: scrape 10 pages
        try:
            # Make your scraping request
            response = session.get(f"https://httpbin.org/delay/{i % 3}", timeout=30)
            
            if response.status_code == 200:
                consecutive_failures = 0
                logging.info(f"Successfully scraped page {i+1}")
            else:
                consecutive_failures += 1
                logging.warning(f"Failed to scrape page {i+1}: HTTP {response.status_code}")
                
        except Exception as e:
            consecutive_failures += 1
            logging.error(f"Error scraping page {i+1}: {e}")
        
        # Handle circuit renewal on consecutive failures
        if use_tor and consecutive_failures >= circuit_renewal_threshold:
            logging.warning(f"Renewing circuit after {consecutive_failures} failures")
            
            if renew_circuit_and_reinitialize_session(
                session,
                initialize_session_example,
                use_tor=True,
                reason="consecutive failures",
                max_attempts=5
            ):
                consecutive_failures = 0
                logging.info("Circuit renewal successful")
            else:
                logging.error("Circuit renewal failed")
                break
    
    logging.info("Scraping completed")
    return True


if __name__ == "__main__":
    example_scraper_with_tor()


