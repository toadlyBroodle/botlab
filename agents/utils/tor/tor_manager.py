import requests
import os
import socket
import subprocess
import time
import logging
import atexit
import threading
import queue
from typing import Optional, Dict, Any, Callable

# Tor configuration
TOR_PROXY = {
    'http': 'socks5://127.0.0.1:9050',
    'https': 'socks5://127.0.0.1:9050'
}

IP_CHECK_SERVICES = [
    {"url": "https://api.ipify.org?format=json", "key": "ip"},
    {"url": "https://httpbin.org/ip", "key": "origin"},
    {"url": "https://ifconfig.me/all.json", "key": "ip_addr"},
]

# Global variable to track Tor process if we started it
tor_process = None

# Session pre-warming globals
_session_pool = queue.Queue(maxsize=3)  # Pool of pre-warmed sessions - increased for better availability
_session_prep_thread = None
_session_prep_stop_event = threading.Event()
_session_init_func = None
_session_init_args = None
_prewarming_enabled = False
_session_prep_lock = threading.Lock()  # Prevent concurrent preparation attempts


def check_tor_connection() -> bool:
    """Check if Tor is running and accessible."""
    try:
        # Test if Tor SOCKS proxy is accessible
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 9050))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_tor_ip() -> Optional[str]:
    """Get current IP address through Tor to verify connection."""
    for service in IP_CHECK_SERVICES:
        try:
            response = requests.get(
                service['url'],
                proxies=TOR_PROXY,
                timeout=15  # Reduced timeout for better responsiveness
            )
            if response.status_code == 200:
                ip = response.json().get(service['key'])
                if ip:
                    return ip
        except requests.exceptions.Timeout:
            logging.debug(f"Timeout getting Tor IP from {service['url']}")
        except requests.exceptions.ConnectionError as e:
            logging.debug(f"Connection error getting Tor IP from {service['url']}: {e}")
        except Exception as e:
            logging.debug(f"Failed to get Tor IP from {service['url']}: {e}")
    return None


def get_worker_tor_proxy(worker_id: int) -> Dict[str, str]:
    """Get Tor proxy configuration with circuit isolation for a specific worker."""
    # Use SOCKS5 with stream isolation
    # Each worker gets its own credentials to force different circuits
    # Tor uses different circuits for different SOCKS authentication credentials
    username = f"worker{worker_id}"
    password = f"session{worker_id}"
    proxy_url = f"socks5://{username}:{password}@127.0.0.1:9050"
    
    return {
        'http': proxy_url,
        'https': proxy_url
    }


def get_tor_ip_for_worker(session: requests.Session, worker_id: Optional[int] = None) -> Optional[str]:
    """Get current IP address through Tor for a specific worker session."""
    for service in IP_CHECK_SERVICES:
        try:
            response = session.get(service['url'], timeout=60)
            if response.status_code == 200:
                ip = response.json().get(service['key'])
                if ip:
                    return ip
        except requests.exceptions.Timeout:
            if worker_id is not None:
                logging.debug(f"Worker {worker_id}: Timeout getting Tor IP from {service['url']}")
            else:
                logging.debug(f"Timeout getting Tor IP from {service['url']}")
        except requests.exceptions.ConnectionError as e:
            if worker_id is not None:
                logging.debug(f"Worker {worker_id}: Connection error getting Tor IP from {service['url']}: {e}")
            else:
                logging.debug(f"Connection error getting Tor IP from {service['url']}: {e}")
        except Exception as e:
            if worker_id is not None:
                logging.debug(f"Worker {worker_id}: Failed to get Tor IP from {service['url']}: {e}")
            else:
                logging.debug(f"Failed to get Tor IP from {service['url']}: {e}")
    return None


def renew_worker_circuit(worker_id: int) -> bool:
    """Request a new Tor circuit for a specific worker."""
    try:
        # Read control settings from environment (fallback to defaults)
        host = os.getenv('TOR_CONTROL_HOST', '127.0.0.1')
        port = int(os.getenv('TOR_CONTROL_PORT', '9051'))
        password = os.getenv('TOR_CONTROL_PASSWORD')

        # Connect to Tor control port and send NEWNYM signal
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        # Send authentication command
        if password:
            # Quote plain password per Tor control spec
            auth_cmd = f'AUTHENTICATE "{password}"\r\n'.encode()
        else:
            # Cookie or no-auth (legacy)
            auth_cmd = b'AUTHENTICATE\r\n'
        sock.send(auth_cmd)
        response = sock.recv(1024)
        
        if b'250 OK' not in response:
            logging.warning(f"Worker {worker_id}: Authentication failed: {response.decode()}")
            sock.close()
            return False
        
        # Send NEWNYM signal to request new circuit
        sock.send(b'SIGNAL NEWNYM\r\n')
        response = sock.recv(1024)
        sock.close()
        
        if b'250 OK' in response:
            time.sleep(3)  # Wait for circuit to establish
            logging.debug(f"Worker {worker_id}: Circuit renewed successfully")
            return True
        else:
            logging.warning(f"Worker {worker_id}: Failed to renew circuit: {response.decode()}")
            return False
            
    except Exception as e:
        logging.warning(f"Worker {worker_id}: Could not renew circuit: {e}")
        return False


def renew_tor_circuit() -> bool:
    """Request a new Tor circuit to change IP address."""
    try:
        # Read control settings from environment (fallback to defaults)
        host = os.getenv('TOR_CONTROL_HOST', '127.0.0.1')
        port = int(os.getenv('TOR_CONTROL_PORT', '9051'))
        password = os.getenv('TOR_CONTROL_PASSWORD')

        # Connect to Tor control port and send NEWNYM signal using socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        # Send authentication command
        if password:
            auth_cmd = f'AUTHENTICATE "{password}"\r\n'.encode()
        else:
            auth_cmd = b'AUTHENTICATE\r\n'
        sock.send(auth_cmd)
        response = sock.recv(1024)
        
        if b'250 OK' not in response:
            logging.warning(f"Authentication failed: {response.decode()}")
            sock.close()
            return False
        
        # Send NEWNYM signal to request new circuit
        sock.send(b'SIGNAL NEWNYM\r\n')
        response = sock.recv(1024)
        sock.close()
        
        if b'250 OK' in response:
            time.sleep(5)  # Wait for circuit to establish
            return True
        else:
            logging.warning(f"Failed to renew Tor circuit: {response.decode()}")
            return False
            
    except ConnectionRefusedError as e:
        logging.warning(f"Could not renew Tor circuit: Control port not available on 127.0.0.1:9051. "
                       f"To enable circuit renewal, add 'ControlPort 9051' and 'CookieAuthentication 0' "
                       f"to /etc/tor/torrc and restart Tor service."
                       f"E.g.: echo -e '\n# Enable control port for circuit renewal\nControlPort 9051\nCookieAuthentication 0' | sudo tee -a /etc/tor/torrc")
        return False
    except Exception as e:
        logging.warning(f"Could not renew Tor circuit: {e}")
        return False


def start_tor_if_needed() -> bool:
    """Start Tor if it's not already running. Returns True if we started it."""
    global tor_process
    
    if check_tor_connection():
        logging.info("Tor is already running")
        return False
    
    logging.info("Tor is not running. Starting Tor...")
    
    try:
        # Start Tor as a daemon with required configuration
        cmd = [
            'tor',
            '--SocksPort', '9050',
            '--ControlPort', '9051', 
            '--CookieAuthentication', '0',
            '--RunAsDaemon', '1'
        ]
        
        tor_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for Tor to start
        time.sleep(3)
        
        # Check if Tor started successfully
        if check_tor_connection():
            return True
        else:
            error_msg = "Failed to start Tor - connection check failed"
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            stop_tor()
            return False
            
    except Exception as e:
        error_msg = f"Failed to start Tor: {e}"
        logging.error(error_msg)
        print(f"ERROR: {error_msg}")
        return False


def stop_tor():
    """Stop Tor process if we started it."""
    global tor_process
    if tor_process:
        try:
            logging.info("Stopping Tor process...")
            tor_process.terminate()
            tor_process.wait()
            logging.info("Tor process stopped successfully")
            tor_process = None
        except Exception as e:
            error_msg = f"Error stopping Tor process: {e}"
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")


def configure_session_for_tor(session: requests.Session, use_tor: bool = False, auto_start: bool = False, worker_id: Optional[int] = None):
    """
    Configure a requests session to use Tor if requested.
    
    Args:
        session: The requests session to configure
        use_tor: Whether to use Tor
        auto_start: Whether to automatically start Tor if it's not running
        worker_id: Optional worker ID for isolated circuits
    
    Raises:
        ConnectionError: If Tor is requested but not available
    """
    if use_tor:
        # Try to start Tor if requested and not running
        tor_started_by_us = False
        if auto_start:
            tor_started_by_us = start_tor_if_needed()
        
        if not check_tor_connection():
            raise ConnectionError(
                "Tor is not running or not accessible on 127.0.0.1:9050. "
                "Please start Tor service first or use --auto-start-tor."
            )
        
        # Use isolated circuits for concurrent workers
        if worker_id is not None:
            session.proxies.update(get_worker_tor_proxy(worker_id))
        else:
            session.proxies.update(TOR_PROXY)
        
        # Verify Tor connection with retry logic
        tor_ip = None
        for attempt in range(3):  # Try up to 3 times
            tor_ip = get_tor_ip_for_worker(session, worker_id)
            if tor_ip:
                break
            else:
                if attempt < 2:  # Not the last attempt
                    time.sleep(2)  # Wait before retry
                    continue
        
        if tor_ip:
            if worker_id is not None:
                logging.info(f"Worker {worker_id}: Connected through Tor. IP: {tor_ip}")
            else:
                logging.info(f"Successfully connected through Tor. Current IP: {tor_ip}")
        else:
            # If IP verification fails, just warn but don't fail completely
            # As long as the SOCKS proxy is accessible, Tor should work
            warning_msg = f"Could not verify Tor IP address, but SOCKS proxy is accessible"
            if worker_id is not None:
                logging.warning(f"Worker {worker_id}: {warning_msg}")
            else:
                logging.warning(warning_msg)
            # Don't raise an exception - proceed with Tor configuration
    else:
        # Clear any existing proxy configuration
        session.proxies.clear()
        if worker_id is not None:
            logging.debug(f"Worker {worker_id}: Using direct connection (no Tor)")
        else:
            logging.info("Using direct connection (no Tor)")


def test_api_connection(session: requests.Session, test_url: str, test_params: Dict[str, Any] = None,
                       use_tor: bool = False, timeout: int = 30, max_retries: int = 0) -> bool:
    """
    Test basic connectivity to an API endpoint without full session initialization.
    
    Args:
        session: The requests session to use for testing
        test_url: URL to test connectivity to
        test_params: Optional parameters to send with the test request
        use_tor: Whether Tor is being used
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts with circuit renewal
    
    Returns:
        bool: True if API is reachable, False otherwise
    """
    if test_params is None:
        test_params = {}
    
    # If max_retries is 0, try at least once, otherwise use the specified number
    effective_max_retries = max(1, max_retries)
    
    for attempt in range(effective_max_retries):
        try:
            response = session.get(test_url, params=test_params, timeout=timeout)
            
            if response.status_code == 200:
                return True
            elif response.status_code == 429:  # Rate limited
                logging.warning(f"API connection test rate limited (attempt {attempt + 1}/{effective_max_retries})")
                if attempt < effective_max_retries - 1 and use_tor:
                    if renew_tor_circuit():
                        new_ip = get_tor_ip()
                        if new_ip:
                            logging.info(f"New Tor IP for API test: {new_ip}")
                    time.sleep(5)
                    continue
            else:
                logging.warning(f"API connection test returned status {response.status_code}")
                if attempt < effective_max_retries - 1 and use_tor:
                    if renew_tor_circuit():
                        new_ip = get_tor_ip()
                        if new_ip:
                            logging.info(f"New Tor IP for API test: {new_ip}")
                    time.sleep(5)
                    continue
                
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logging.warning(f"API connection test failed (attempt {attempt + 1}/{effective_max_retries}): {e}")
            if attempt < effective_max_retries - 1 and use_tor:
                if renew_tor_circuit():
                    new_ip = get_tor_ip()
                    if new_ip:
                        logging.info(f"New Tor IP for API test: {new_ip}")
                time.sleep(5)
                continue
                
        except Exception as e:
            logging.warning(f"API connection test failed (attempt {attempt + 1}/{effective_max_retries}): {e}")
            if attempt < effective_max_retries - 1 and use_tor:
                if renew_tor_circuit():
                    new_ip = get_tor_ip()
                    if new_ip:
                        logging.info(f"New Tor IP for API test: {new_ip}")
                time.sleep(5)
                continue
    
    error_msg = "API connection test failed after all attempts"
    logging.error(error_msg)
    print(f"ERROR: {error_msg}")
    return False


def _basic_session_init(session: requests.Session, use_tor: bool, max_retries: int) -> bool:
    """Basic session initialization for pre-warming when main module isn't available."""
    try:
        # Basic test to verify session is working
        timeout = 60 if use_tor else 30
        
        # Simple test request using multiple services
        for service in IP_CHECK_SERVICES:
            try:
                test_response = session.get(
                    service['url'],
                    timeout=timeout
                )
                
                if test_response.status_code == 200:
                    return True
            except Exception:
                continue
        
        logging.warning("Basic session initialization failed: Could not reach any IP check service")
        return False
            
    except Exception as e:
        logging.warning(f"Basic session initialization failed: {e}")
        return False


def _init_prewarmed_session(new_session: requests.Session, use_tor: bool, max_retries: int) -> bool:
    """
    Helper function to fully initialize a pre-warmed session.
    This should make the session completely ready for immediate use.
    """
    try:
        import sys
        if 'scrapers.allakiyas_scraper_v2' in sys.modules:
            scraper_module = sys.modules['scrapers.allakiyas_scraper_v2']
            old_session = scraper_module.session
            scraper_module.session = new_session
            try:
                # Use reduced retries for pre-warming to avoid long blocks
                result = _session_init_func(use_tor, min(max_retries, 2))
                return result
            finally:
                # Restore original session
                scraper_module.session = old_session
        else:
            # Module not loaded yet, create a minimal session init
            return _basic_session_init(new_session, use_tor, max_retries)
    except Exception as e:
        logging.warning(f"Error in pre-warmed session initialization: {e}")
        return False


def _session_preparation_worker():
    """
    Background worker thread that prepares fully-ready sessions for the pool.
    Sessions are prepared with fresh circuits and complete initialization.
    """
    global _session_pool, _session_prep_stop_event, _session_init_func, _session_init_args, _session_prep_lock
    
    preparation_interval = 15  # Prepare sessions more frequently
    
    while not _session_prep_stop_event.is_set():
        try:
            # Use lock to prevent concurrent preparation attempts
            with _session_prep_lock:
                # Check if we need more sessions in the pool
                current_pool_size = _session_pool.qsize()
                if current_pool_size < _session_pool.maxsize and _session_init_func:
                    
                    # Create new session with Tor configuration
                    new_session = requests.Session()
                    
                    try:
                        # Configure for Tor without auto-start (assume Tor is running)
                        # Use a more lenient approach for background preparation
                        if not check_tor_connection():
                            logging.debug("Tor connection not available for pre-warmed session, skipping")
                            new_session.close()
                            preparation_interval = min(60, preparation_interval + 5)
                            continue
                        
                        # Configure session for Tor
                        new_session.proxies.update(TOR_PROXY)
                        
                        # Always get a fresh circuit for pre-warmed sessions to ensure diversity
                        circuit_renewed = False
                        try:
                            circuit_renewed = renew_tor_circuit()
                            if circuit_renewed:
                                time.sleep(2)  # Give circuit time to establish
                        except Exception as e:
                            logging.debug(f"Circuit renewal failed during pre-warming: {e}")
                        
                        # Verify new IP (but don't fail if verification fails temporarily)
                        new_ip = get_tor_ip()
                        if not new_ip:
                            logging.debug("Could not verify Tor IP for pre-warmed session, but continuing")
                        
                        # Initialize the session using the provided function (this fetches fresh cookies)
                        init_args = _session_init_args or (True, 2)  # Default: use_tor=True, max_retries=2
                        if _init_prewarmed_session(new_session, init_args[0], init_args[1]):
                            # Session initialized successfully with fresh cookies, add to pool
                            try:
                                _session_pool.put(new_session, timeout=1)
                                
                                # Reduce preparation interval when successful
                                preparation_interval = max(10, preparation_interval - 1)
                                logging.debug(f"Successfully prepared pre-warmed session with IP: {new_ip or 'unknown'}")
                                
                            except queue.Full:
                                new_session.close()
                        else:
                            new_session.close()
                            # Increase preparation interval when failing
                            preparation_interval = min(60, preparation_interval + 5)
                            
                    except Exception as e:
                        logging.debug(f"Error configuring pre-warmed session: {e}")
                        new_session.close()
                        preparation_interval = min(60, preparation_interval + 5)
            
            # Wait before next preparation attempt
            _session_prep_stop_event.wait(timeout=preparation_interval)
                
        except Exception as e:
            error_msg = f"Error in session preparation worker: {e}"
            logging.error(error_msg)
            print(f"ERROR: {error_msg}")
            # Back off on errors
            _session_prep_stop_event.wait(timeout=30)
    
    logging.debug("Session preparation worker stopped")


def enable_session_prewarming(session_init_func: Callable[[bool, int], bool], 
                             use_tor: bool = True, max_retries: int = 3):
    """
    Enable session pre-warming with background preparation.
    
    Args:
        session_init_func: Function to initialize sessions, should accept (use_tor, max_retries) parameters
        use_tor: Whether to use Tor for pre-warmed sessions
        max_retries: Max retries for session initialization
    """
    global _session_prep_thread, _session_prep_stop_event, _session_init_func, _session_init_args, _prewarming_enabled
    
    if _prewarming_enabled:
        return
    
    _session_init_func = session_init_func
    _session_init_args = (use_tor, max_retries)
    _prewarming_enabled = True
    
    # Start background preparation thread
    _session_prep_stop_event.clear()
    _session_prep_thread = threading.Thread(
        target=_session_preparation_worker,
        name="SessionPrepWorker",
        daemon=True
    )
    _session_prep_thread.start()
    
    logging.info("Session pre-warming enabled - background preparation started")


def disable_session_prewarming():
    """Disable session pre-warming and clean up resources."""
    global _session_prep_thread, _session_prep_stop_event, _session_init_func, _session_init_args, _prewarming_enabled
    
    if not _prewarming_enabled:
        return
    
    _prewarming_enabled = False
    _session_prep_stop_event.set()
    
    # Wait for thread to stop
    if _session_prep_thread and _session_prep_thread.is_alive():
        _session_prep_thread.join(timeout=5)
    
    # Clear the pool
    while not _session_pool.empty():
        try:
            session = _session_pool.get_nowait()
            session.close()
        except queue.Empty:
            break
    
    _session_init_func = None
    _session_init_args = None
    
    logging.info("Session pre-warming disabled - background preparation stopped")


def get_prewarmed_session() -> Optional[requests.Session]:
    """
    Get a pre-warmed session from the pool if available.
    
    Returns:
        Pre-warmed session or None if pool is empty
    """
    try:
        session = _session_pool.get_nowait()
        return session
    except queue.Empty:
        return None


def get_session_pool_status() -> Dict[str, Any]:
    """
    Get status information about the session pool.
    
    Returns:
        Dictionary with pool status information
    """
    return {
        'prewarming_enabled': _prewarming_enabled,
        'pool_size': _session_pool.qsize(),
        'max_pool_size': _session_pool.maxsize,
        'worker_thread_alive': _session_prep_thread and _session_prep_thread.is_alive() if _session_prep_thread else False,
        'pool_utilization_pct': (_session_pool.qsize() / _session_pool.maxsize * 100) if _session_pool.maxsize > 0 else 0
    }


def log_session_pool_status(level: int = logging.INFO):
    """Log current session pool status."""
    status = get_session_pool_status()
    logging.log(level, f"Session pool: {status['pool_size']}/{status['max_pool_size']} sessions "
                      f"({status['pool_utilization_pct']:.0f}% full), prewarming: {status['prewarming_enabled']}, "
                      f"worker: {'alive' if status['worker_thread_alive'] else 'stopped'}")


def instant_session_hotswap(session: requests.Session, reason: str = "session hotswap", session_init_func: Callable[[bool, int], bool] = None) -> bool:
    """
    Instantly swap the current session with a pre-warmed session and refresh cookies.
    This swaps the session instantly but then refreshes cookies to ensure compatibility with new IP.
    If cookie refresh fails, the hotswap is considered failed.
    
    Args:
        session: The requests session to replace
        reason: Reason for hotswap (for logging)
        session_init_func: Function to reinitialize session and fetch fresh cookies
    
    Returns:
        bool: True if successful (including valid cookies), False if no pre-warmed session available or cookie refresh failed
    """
    if not _prewarming_enabled:
        return False
    
    prewarmed_session = get_prewarmed_session()
    if prewarmed_session:
        logging.debug(f"Starting instant hotswap for {reason}...")
        
        # Store old session for cleanup
        old_session = session
        
        # Instantly replace session attributes with pre-warmed session
        session.__dict__.update(prewarmed_session.__dict__)
        
        # Clean up old session in background to avoid blocking
        def cleanup_old_session():
            try:
                old_session.close()
            except Exception as e:
                pass
        
        cleanup_thread = threading.Thread(target=cleanup_old_session, daemon=True)
        cleanup_thread.start()
        
        # Refresh cookies with new IP if session_init_func is provided
        if session_init_func:
            logging.debug(f"Refreshing cookies for new session after hotswap...")
            
            # Try up to 3 times to get valid cookies for the new session
            cookie_refresh_success = False
            for attempt in range(3):
                try:
                    # Clear old cookies and reinitialize with new IP
                    session.cookies.clear()
                    if session_init_func(True, 2):  # Use 2 retries for cookie refresh
                        logging.debug(f"✓ Cookies refreshed successfully after hotswap (attempt {attempt + 1})")
                        cookie_refresh_success = True
                        break
                    else:
                        logging.warning(f"Cookie refresh attempt {attempt + 1} failed after hotswap")
                        if attempt < 2:  # Not the last attempt
                            time.sleep(2)  # Brief pause before retry
                except Exception as e:
                    logging.warning(f"Error in cookie refresh attempt {attempt + 1} after hotswap: {e}")
                    if attempt < 2:  # Not the last attempt
                        time.sleep(2)  # Brief pause before retry
            
            if not cookie_refresh_success:
                logging.error(f"Cookie refresh failed after all attempts - hotswap failed")
                return False
        
        logging.debug(f"Instant hotswap completed for {reason}")
        return True
    
    return False


def renew_circuit_and_reinitialize_session(session: requests.Session, 
                                         session_init_func: Callable[[bool, int], bool],
                                         use_tor: bool = True, 
                                         reason: str = "circuit renewal", 
                                         max_attempts: int = 0) -> bool:
    """
    Enhanced function to renew Tor circuit and reinitialize session with instant hotswap support.
    
    This function prioritizes instant hotswap using pre-warmed sessions,
    falling back to traditional renewal only when necessary.
    
    Args:
        session: The requests session to reinitialize
        session_init_func: Function to call for session initialization, should accept (use_tor, max_retries) parameters
        use_tor: Whether to use Tor (if False, just reinitializes session)
        reason: Reason for renewal (for logging)
        max_attempts: Maximum number of attempts for session reinitialization
    
    Returns:
        bool: True if successful, False if failed
    """
    if not use_tor:
        # If not using Tor, just try to reinitialize session
        session.cookies.clear()
        return session_init_func(False, 3)
    
    # Try instant hotswap first - this should be sub-second
    if instant_session_hotswap(session, reason, session_init_func):
        return True
    
    # No pre-warmed session available, fall back to traditional renewal
    logging.debug(f"No pre-warmed session available for {reason}, using traditional renewal...")
    
    # Traditional renewal process (blocking)
    if not renew_tor_circuit():
        error_msg = "Failed to renew Tor circuit"
        logging.error(error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Get new IP
    new_ip = get_tor_ip()
    if new_ip:
        logging.debug(f"New Tor IP for {reason}: {new_ip}")
    else:
        logging.warning(f"Could not verify new Tor IP for {reason}")
    
    # Clear old cookies and re-initialize session with new IP
    logging.debug(f"Clearing old cookies and refreshing session for new IP...")
    session.cookies.clear()
    
    # Use exponential backoff to ensure session re-initialization succeeds
    session_reinit_success = False
    reinit_attempt = 0
    
    # If max_attempts is 0, try at least once, otherwise use the specified number
    effective_max_attempts = max(1, max_attempts)
    
    while not session_reinit_success and reinit_attempt < effective_max_attempts:
        reinit_attempt += 1
        
        if session_init_func(use_tor, 3):
            new_cookies = dict(session.cookies)
            logging.debug(f"New cookies after re-initialization for {reason}: {new_cookies}")
            logging.debug(f"✓ Session re-initialized successfully for {reason}")
            session_reinit_success = True
            break
        else:
            logging.warning(f"Session re-initialization failed for {reason} (attempt {reinit_attempt})")
        
        # Exponential backoff with jitter - max delay of 5 minutes
        if not session_reinit_success:
            import random
            base_delay = min(5 * (2 ** min(reinit_attempt - 1, 6)), 300)  # Cap at 5 minutes
            jitter = random.uniform(0, base_delay * 0.1)  # Add up to 10% jitter
            total_delay = base_delay + jitter
            
            logging.warning(f"Session re-initialization failed for {reason}. Retrying in {total_delay:.1f}s (attempt {reinit_attempt + 1})")
            time.sleep(total_delay)
            
            # Try renewing circuit again every 5 failed attempts
            if reinit_attempt % 5 == 0:
                if renew_tor_circuit():
                    new_ip = get_tor_ip()
                    if new_ip:
                        logging.debug(f"New Tor IP after additional renewal for {reason}: {new_ip}")
                    session.cookies.clear()
    
    if not session_reinit_success:
        error_msg = f"Session re-initialization failed after {effective_max_attempts} attempts for {reason}!"
        logging.error(error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    return True


def setup_tor_cleanup():
    """Register cleanup function to stop Tor and session pre-warming on exit."""
    atexit.register(_cleanup_on_exit)


def _cleanup_on_exit():
    """Clean up Tor and session pre-warming resources on exit."""
    disable_session_prewarming()
    stop_tor()


