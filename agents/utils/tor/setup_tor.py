#!/usr/bin/env python3
"""
Tor Setup Utility for AllAkiyas Scraper

This script helps users set up and start Tor for use with the AllAkiyas scraper.
It provides instructions and basic checks for Tor installation and configuration.
"""

import subprocess
import sys
import socket
import time
import argparse
import logging
import os
from typing import List

def check_tor_installed():
    """Check if Tor is installed on the system."""
    try:
        result = subprocess.run(['tor', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ Tor is installed: {version}")
            return True
        else:
            print("✗ Tor is not installed or not in PATH")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ Tor is not installed or not in PATH")
        return False

def check_tor_running():
    """Check if Tor is currently running."""
    try:
        # Check SOCKS proxy port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        socks_result = sock.connect_ex(('127.0.0.1', 9050))
        sock.close()
        
        # Check control port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        control_result = sock.connect_ex(('127.0.0.1', 9051))
        sock.close()
        
        if socks_result == 0 and control_result == 0:
            print("✓ Tor is running (SOCKS: 9050, Control: 9051)")
            return True
        elif socks_result == 0:
            print("⚠ Tor SOCKS proxy is running but control port is not accessible")
            print("  Control port is needed for circuit renewal")
            return False
        else:
            print("✗ Tor is not running")
            return False
            
    except Exception as e:
        print(f"✗ Error checking Tor status: {e}")
        return False

def start_tor():
    """Start Tor with appropriate configuration."""
    print("Starting Tor...")
    
    try:
        # Start Tor as daemon with required configuration
        cmd = [
            'tor',
            '--SocksPort', '9050',
            '--ControlPort', '9051',
            '--CookieAuthentication', '0',  # standalone mode without cookie auth
            '--RunAsDaemon', '1'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("Tor started successfully as daemon")
            time.sleep(2)  # Give Tor time to initialize
            
            if check_tor_running():
                print("✓ Tor is now running and accessible")
            else:
                print("⚠ Tor started but may not be fully ready yet")
        else:
            print(f"Failed to start Tor: {result.stderr}")
            print("\nTry starting Tor manually:")
            print("  tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0 --RunAsDaemon 1")
        
    except subprocess.TimeoutExpired:
        print("Tor startup timed out")
    except Exception as e:
        print(f"Error starting Tor: {e}")
        print("\nTry starting Tor manually:")
        print("  tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0 --RunAsDaemon 1")

def _ensure_lines_present(lines: List[str], required: List[str]) -> List[str]:
    """Ensure each required directive is present (and not commented) replacing any existing entries."""
    existing = []
    for line in lines:
        stripped = line.strip()
        # Skip comments when checking duplicates
        if stripped.startswith('#'):
            existing.append(line)
            continue
        key = stripped.split()[0] if stripped else ''
        # If this line sets one of the required keys, we will drop it to replace with our value later
        if key in {r.split()[0] for r in required}:
            continue
        existing.append(line)
    # Append required directives
    for r in required:
        existing.append(r + "\n")
    return existing

def setup_control_port_for_renewals():
    """Configure system Tor for circuit renewals (control port + cookie auth). Requires root.

    Actions:
    - Ensure /etc/tor/torrc has:
        SocksPort 9050
        ControlPort 9051
        CookieAuthentication 1
        CookieAuthFileGroupReadable 1
    - Add current user to 'debian-tor' group so it can read the control cookie
    - Restart tor service
    """
    if os.geteuid() != 0:
        user = os.environ.get('SUDO_USER') or os.environ.get('USER') or 'your-user'
        print("\nThis operation requires root. Run:")
        print("  sudo python agents/utils/tor/setup_tor.py --setup-renewals")
        print("\nManual commands if you prefer:")
        print("  sudo cp /etc/tor/torrc /etc/tor/torrc.bak")
        print("  sudo bash -c 'cat > /etc/tor/torrc <<\nEOF\nSocksPort 9050\nControlPort 9051\nCookieAuthentication 1\nCookieAuthFileGroupReadable 1\nEOF'\n")
        print(f"  sudo usermod -aG debian-tor {user}")
        print("  sudo systemctl restart tor")
        return False

    torrc_path = '/etc/tor/torrc'
    try:
        # Backup existing config
        if os.path.exists(torrc_path):
            subprocess.run(['cp', torrc_path, f'{torrc_path}.bak'], check=False)
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(torrc_path), exist_ok=True)

        required_directives = [
            'SocksPort 9050',
            'ControlPort 9051',
            'CookieAuthentication 1',
            'CookieAuthFileGroupReadable 1',
        ]

        old_lines: List[str] = []
        if os.path.exists(torrc_path):
            with open(torrc_path, 'r', encoding='utf-8', errors='ignore') as f:
                old_lines = f.readlines()

        new_lines = _ensure_lines_present(old_lines, required_directives)
        with open(torrc_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        # Add user to debian-tor group
        user = os.environ.get('SUDO_USER') or os.environ.get('USER')
        if user:
            subprocess.run(['usermod', '-aG', 'debian-tor', user], check=False)
            print(f"✓ Added user '{user}' to 'debian-tor' group (log out/in for it to take effect)")

        # Restart Tor service
        restarted = False
        for cmd in [
            ['systemctl', 'restart', 'tor'],
            ['service', 'tor', 'restart']
        ]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                restarted = True
                print("✓ Tor service restarted")
                break
        if not restarted:
            print("⚠ Could not restart Tor via service manager. Start manually:")
            print("  tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 1 --RunAsDaemon 1")

        # Verify
        time.sleep(2)
        check_tor_running()
        return True
    except Exception as e:
        print(f"Error configuring Tor for renewals: {e}")
        return False

def install_instructions():
    """Provide installation instructions for different platforms."""
    print("\n=== Tor Installation Instructions ===")
    print("\nUbuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install tor")
    
    print("\nCentOS/RHEL/Fedora:")
    print("  sudo dnf install tor")
    print("  # or for older versions:")
    print("  sudo yum install tor")
    
    print("\nmacOS (with Homebrew):")
    print("  brew install tor")
    
    print("\nWindows:")
    print("  Download Tor Browser from https://www.torproject.org/")
    print("  Or install via Chocolatey: choco install tor")
    
    print("\nAfter installation, you can start Tor with:")
    print("  tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0")

def main():
    parser = argparse.ArgumentParser(description='Tor Setup Utility for AllAkiyas Scraper')
    parser.add_argument('--start', action='store_true', 
                       help='Start Tor with scraper-friendly configuration')
    parser.add_argument('--check', action='store_true', 
                       help='Check Tor installation and status')
    parser.add_argument('--install-help', action='store_true',
                       help='Show installation instructions')
    parser.add_argument('--setup-renewals', action='store_true',
                       help='Configure system Tor for control port + cookie auth and restart the service (requires sudo)')
    
    args = parser.parse_args()
    
    if args.install_help:
        install_instructions()
        return
    
    if args.setup_renewals:
        setup_control_port_for_renewals()
        return

    if args.check or not any([args.start, args.install_help, args.setup_renewals]):
        print("=== Tor Status Check ===")
        tor_installed = check_tor_installed()
        tor_running = check_tor_running()
        
        if not tor_installed:
            print("\nTor is not installed. Use --install-help for installation instructions.")
            return
        
        if not tor_running:
            print("\nTor is not running. Use --start to start Tor or start it manually:")
            print("  tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0")
            return
        
        print("\n✓ Tor is ready for use with the AllAkiyas scraper!")
        print("\nYou can now run the scraper with:")
        print("  python scrapers/allakiyas_scraper.py --use-tor [other options]")
    
    if args.start:
        if not check_tor_installed():
            print("Tor is not installed. Use --install-help for installation instructions.")
            return
        
        if check_tor_running():
            print("Tor is already running!")
            return
        
        start_tor()

if __name__ == "__main__":
    main()


