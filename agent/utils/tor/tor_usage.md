# Using Tor with AllAkiyas Scraper

This guide explains how to use Tor with the AllAkiyas scraper for enhanced privacy and to avoid potential IP blocking.

## Why Use Tor?

- **Privacy**: Hide your real IP address from the target website
- **Avoid Rate Limiting**: Rotate IP addresses to avoid being blocked
- **Geographic Diversity**: Access content as if from different locations
- **Anonymity**: Prevent tracking of scraping activities

## Prerequisites

### 1. Install Tor

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tor
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install tor
```

**macOS (with Homebrew):**
```bash
brew install tor
```

**Windows:**
- Download Tor Browser from https://www.torproject.org/
- Or install via Chocolatey: `choco install tor`

### 2. Install Python Dependencies

The scraper requires the `requests[socks]` package for SOCKS proxy support:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Setup and Usage

### Quick Setup with Utility Script

Use the provided utility script to check and start Tor:

```bash
# Check if Tor is installed and running
python utils/setup_tor.py --check

# Get installation instructions
python utils/setup_tor.py --install-help

# Start Tor with scraper-friendly configuration
python utils/setup_tor.py --start
```

### Manual Tor Setup

Start Tor with the required ports:

```bash
tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0
```

This command:
- Opens SOCKS proxy on port 9050 (for routing traffic)
- Opens control port on 9051 (for circuit renewal)
- Disables cookie authentication (for easier control access)

### Running the Scraper with Tor

Once Tor is running, use the `--use-tor` flag:

```bash
# Basic usage with Tor (requires Tor to be running)
python scrapers/allakiyas_scraper.py --use-tor --save-to-db

# Auto-start Tor and clean up when done
python scrapers/allakiyas_scraper.py --use-tor --auto-start-tor --save-to-db

# With scheduled circuit renewal (every 25 requests) + smart renewal
python scrapers/allakiyas_scraper.py --use-tor --auto-start-tor --tor-circuit-renewal 25 --save-to-db

# Full example with all options
python scrapers/allakiyas_scraper.py \
    --use-tor \
    --tor-circuit-renewal 50 \
    --delay 3.0 \
    --save-to-db \
    --setup-db \
    --verbose \
    --output-file tor_scrape_results.json
```

## Command Line Options

### Tor-Specific Options

- `--use-tor`: Enable Tor routing (requires Tor to be running)
- `--auto-start-tor`: Automatically start Tor if not running (only used with --use-tor)
- `--tor-circuit-renewal N`: Renew Tor circuit every N requests (default: 0)
  - **0 (default)**: Only renew circuit on failures (rate limits, timeouts, blocks)
  - **>0**: Also renew circuit every N requests for additional anonymity
  - **Smart renewal**: Automatically triggers on consecutive failures regardless of setting

### Recommended Settings

For **maximum anonymity**:
```bash
--use-tor --tor-circuit-renewal 10 --delay 5.0
```

For **balanced performance** (recommended):
```bash
--use-tor --delay 2.0
```

For **maximum speed** (with Tor):
```bash
--use-tor --delay 1.0
```

## How It Works

1. **Connection Setup**: The scraper configures requests to route through Tor's SOCKS proxy (127.0.0.1:9050)

2. **IP Verification**: Before starting, the scraper verifies the Tor connection by checking the external IP

3. **Smart Circuit Renewal**: Automatically requests new Tor circuits when encountering failures (rate limits, timeouts, blocks) or optionally on a schedule

4. **Auto-Start & Cleanup**: Optionally start Tor automatically and clean up when scraping completes

5. **Error Handling**: If Tor becomes unavailable, the scraper will log errors and may fall back to direct connections

## Monitoring

The scraper provides detailed logging when using Tor:

```
INFO - Configuring Tor connection...
INFO - Successfully connected through Tor. Current IP: 185.220.101.32
INFO - Renewing Tor circuit after 50 requests...
INFO - Successfully requested new Tor circuit
INFO - New Tor IP: 199.87.154.255
```

## Troubleshooting

### Common Issues

**"Tor is not running or not accessible"**
- Ensure Tor is started with the correct ports
- Check if ports 9050 and 9051 are available
- Try running: `tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0`

**"Failed to verify Tor connection"**
- Check internet connectivity
- Verify Tor is properly routing traffic
- Try restarting Tor

**Slow performance**
- Tor adds latency due to routing through multiple nodes
- Increase `--delay` to reduce load on Tor network
- Consider reducing `--tor-circuit-renewal` frequency

**Circuit renewal failures**
- Ensure control port (9051) is accessible
- Check if CookieAuthentication is disabled
- Verify Tor control port permissions

### Testing Tor Connection

You can test your Tor connection manually:

```bash
# Check your real IP
curl https://httpbin.org/ip

# Check IP through Tor
curl --socks5 127.0.0.1:9050 https://httpbin.org/ip
```

## Security Considerations

1. **Exit Node Monitoring**: Remember that Tor exit nodes can see unencrypted traffic
2. **Timing Attacks**: Avoid predictable request patterns
3. **DNS Leaks**: The scraper routes all traffic through Tor, including DNS
4. **Local Logs**: Be aware that scraping logs may contain timing information

## Performance Impact

Using Tor will:
- **Increase latency**: 2-5x slower than direct connections
- **Reduce bandwidth**: Tor network has capacity limitations
- **Add overhead**: Circuit establishment and renewal takes time

Plan accordingly and consider using longer delays between requests.

## Best Practices

1. **Respect the Tor Network**: Use reasonable delays and don't overload
2. **Monitor Performance**: Watch for timeouts and adjust settings
3. **Plan for Failures**: Have fallback strategies if Tor becomes unavailable
4. **Regular Circuit Renewal**: Change IP addresses periodically for better anonymity
5. **Combine with Other Techniques**: Use random delays, user agent rotation, etc.

## Example Session

```bash
# Terminal 1: Start Tor
tor --SocksPort 9050 --ControlPort 9051 --CookieAuthentication 0

# Terminal 2: Run scraper
source .venv/bin/activate
python scrapers/allakiyas_scraper.py \
    --use-tor \
    --auto-start-tor \
    --delay 3.0 \
    --save-to-db \
    --verbose \
    --limit-boxes 100
```

This setup will:
- Auto-start Tor if not running
- Route all traffic through Tor
- Smart circuit renewal (only on failures)
- Wait 3 seconds between requests
- Save results to database
- Process only 100 grid boxes (for testing)
- Show detailed logging
- Auto-cleanup Tor when done 


