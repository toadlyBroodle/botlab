# Installing BotLab Dependencies

This directory contains a script to easily set up a virtual environment and install all Python dependencies required by BotLab.

## Quick Installation

Run the following commands to set up the environment and install all dependencies:

```bash
# Make the script executable (if needed)
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

## What Does It Do?

The script:
1. Creates a `.venv` directory at the root of the project
2. Installs all dependencies from `requirements.txt`
3. Installs local packages (`agents` and `swarms`) in development mode

## Usage

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

If you encounter permission issues, make sure the script is executable:

```bash
chmod +x setup_env.sh
```

For manual installation:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install local packages
pip install -e ./agents
#pip install -e ./swarms
``` 

If you prefer to use Poetry you may use the `poetry.toml` file or `install_dependencies.sh` script.

## Setting Up User Feedback Agent

The user feedback agent requires a dedicated system user (`fb_agent`) for handling email communication. Follow these steps to set it up:

### 1. Create the fb_agent User

```bash
# Create the new user without login capabilities
sudo adduser --disabled-password --gecos "Feedback Agent" fb_agent

# Add user to mail group
sudo usermod -a -G mail fb_agent
```

### 2. Set Up Email Configuration

Configure email forwarding for the fb_agent user:

```bash
# Set up email forwarding in Postfix virtual file
sudo sh -c 'echo "fb_agent@botlab.dev fb_agent" >> /etc/postfix/virtual'
sudo postmap /etc/postfix/virtual

# Create and set permissions for mailbox directory
sudo mkdir -p /var/mail/fb_agent
sudo chown fb_agent:mail /var/mail/fb_agent
sudo chmod 660 /var/mail/fb_agent

# Update Postfix configuration
sudo postconf -e "home_mailbox = /var/mail/"
sudo systemctl restart postfix
```

### 3. Set Up Permissions for Your Application User

Add your application user to the mail group so it can read the fb_agent's mailbox:

```bash
# Replace 'your_username' with your actual username
sudo usermod -a -G mail your_username

# You'll need to log out and back in for this change to take effect
```

### 4. Configure Environment Variables

Add the following to your `.env` file:

```
# User email for receiving feedback reports
LOCAL_USER_EMAIL=your_email@example.com

# External email that will be sending commands
REMOTE_USER_EMAIL=external_user@example.com
```

### 5. Test the Setup

Run the example script to test if everything is properly configured:

```bash
python -m agents.user_feedback.example
```

The output should show that:
- The fb_agent user exists
- The mailbox exists and is accessible
- Your user has at least read access to the mailbox