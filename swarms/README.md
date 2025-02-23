# Swarms

## Install dependencies

```bash
# Clone and enter directory
git clone https://github.com/yourusername/botlab.git
cd botlab

# Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: `.venv\Scripts\activate`
pip install -r requirements.txt

Some teams require additional dependencies (specified in secondary `requirements.txt` files). For example, the `writer-critic` team requires the `swarm` library.

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=<your-api-key>" >> .env
```