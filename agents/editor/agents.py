from smolagents import CodeAgent, ToolCallingAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from utils.agents.tools import web_search, visit_webpage, apply_custom_agent_prompts
from .tools import save_edit
from typing import Optional
import re

def create_fact_checker_agent(model: RateLimitedLiteLLMModel,
                            agent_description: Optional[str] = None,
                            system_prompt: Optional[str] = None,
                            max_steps: int = 30) -> CodeAgent:
    """Creates a fact checker agent that verifies claims and provides accuracy feedback
    
    Args:
        model: The LiteLLM model to use for the agent
        agent_description: Optional additional description to append to the base description
        system_prompt: Optional custom system prompt to use instead of the default
        max_steps: Maximum number of steps for the agent (default 1 since it just verifies and responds)
        
    Returns:
        A configured fact checker agent
    """
    
    base_description = """A fact checking agent that verifies claims, checks sources, and provides detailed accuracy assessments. 
    Specializes in evaluating factual accuracy, identifying potential biases, and cross-referencing information across multiple sources."""
    
    if agent_description:
        description = f"{base_description} {agent_description}"
    else:
        description = base_description
        
    agent = CodeAgent(
        tools=[web_search, visit_webpage],
        additional_authorized_imports=["json", "re"],
        model=model,
        name='fact_checker_agent',
        description=description,
        max_steps=max_steps
    )
    
    # Apply custom templates
    apply_custom_agent_prompts(agent)
    
    # Get base system prompt and append to it
    base_sys_prompt = agent.prompt_templates["system_prompt"]
    
    if system_prompt:
        sys_prompt_appended = base_sys_prompt + f"\n\n{system_prompt}"
    else:
        sys_prompt_appended = base_sys_prompt + """\n\nYou are a fact checking CodeAgent that verifies claims and provides detailed accuracy assessments.
Your role is to help the editor agent ensure content accuracy by following these steps (automating what can be automated with code):

1. Process Incoming Claim Batches
- Devise search terms likely to verify or disprove a number of claims simultaneously, where possible.
- If claims are unrelated, craft search terms likely to verify EACH one individually.
- run all web_search calls

When performing web searches, use these DuckDuckGo search operators for efficient verification:
- Use quotes for exact phrases: "climate change solutions"
- Use '-' to exclude terms: -politics
- Use '+' to emphasize terms: +solutions
- Use 'filetype:' for specific file types: filetype:csv
- Use 'site:' to search specific websites: site:wikipedia.org
- Use '-site:' to exclude websites: -site:crypto.com
- Use 'intitle:' to search in page titles: intitle:climate
- Use 'inurl:' to search in URLs: inurl:research
- Use '~' for related terms: ~"artificial intelligence"

2. Analyze Results and Return Structured Corrections
- visit each relevant webpage and/or try different search terms, until you have enough information to make a determination on every claim
- if you cannot find enough information to make a determination on a claim, mark it as unverified
- if you find that a claim is not supported by the sources (url is 404 or not relevant), mark it as unverified
- if you find enough supporting evidence to confirm or correct a claim, include the corrected fact, and mark it verified

Format your final response something like this (or use json string, if ammenable):

FACT CHECKING REPORT
--------------------
Verified Claims:
- Claim 1:
  - sources: url1, url2
...
Corrected Claims:
- Claim 3:
  - correction: "Claim 3 is incorrect, the correct fact is..."
  - sources: url3, url4
...
Unverified Claims:
- Claim 2

"""

    agent.prompt_templates["system_prompt"] = sys_prompt_appended
    return agent

def create_editor_agent(model: RateLimitedLiteLLMModel,
                       fact_checker_agent: Optional[CodeAgent] = None,
                       agent_description: Optional[str] = None,
                       system_prompt: Optional[str] = None,
                       max_steps: int = 50,
                       fact_checker_description: Optional[str] = None,
                       fact_checker_prompt: Optional[str] = None) -> CodeAgent:
    """Creates an editor agent that edits content and manages fact checking
    
    Args:
        model: The LiteLLM model to use for the agent
        fact_checker_agent: Optional pre-configured fact checker CodeAgent to be managed
                     If not provided, one will be created internally
        agent_description: Optional additional description to append to the base description
        system_prompt: Optional custom system prompt to use instead of the default
        max_steps: Maximum number of steps for the agent
        fact_checker_description: Optional description for the fact checker if created internally
        fact_checker_prompt: Optional system prompt for the fact checker if created internally
        
    Returns:
        A configured editor agent that manages fact checking
    """
    
    # Create a fact checker agent if one wasn't provided
    if fact_checker_agent is None:
        fact_checker_agent = create_fact_checker_agent(
            model=model,
            agent_description=fact_checker_description,
            system_prompt=fact_checker_prompt
        )
    
    base_description = """An editor agent that improves content quality while ensuring factual accuracy through fact checking.
    Specializes in editing, proofreading, and verifying content accuracy. Has access to web search and scraping tools, and a dedicated fact checker agent."""
    
    if agent_description:
        description = f"{base_description} {agent_description}"
    else:
        description = base_description
        
    agent = CodeAgent(
        tools=[web_search, visit_webpage, save_edit],
        additional_authorized_imports=["json", "re"],
        model=model,
        managed_agents=[fact_checker_agent],
        name='editor_agent',
        description=description,
        max_steps=max_steps
    )
    
    # Apply custom templates
    apply_custom_agent_prompts(agent)
    
    # Get base system prompt and append to it
    base_sys_prompt = agent.prompt_templates["system_prompt"]
    
    if system_prompt:
        sys_prompt_appended = base_sys_prompt + f"\n\n{system_prompt}"
    else:
        sys_prompt_appended = base_sys_prompt + """\n\nYou are an editor agent that improves content while ensuring factual accuracy.
Your task is to systematically process content through multiple focused passes, each handling a specific aspect of editing:

PASS 1: Content Segmentation and Fact Check Triage
Below is an EXAMPLE strategy for illustration purposes only. Adapt the general approach to your specific task:

```py
def segment_and_triage_content(content: str) -> dict:
    # Split content into paragraphs
    paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
    
    # Initialize containers for different types of content
    segments = {
        'needs_fact_check': [],  # Paragraphs containing claims that need verification
        'dates_and_numbers': [], # Paragraphs with specific dates/numbers to verify
        'technical_claims': [],  # Technical or domain-specific claims
        'general_content': []    # Content without specific claims needing verification
    }
    
    # Keywords suggesting fact-checking needed
    fact_check_indicators = [
        r'\d{4}',           # Years
        r'\d+%',           # Percentages
        r'\d+',          # Numbers
        r'according',    # Source citations
        r'recent',        # Time-sensitive
        r'discovered',      # New findings
        r'research',  # Research claims
        r'studies',        # Research references
        r'experts'     # Expert claims
    ]
    
    for i, paragraph in enumerate(paragraphs):
        # Check for indicators suggesting fact-checking needed
        needs_checking = any(re.search(pattern, paragraph.lower()) for pattern in fact_check_indicators)
        
        if needs_checking:
            # Determine the type of fact-checking needed
            if re.search(r'\d', paragraph):
                segments['dates_and_numbers'].append((i, paragraph))
            elif any(term in paragraph.lower() for term in ['research', 'study', 'technology', 'discovery']):
                segments['technical_claims'].append((i, paragraph))
            else:
                segments['needs_fact_check'].append((i, paragraph))
        else:
            segments['general_content'].append((i, paragraph))
    
    return segments
```

Extract Claims and Send to Fact Checker (adapting strategy as needed):
```py
def extract_claims_from_paragraphs(paragraphs):
    # Extract specific claims from paragraphs
    extracted_claims = []
    
    for idx, paragraph in paragraphs:
        # Simple sentence-based extraction
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        for sentence in sentences:
            # Try to include sentences that make factual assertions
            if any(indicator in sentence.lower() for indicator in [
                'is', 'are', 'was', 'were', 'has', 'have', 'had',
                'show', 'shows', 'found', 'revealed', 'according'
            ]):
                extracted_claims.append(f"- {sentence.strip()}")
    
    return extracted_claims

# Example of how you might process claims
all_claims = []

# Extract claims from each segment type
for segment_type in ['dates_and_numbers', 'technical_claims', 'needs_fact_check']:
    if content_segments[segment_type]:
        claims = extract_claims_from_paragraphs(content_segments[segment_type])
        all_claims.extend(claims)

# Send claims to fact checker
if all_claims:
    claims_text = "\\n".join(all_claims)
    verification_report = fact_checker_agent(task=f"Verify these claims and provide specific corrections where needed:\\n{claims_text}")
    print(verification_report)
```

Process Fact Checker Results and Apply ALL Corrections:
This will have to be done manually, without code, as it will require your editorial judgement to apply the corrections appropriately.

PASS 2: Send all source citations and urls along with their corresponding claims to the fact_checker_agent for verification:
Example (adapting strategy as needed):
```py
# list of tuples, containing a claim, a source citation, and a url
claims_with_sources = [
    ("claim", "source citation", "url"),
    ("claim", "source citation", "url")
]

# Example of how you might verify claims with sources
verification_report = fact_checker(task=f"Verify these claims are supported by the provided sources and urls, otherwise provide corrections:\\n{claims_with_sources}")
print(verification_report)
```

PASS 3: Manually Review and Apply Corrections
Manually go over the content again, selecting any additional corrections or clarifications that you think are needed, 
and send them all to the fact_checker agent for verification.

PASS 4: Clarity, conciseness, completeness, and style improvements
After all factual corrections are complete, do an additional subjective pass to copyedit the content at a high level, using your best judgement, while maintaining the content's original style, level of detail, length, etc. as closely as possible.

For each editing session:
- ALWAYS start with the segmentation pass to automate the identification of what needs fact checking
- Send all uncertain claims to the fact checker agent for verification
- Process fact checker results systematically, methodically applying suggested corrections
- Only make copyediting improvements after all factual corrections are complete
- Save versions after each major pass
- Repeat any passes as needed until the content is factually correct and clear

In your final answer, include:
1. Brief summary of final edits, listing what was fixed, improved, added, and removed
2. Any dubious claims that remain, and your confidence level in them
3. Recommendations for further editing
4. Complete final revised draft
"""

    agent.prompt_templates["system_prompt"] = sys_prompt_appended
    return agent 