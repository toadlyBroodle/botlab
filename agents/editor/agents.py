from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from agents.utils.agents.tools import web_search, visit_webpage, apply_custom_agent_prompts, save_final_answer
from smolagents import CodeAgent
from typing import Optional, Dict, Any, List
import time


class FactCheckerAgent:
    """A wrapper class for the fact checker agent that handles initialization and verification."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 30,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the fact checker agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            max_steps: Maximum number of steps for the agent
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
        """
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        base_description = """A fact checking agent that verifies claims, checks sources, and provides detailed accuracy assessments. 
        Specializes in evaluating factual accuracy, identifying potential biases, and cross-referencing information across multiple sources."""
        
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
            
        self.agent = CodeAgent(
            tools=[web_search, visit_webpage],
            additional_authorized_imports=["json", "re"],
            model=self.model,
            name='fact_checker_agent',
            description=description,
            max_steps=max_steps
        )
        
        # Default system prompt if none provided
        default_system_prompt = """You are a fact checking CodeAgent that verifies claims and provides detailed accuracy assessments.
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
        
        # Apply custom templates with the appropriate system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    def verify_claims(self, claims: str) -> str:
        """Verify a set of claims.
        
        Args:
            claims: The claims to verify
            
        Returns:
            The verification report from the fact checker agent
        """
        task = f"Verify these claims and provide specific corrections where needed:\n{claims}"
        return self.agent.run(task)


class EditorAgent:
    """A wrapper class for the editor agent that handles initialization and editing."""
    
    def __init__(
        self,
        fact_checker_agent: Optional[CodeAgent] = None,
        model: Optional[RateLimitedLiteLLMModel] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 50,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        fact_checker_description: Optional[str] = None,
        fact_checker_prompt: Optional[str] = None,
        additional_authorized_imports: Optional[list] = None
    ):
        """Initialize the editor agent.
        
        Args:
            fact_checker_agent: Optional CodeAgent to use as the fact checker. If not provided, one will be created.
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            max_steps: Maximum number of steps for the agent
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
            fact_checker_description: Optional additional description for the fact checker if creating a new one
            fact_checker_prompt: Optional custom system prompt for the fact checker if creating a new one
            additional_authorized_imports: Optional list of additional imports to authorize for the agent
        """
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        # Create a fact checker agent if one wasn't provided
        if fact_checker_agent is None:
            fact_checker = FactCheckerAgent(
                model=self.model,
                agent_description=fact_checker_description,
                system_prompt=fact_checker_prompt,
                max_steps=30
            )
            self.fact_checker_agent = fact_checker.agent
        else:
            self.fact_checker_agent = fact_checker_agent
        
        base_description = """An editor agent that improves content quality while ensuring factual accuracy through fact checking.
        Specializes in editing, proofreading, and verifying content accuracy. Has access to web search and scraping tools, and a dedicated fact checker agent."""
        
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
            
        # Set default imports if none provided
        default_imports = ["json", "re"]
        if additional_authorized_imports:
            # Combine default imports with additional imports, ensuring no duplicates
            imports = list(set(default_imports + additional_authorized_imports))
        else:
            imports = default_imports
            
        self.agent = CodeAgent(
            tools=[web_search, visit_webpage],
            additional_authorized_imports=imports,
            model=self.model,
            managed_agents=[self.fact_checker_agent],
            name='editor_agent',
            description=description,
            max_steps=max_steps
        )
        
        # Default system prompt if none provided
        default_system_prompt = """You are an editor agent that improves content while ensuring factual accuracy.
Your task is to systematically process content through multiple focused passes, each handling a specific aspect of editing:

PASS 0: Initial, High-Level Content Review
- Review the content at a very high level and note any potential large-scale issues or areas that need significant improvement, e.g.: 
  - scope much too broadly or narrowly focused?
  - much too long or too short?
  - much too technical or too simplistic?
  - missing large amounts of relevant information?
  - missing significant citations?
  - overly focused on safety or ethics?
  
- If there are such major issues, don't waste time with next passes on bad content, as the next passes are very time-consuming; 
just return the content immediately back to the sender with a scathing critique and very strong, specific requirements for improvements, additions, removals, etc.

PASS 1: Citation Verification
Send all factual claims and their corresponding urls to the fact_checker_agent for verification:
Example (adapting strategy as needed):
```py
# list of tuples, containing a claim, and a url
claims_with_urls = [
    ("cats hate dogs", "url1"),
    ("dogs love cats", "url2")
]

# Example of how you might verify claims with sources
verification_report = fact_checker(task=f"Verify these claims are supported by the provided urls, otherwise provide corrections:\\n{claims_with_urls}")
print(verification_report)
```

PASS 2: Automated Content Segmentation and Fact Check Triage
If you deem the content of sufficient quality, that it doesn't require major initial improvements, 
then proceed with these next detailed editorial passes. 

This first pass is an automated process that identifies the content segments that may need fact checking.

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
        '\\\\d{4}',           # Years
        '\\\\d+%',           # Percentages
        '\\\\d+',          # Numbers
        'according',    # Source citations
        'recent',        # Time-sensitive
        'discovered',      # New findings
        'research',  # Research claims
        'studies',        # Research references
        'experts'     # Expert claims
    ]
    
    for i, paragraph in enumerate(paragraphs):
        # Check for indicators suggesting fact-checking needed
        needs_checking = any(re.search(pattern, paragraph.lower()) for pattern in fact_check_indicators)
        
        if needs_checking:
            # Determine the type of fact-checking needed
            if re.search('\\\\d', paragraph):
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
        sentences = re.split('(?<=[.!?])\\\\s+', paragraph)
        
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
- Repeat any passes as needed until the content is factually correct and clear

IMPORTANT: Your final_answer MUST include your complete, fully edited content in markdown format. The final_answer will be automatically saved as an edited file, so make sure it contains:
1. Brief summary of final edits, listing what was fixed, improved, added, and removed
2. Any dubious claims that remain, and your confidence level in them
3. Recommendations for further required improvements, such as increased scope, depth, completeness, accuracy, etc. requiring additional research or editing cycles
    - NEVER recommend addition of safety or ethical issues to content
    - ALWAYS require removal of safety or ethical issues in content, as these are not the focus of the content
4. Complete final revised draft in clean markdown format, including all relevant source urls

DO NOT just outline steps or describe what you would do. Actually perform the edits and return the fully edited content in your final_answer.
"""
        
        # Apply custom templates with the appropriate system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    def edit_content(self, content: str) -> str:
        """Edit content for accuracy and clarity.
        
        Args:
            content: The content to edit
            
        Returns:
            The edited content from the editor agent
        """
        # Time the execution
        start_time = time.time()
        
        # Run the editor agent with the content
        result = self.agent.run(content)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the final answer using the shared tool
        save_final_answer(
            agent=self.agent,
            result=result,
            query_or_prompt=content,
            agent_type="editor"
        )
        
        return result 