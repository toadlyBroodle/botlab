import os
import argparse
from dotenv import load_dotenv
from agents.agent_loop import AgentLoop


def build_promoter_loop() -> AgentLoop:
    agent_configs = {
        "promoter_description": "General-purpose promotion agent using Playwright MCP to browse, login, search and reply.",
        "promoter_prompt": None,
    }

    # Add Playwright MCP tools to promoter via agent_contexts additional_tools
    from .tools import (
        pw_navigate,
        pw_click,
        pw_fill,
        pw_get_visible_text,
        pw_get_visible_html,
    )

    # Only inject these tools once. AgentLoop will create a single promoter agent instance.
    agent_contexts = {
        "promoter": {
            "additional_tools": [
                pw_navigate,
                pw_click,
                pw_fill,
                pw_get_visible_text,
                pw_get_visible_html,
            ],
            "use_rate_limiting": False,
        }
    }

    loop = AgentLoop(
        agent_sequence=["promoter"],
        max_iterations=1,
        max_steps_per_agent=30,
        model_id=os.getenv("PROMOTER_MODEL", "gemini/gemini-2.0-flash"),
        use_custom_prompts=False,
        agent_configs=agent_configs,
        agent_contexts={
            **agent_contexts,
            # Ensure Promoter uses SimpleLiteLLMModel (no rate limiting)
            "promoter": {**agent_contexts.get("promoter", {}), "use_rate_limiting": False},
        },
    )
    return loop


def main():
    parser = argparse.ArgumentParser(description="Run the general-purpose Promoter agent")
    parser.add_argument("--query", type=str, required=True, help="High-level instruction for the promoter agent")
    args = parser.parse_args()
    load_dotenv()
    loop = build_promoter_loop()
    result = loop.run(args.query)
    print(result.get("results", {}).get("promoter", "No result"))


if __name__ == "__main__":
    main()


