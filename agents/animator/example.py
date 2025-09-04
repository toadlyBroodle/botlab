import os
import argparse
from dotenv import load_dotenv


def run_direct_tool(prompt: str, output_filename: str | None, seed_image_path: str | None) -> str:
    """Run Veo 3 generation directly via the tool, bypassing the LLM agent."""
    from .tools import generate_video_with_veo3

    # Tool loads agents/.env internally, but we also load here for consistency
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(agents_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

    out_path = generate_video_with_veo3(
        prompt=prompt,
        output_filename=output_filename,
        seed_image_path=seed_image_path,
    )
    print(out_path)
    return out_path


def run_agent(
    prompt: str,
    *,
    max_steps: int,
    model_id: str,
    model_info_path: str,
    base_wait_time: float,
    max_retries: int,
    use_rate_limiting: bool,
) -> str:
    """Run the AnimatorAgent and let it call the Veo 3 tool from the prompt."""
    from .agents import AnimatorAgent

    # Load environment first
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(agents_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

    agent = AnimatorAgent(
        max_steps=max_steps,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries,
        use_rate_limiting=use_rate_limiting,
    )

    # Nudge the agent to call the tool explicitly
    task = (
        "If the initial prompt could use improvement, improve it. Then use the improved prompt to create a concise Veo 3 prompt and CALL THE TOOL to render an 8s 720p video with audio.\n\n"
        f"PROMPT: {prompt}"
    )
    result = agent.run(task)
    print(result)
    return result


def main(args: argparse.Namespace) -> None:
    """Main function to parse arguments."""
    if args.direct_tool:
        run_direct_tool(
            prompt=args.prompt,
            output_filename=args.output_filename,
            seed_image_path=args.seed_image_path,
        )
    else:
        run_agent(
            prompt=args.prompt,
            max_steps=args.max_steps,
            model_id=args.model_id,
            model_info_path=args.model_info_path,
            base_wait_time=args.base_wait_time,
            max_retries=args.max_retries,
            use_rate_limiting=args.use_rate_limiting,
        )


if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Run the AnimatorAgent to generate Veo 3 videos.")
    parser.add_argument("--prompt", type=str, default=(
        "A single-take drone shot weaving through a rain-soaked cyberpunk market at night. "
        "Neon reflections ripple on puddles, steam rises from street food stalls, crowds and umbrellas, "
        "lens flare, shallow depth of field, smooth gimbal motion from low dolly to rising crane, "
        "moody synth ambience, cinematic color grade, 8 seconds."
    ), help="Production prompt for the video")
    parser.add_argument("--output-filename", type=str, help="Optional output mp4 filename (direct tool mode only)")
    parser.add_argument("--seed-image-path", type=str, help="Optional seed image path for first frame (direct tool mode)")
    parser.add_argument("--direct-tool", action="store_true", help="Bypass LLM; call the generation tool directly")
    parser.add_argument("--use-rate-limiting", action="store_true", help="Use RateLimitedLiteLLMModel instead of Simple")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps for the agent")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use for the agent")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    main(parser.parse_args())


