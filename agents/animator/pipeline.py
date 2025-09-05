import os
import json
import time
import argparse
import subprocess
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def _ensure_animator_data_dir() -> str:
    """Ensure the animator data directory exists and return its absolute path."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "animator", "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _timestamp() -> str:
    """Return a human-readable timestamp for filenames."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def build_writer_prompt(initial_prompt: str, num_scenes: int) -> str:
    """Build a prompt tailored for CodeAgent Thought + <code> execution in ONE step.

    The writer must construct the storyboard and return it via final_answer in the
    same code block. Do NOT print or call any tools; no multi-step iteration.
    """
    return (
        "Task: Develop a concise, cinematic story broken into fixed 8-second scenes.\n"
        "Process:\n"
        "1) Build a storyboard with exactly the requested number of scenes.\n"
        "2) Validate durations and counts.\n"
        "3) Persist the current storyboard ONLY by calling the provided save_storyboard_metadata tool like: save_storyboard_metadata(storyboard, expected_num_scenes="
        f"{num_scenes}). Do NOT read or write files directly.\n"
        "4) Call critic_agent to review your storyboard and revise based on feedback; iterate until satisfied.\n"
        "5) When satisfied, return the final JSON via final_answer in THIS run.\n\n"
        "ABSOLUTE PROHIBITIONS (enforced):\n"
        "- Do NOT import or use os, pathlib, shutil, posixpath, or open() to access the filesystem.\n"
        "- You are NOT allowed to create directories or write files.\n"
        "- The ONLY allowed persistence is via the save_storyboard_metadata tool.\n\n"
        "Storyboard JSON fields you must include (and validate):\n"
        "- title (string)\n"
        "- total_duration_seconds (integer == scenes * 8)\n"
        "- scenes: list of {id, title, duration_seconds=8, veo_prompt, audio_cues, transition_from_previous, seed_instructions, notes}\n\n"
        f"Constraints:\n- Exactly {num_scenes} scenes; each exactly 8 seconds.\n"
        f"- Strong visual continuity; include carry-over guidance in seed_instructions.\n"
        f"- Transitions explicit in transition_from_previous.\n"
        "- Default to widescreen 16:9 framing.\n"
        "- For each scene's 'veo_prompt', follow Veo best practices: include Subject, Context, Action, Style (film/visual), Camera motion, Composition, Ambiance.\n"
        "- Clearly state if audio is present and describe it in separate sentences (for example, sound effects, music, or speech).\n"
        "- Optionally include negatives by naming elements to avoid (for example, 'wall, frame').\n\n"
        f"Initial concept: {initial_prompt}"
    )


def extract_last_frame(input_video_path: str, output_image_path: str) -> None:
    """Extract the last frame of a video as a PNG using ffmpeg.

    If ffmpeg is not available on PATH, logs a warning and returns without raising.
    Overwrites output if it exists.
    """
    # Check ffmpeg availability
    from shutil import which
    if which("ffmpeg") is None:
        print("⚠ ffmpeg not found on PATH; skipping last-frame extraction for continuity seeding.")
        return

    # Use -sseof -1 to seek to 1 second from the end, then grab 1 frame
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-1",
        "-i",
        input_video_path,
        "-vsync",
        "0",
        "-frames:v",
        "1",
        output_image_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        # Redundant safety in case PATH changed
        print("⚠ ffmpeg not found during execution; skipping last-frame extraction.")
    except subprocess.CalledProcessError as e:
        print(f"⚠ ffmpeg failed to extract last frame: {e}. Continuing without seed image.")


def run_storyboard_generation(
    initial_prompt: str,
    num_scenes: int,
    max_iterations: int,
    model_id: str,
    model_info_path: str,
    use_custom_prompts: bool = True,
) -> Dict[str, Any]:
    """Use WriterAgent (with internal CriticAgent) directly to produce storyboard JSON."""
    # Lazy import to avoid heavy deps on module import
    try:
        from ..writer_critic.agents import WriterAgent
        from ..utils.agents.simple_llm import SimpleLiteLLMModel
    except ImportError:
        from agents.writer_critic.agents import WriterAgent
        from agents.utils.agents.simple_llm import SimpleLiteLLMModel

    prompt_for_writer = build_writer_prompt(initial_prompt, num_scenes)

    # Veo-focused system prompt for writer to iterate with critic and then finalize JSON
    writer_system_prompt = (
        "You are a screenplay writer agent. "
        "Always respond with a Thought: section followed by a single <code> Python block. "
        "In the code: build a Python dict 'storyboard' with required fields; "
        "assert len(scenes) matches the requested count and all duration_seconds == 8; "
        "after each draft iteration (before calling the critic), persist the current storyboard ONLY by calling the save_storyboard_metadata tool: "
        "save_path = save_storyboard_metadata(storyboard, expected_num_scenes={num_scenes}); "
        "call critic_agent with your current storyboard to get feedback, revise accordingly, and iterate until you are satisfied; "
        "when satisfied, call final_answer(json.dumps(storyboard)). "
        "CRITICAL: You are NOT allowed to read or write files, create directories, or otherwise access the filesystem. "
        "Do NOT import or use os, pathlib, shutil, posixpath, or open(). The ONLY allowed persistence is via save_storyboard_metadata. "
        "For each scene's 'veo_prompt', adhere to Veo best practices: explicitly include Subject, Context, Action, Style (film/visual), Camera motion, Composition, and Ambiance; "
        "default framing to widescreen 16:9 unless otherwise required; state audio desires in separate sentences (sound effects, music, speech) and, if useful, name elements to avoid as negatives (e.g., 'wall, frame'). "
        "Ensure continuity by carrying forward key visual elements in 'seed_instructions' and specify transitions in 'transition_from_previous'."
    )

    # Create a SimpleLiteLLMModel and pass it to WriterAgent so both writer and critic use it
    simple_model = SimpleLiteLLMModel(
        model_id=model_id,
        model_info_path=model_info_path,
    )

    writer = WriterAgent(
        model=simple_model,
        max_steps=max_iterations or 10,
        system_prompt=writer_system_prompt,
        model_id=model_id,
        model_info_path=model_info_path,
    )

    writer_output = writer.write_draft(prompt_for_writer)

    # Writer should have returned JSON via final_answer
    storyboard: Optional[Dict[str, Any]] = None
    if isinstance(writer_output, str):
        try:
            storyboard = json.loads(writer_output)
        except json.JSONDecodeError:
            last_brace = writer_output.rfind("{")
            if last_brace != -1:
                try:
                    storyboard = json.loads(writer_output[last_brace:])
                except json.JSONDecodeError:
                    storyboard = None

    if storyboard is None:
        raise RuntimeError("Writer did not return valid storyboard JSON via final_answer.")

    # Basic validation
    scenes = storyboard.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != num_scenes:
        raise ValueError("Storyboard does not contain the expected number of scenes")
    for idx, scene in enumerate(scenes, start=1):
        if int(scene.get("duration_seconds", 0)) != 8:
            raise ValueError(f"Scene {idx} duration is not 8 seconds")

    return storyboard


def generate_scenes_with_veo(
    storyboard: Dict[str, Any],
    output_prefix: Optional[str] = None,
) -> List[str]:
    """Generate each 8s scene with Veo 3, injecting last frame as seed for continuity.

    Returns a list of absolute file paths to the generated scene mp4 files.
    """
    from .tools import generate_video_with_veo3

    data_dir = _ensure_animator_data_dir()
    ts = _timestamp()
    movie_basename = output_prefix or f"story_{ts}"

    scene_files: List[str] = []
    last_frame_path: Optional[str] = None

    for i, scene in enumerate(storyboard["scenes"], start=1):
        veo_prompt: str = scene.get("veo_prompt", "").strip()
        if not veo_prompt:
            raise ValueError(f"Scene {i} missing 'veo_prompt'")

        output_filename = f"{movie_basename}_scene_{i:02d}.mp4"
        output_path = generate_video_with_veo3(
            prompt=veo_prompt,
            output_filename=output_filename,
            seed_image_path=last_frame_path,
        )

        abs_output_path = output_path if os.path.isabs(output_path) else os.path.join(data_dir, output_path)
        scene_files.append(abs_output_path)

        # Extract last frame for next scene
        last_frame_path = os.path.join(data_dir, f"{movie_basename}_scene_{i:02d}_last.png")
        extract_last_frame(abs_output_path, last_frame_path)

    return scene_files


def save_metadata(storyboard: Dict[str, Any], scene_files: List[str]) -> str:
    """Save storyboard and scene file mapping to a JSON manifest in animator/data and return its path."""
    data_dir = _ensure_animator_data_dir()
    ts = _timestamp()
    manifest = {
        "created_at": ts,
        "title": storyboard.get("title"),
        "total_duration_seconds": storyboard.get("total_duration_seconds"),
        "scenes": storyboard.get("scenes"),
        "outputs": [{"index": i + 1, "file": path} for i, path in enumerate(scene_files)],
    }
    manifest_path = os.path.join(data_dir, f"{storyboard.get('title','story').replace(' ', '_')}_{ts}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def regenerate_scenes_from_manifest(
    manifest_path: str,
    storyboard_override_path: Optional[str],
    start_scene: int,
    replace_mode: str = "single",
) -> str:
    """Regenerate scenes starting at a specific scene using an existing manifest.

    Args:
        manifest_path: Path to existing manifest JSON produced by this pipeline
        storyboard_override_path: Optional path to a storyboard JSON to override prompts/scenes
        start_scene: 1-based scene index to begin regeneration
        replace_mode: 'single' to regenerate only that scene, 'forward' to regenerate that scene and all following scenes

    Returns:
        Path to the updated manifest JSON (same file, updated in place)
    """
    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_obj = json.load(f)

    # Derive storyboard
    if storyboard_override_path:
        with open(storyboard_override_path, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
    else:
        storyboard = {
            "title": manifest_obj.get("title"),
            "total_duration_seconds": manifest_obj.get("total_duration_seconds"),
            "scenes": manifest_obj.get("scenes", []),
        }

    scenes = storyboard.get("scenes") or []
    outputs = manifest_obj.get("outputs") or []

    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard scenes missing or invalid in manifest/override")
    if not isinstance(outputs, list) or len(outputs) != len(scenes):
        raise ValueError("Manifest outputs do not align with storyboard scenes count")

    num_scenes = len(scenes)
    if start_scene < 1 or start_scene > num_scenes:
        raise ValueError(f"start_scene must be between 1 and {num_scenes}")
    if replace_mode not in {"single", "forward"}:
        raise ValueError("replace_mode must be 'single' or 'forward'")

    # Build ordered list of scene files from manifest
    # Ensure sorted by index (manifest stores index 1-based)
    outputs_sorted = sorted(outputs, key=lambda o: int(o.get("index", 0)))
    scene_files: List[str] = [o.get("file") for o in outputs_sorted]

    data_dir = _ensure_animator_data_dir()

    # Determine indices to regenerate
    if replace_mode == "single":
        indices_to_regen = [start_scene]
    else:
        indices_to_regen = list(range(start_scene, num_scenes + 1))

    # Prepare continuity seed for the first regenerated scene using the kept previous scene
    last_frame_path: Optional[str] = None
    if start_scene > 1:
        prev_scene_file = scene_files[start_scene - 2]
        if prev_scene_file and os.path.exists(prev_scene_file):
            # Extract last frame from previous existing scene to seed the first regenerated scene
            prev_basename = os.path.basename(prev_scene_file)
            seed_base = os.path.splitext(prev_basename)[0] + "_last.png"
            last_frame_path = os.path.join(data_dir, seed_base)
            extract_last_frame(prev_scene_file, last_frame_path)

    # Generate new scenes
    from .tools import generate_video_with_veo3

    # Try to derive a stable movie basename from existing files
    # Fallback to title_timestamp if not available
    first_existing = next((p for p in scene_files if isinstance(p, str) and p), None)
    if first_existing:
        movie_basename = os.path.splitext(os.path.basename(first_existing))[0]
        # strip trailing _scene_XX if present
        if "_scene_" in movie_basename:
            movie_basename = movie_basename.split("_scene_", 1)[0]
    else:
        ts = _timestamp()
        movie_basename = (storyboard.get("title", "story").replace(" ", "_") or "story") + f"_{ts}"

    for idx in indices_to_regen:
        scene = scenes[idx - 1]
        veo_prompt: str = str(scene.get("veo_prompt", "")).strip()
        if not veo_prompt:
            raise ValueError(f"Scene {idx} missing 'veo_prompt'")

        # Prefer overwriting the existing file path if available for this index
        existing_path = scene_files[idx - 1]
        if existing_path:
            output_filename = os.path.basename(existing_path)
        else:
            output_filename = f"{movie_basename}_scene_{idx:02d}.mp4"

        # Generate with optional seed for continuity
        output_path = generate_video_with_veo3(
            prompt=veo_prompt,
            output_filename=output_filename,
            seed_image_path=last_frame_path,
        )

        abs_output_path = output_path if os.path.isabs(output_path) else os.path.join(data_dir, output_path)
        scene_files[idx - 1] = abs_output_path

        # Update seed for next iteration if regenerating forward
        last_frame_path = os.path.join(data_dir, f"{movie_basename}_scene_{idx:02d}_last.png")
        extract_last_frame(abs_output_path, last_frame_path)

    # Rebuild final concatenated movie if possible
    # Try to reuse previous final movie path if present
    final_movie_path = None
    if isinstance(manifest_obj.get("final_movie"), str):
        final_movie_path = manifest_obj["final_movie"]
    else:
        final_movie_basename = (storyboard.get("title", "story").replace(" ", "_")) + "_final.mp4"
        final_movie_path = os.path.join(data_dir, final_movie_basename)

    concatenated_path = _concat_videos_ffmpeg(scene_files, final_movie_path)

    # Persist updated manifest in place
    manifest_obj["title"] = storyboard.get("title", manifest_obj.get("title"))
    manifest_obj["total_duration_seconds"] = storyboard.get("total_duration_seconds", manifest_obj.get("total_duration_seconds"))
    manifest_obj["scenes"] = scenes
    manifest_obj["outputs"] = [{"index": i + 1, "file": path} for i, path in enumerate(scene_files)]
    if concatenated_path:
        manifest_obj["final_movie"] = concatenated_path

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=False, indent=2)

    print(f"Updated manifest saved to {manifest_path}")
    if concatenated_path:
        print(f"Saved final movie to {concatenated_path}")
    else:
        print("⚠ Skipped saving final movie (ffmpeg unavailable or concat failed).")

    return manifest_path


def _concat_videos_ffmpeg(scene_files: List[str], output_path: str) -> Optional[str]:
    """Concatenate MP4 scene files into a single movie using ffmpeg.

    Returns the output path if successful, else None. Tries stream copy first, then re-encode fallback.
    Skips if ffmpeg is not available.
    """
    from shutil import which
    if which("ffmpeg") is None:
        print("⚠ ffmpeg not found on PATH; skipping final movie concatenation.")
        return None

    # Build a temporary concat list file
    concat_list_path = output_path + ".list"
    try:
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in scene_files:
                # ffmpeg concat demuxer requires escaped paths in single quotes
                f.write(f"file '{sf}'\n")

        # Try fast stream copy
        cmd_fast = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd_fast, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError:
            # Fallback: re-encode to ensure compatibility
            cmd_encode = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
            try:
                subprocess.run(cmd_encode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return output_path
            except subprocess.CalledProcessError as e:
                print(f"⚠ ffmpeg failed to concatenate scenes: {e}.")
                return None
    finally:
        try:
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
        except Exception:
            pass


def main(args: argparse.Namespace) -> None:
    """Main CLI to generate a storyboard and per-scene videos with continuity."""
    # Load environment
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(agents_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

    # If continuing from an existing manifest, perform partial regeneration and exit
    if getattr(args, "existing_manifest", None):
        if not args.start_scene:
            raise ValueError("--start-scene is required when using --existing-manifest")
        regenerate_scenes_from_manifest(
            manifest_path=args.existing_manifest,
            storyboard_override_path=args.existing_storyboard,
            start_scene=args.start_scene,
            replace_mode=args.replace_mode,
        )
        return

    # Guard: fresh run requires a prompt
    if not args.prompt:
        raise ValueError("--prompt is required unless --existing-manifest is provided")

    storyboard = run_storyboard_generation(
        initial_prompt=args.prompt,
        num_scenes=args.num_scenes,
        max_iterations=args.max_iterations,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        use_custom_prompts=True,
    )

    # Persist storyboard immediately
    data_dir = _ensure_animator_data_dir()
    storyboard_path = os.path.join(data_dir, f"storyboard_{_timestamp()}.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)
    print(f"Saved storyboard to {storyboard_path}")

    # Optionally stop after producing the storyboard
    if getattr(args, "storyboard_only", False):
        print("Skipping video generation and concatenation; storyboard-only mode enabled.")
        return

    # Generate scenes
    scene_files = generate_scenes_with_veo(storyboard, output_prefix=args.output_prefix)

    # Concatenate scenes into final movie (if ffmpeg available)
    data_dir = _ensure_animator_data_dir()
    final_movie_basename = (args.output_prefix or storyboard.get("title", "story")).replace(" ", "_")
    final_movie_path = os.path.join(data_dir, f"{final_movie_basename}_final.mp4")
    concatenated_path = _concat_videos_ffmpeg(scene_files, final_movie_path)
    if concatenated_path:
        print(f"Saved final movie to {concatenated_path}")
    else:
        print("⚠ Skipped saving final movie (ffmpeg unavailable or concat failed).")

    # Save manifest (include final movie if present)
    manifest_path = save_metadata(storyboard, scene_files)
    if concatenated_path:
        # Update manifest to include final movie path
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_obj = json.load(f)
            manifest_obj["final_movie"] = concatenated_path
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_obj, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Failed to record final movie in manifest: {e}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Storyboard to Veo pipeline with writer-critic and continuity seeding")
    parser.add_argument("--prompt", type=str, required=False, help="Initial concept prompt for the story (omit when using --existing-manifest)")
    parser.add_argument("--num-scenes", type=int, default=4, help="Number of 8-second scenes to create")
    parser.add_argument("--max-iterations", type=int, default=10, help="Writer improvement iterations/steps with critic")
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output scene filenames")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID for writer/critic agents")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--storyboard-only", action="store_true", help="Only generate and save the storyboard JSON; skip video generation")
    # Partial regeneration options
    parser.add_argument("--existing-manifest", type=str, help="Path to an existing manifest JSON to continue from")
    parser.add_argument("--existing-storyboard", type=str, help="Optional path to a storyboard JSON to override scene prompts")
    parser.add_argument("--start-scene", type=int, help="1-based scene index to regenerate from (required with --existing-manifest)")
    parser.add_argument("--replace-mode", type=str, default="single", choices=["single", "forward"], help="Regenerate only the specified scene or it and all following scenes")
    main(parser.parse_args())


