import os
import time
import base64
from typing import Optional
from smolagents import tool
from dotenv import load_dotenv
import json


@tool
def generate_video_with_veo3(prompt: str, output_filename: Optional[str] = None, seed_image_path: Optional[str] = None) -> str:
    """Generate an 8s 720p video with audio using Veo 3 via Gemini API.

    Args:
        prompt: Detailed description of the scene, motion, mood, lighting, etc.
        output_filename: Optional filename (e.g., "dialogue_example.mp4"). Defaults to timestamped name.
        seed_image_path: Optional local path to a PNG/JPG used as the first frame.

    Returns:
        Path to the saved MP4 file.
    """
    from google import genai
    from google.genai import types

    # Load GEMINI_API_KEY from agents/.env explicitly
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(agents_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Fallback to default .env discovery
        load_dotenv()

    # Ensure animator data directory exists
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "animator", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Prefer GEMINI_API_KEY; fall back to GOOGLE_API_KEY if present
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Please add it to /home/rob/Dev/botlab/agents/.env as GEMINI_API_KEY=..."
        )

    client = genai.Client(api_key=api_key)

    image_arg = None
    if seed_image_path and os.path.exists(seed_image_path):
        with open(seed_image_path, "rb") as f:
            image_bytes = f.read()
        image_arg = types.Image(image_bytes=image_bytes, mime_type="image/png")

    # models: veo-3.0-generate-preview, veo-3.0-fast-generate-preview
    operation = client.models.generate_videos(
        model="veo-3.0-fast-generate-preview",
        prompt=prompt,
        image=image_arg,
    )

    # Poll until done (per docs)
    while not operation.done:
        time.sleep(10)
        operation = client.operations.get(operation)

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)

    # Determine filename
    if not output_filename:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"veo3_{ts}.mp4"
    out_path = os.path.join(data_dir, output_filename)

    generated_video.video.save(out_path)

    return out_path



@tool
def save_storyboard_metadata(data: str | dict, expected_num_scenes: Optional[int] = None) -> str:
    """Validate and save finalized storyboard metadata as JSON, returning the saved path.

    Args:
        data: The storyboard metadata to save. May be provided either as a JSON string
            or as a Python dict with the required fields described below.
        expected_num_scenes: Optional number of scenes to enforce. If provided, the
            function will raise an error if the number of scenes in `data` does not match.

    The schema is enforced to ensure downstream consistency for the animator pipeline:

    Required top-level fields:
    - title: string
    - total_duration_seconds: integer
    - scenes: list of scene objects

    Optional top-level fields:
    - character_bible: string (concise description of main character(s), outfits, props)

    Each scene object must include:
    - id: integer (1-based index preferred)
    - title: string
    - duration_seconds: integer (must be exactly 8)
    - veo_prompt: string (concise, cinematic prompt for Veo 3)
    - audio_cues: string
    - transition_from_previous: string (e.g., cut, match cut, whip pan, dissolve)
    - seed_instructions: string (describe what to carry from the previous scene's last frame)
    - notes: string

    Optional per-scene fields:
    - continuity: string ("append" to continue from last frame; "cut" for new angle)

    Notes:
    - When continuity == "append", the pipeline will seed the next scene with the previous
      scene's last frame to create a direct continuation.
    - When continuity == "cut" or omitted, the pipeline will not necessarily seed from the previous scene's last frame.

    If expected_num_scenes is provided, the number of scenes must match.
    Saves to agents/animator/data/storyboard_YYYY-MM-DD_HH-MM-SS.json.

    Returns:
        The absolute path to the saved JSON file.
    """
    # Parse input
    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Input 'data' is not valid JSON: {e}")
    elif isinstance(data, dict):
        payload = data
    else:
        raise ValueError("'data' must be a JSON string or a dict")

    # Basic schema validation
    if not isinstance(payload.get("title"), str) or not payload.get("title"):
        raise ValueError("Missing or invalid 'title'")
    if not isinstance(payload.get("total_duration_seconds"), int):
        raise ValueError("Missing or invalid 'total_duration_seconds'")

    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise ValueError("'scenes' must be a non-empty list")

    if expected_num_scenes is not None and len(scenes) != int(expected_num_scenes):
        raise ValueError(f"Storyboard must have exactly {expected_num_scenes} scenes")

    required_scene_fields = {
        "id": int,
        "title": str,
        "duration_seconds": int,
        "veo_prompt": str,
        "audio_cues": str,
        "transition_from_previous": str,
        "seed_instructions": str,
        "notes": str,
    }

    total_duration = 0
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {idx} is not an object")
        for key, typ in required_scene_fields.items():
            if key not in scene:
                raise ValueError(f"Scene {idx} missing required field '{key}'")
            if not isinstance(scene[key], typ):
                raise ValueError(f"Scene {idx} field '{key}' must be {typ.__name__}")
        if scene["duration_seconds"] != 8:
            raise ValueError(f"Scene {idx} duration_seconds must be 8")
        total_duration += scene["duration_seconds"]

    if total_duration != payload.get("total_duration_seconds"):
        raise ValueError("total_duration_seconds must equal the sum of scene durations")

    # Save file
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "animator", "data")
    os.makedirs(data_dir, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_title = payload.get("title", "story").replace(" ", "_")[:60]
    out_path = os.path.join(data_dir, f"storyboard_{safe_title}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path

