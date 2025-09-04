import os
import time
import base64
from typing import Optional
from smolagents import tool
from dotenv import load_dotenv


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

    operation = client.models.generate_videos(
        model="veo-3.0-generate-preview",
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


