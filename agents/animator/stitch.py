import os
import json
import argparse
import subprocess
from typing import List, Optional


def _concat_videos_ffmpeg(scene_files: List[str], output_path: str) -> Optional[str]:
    """Concatenate MP4 scene files into a single movie using ffmpeg.

    Returns the output path if successful, else None. Tries stream copy first, then re-encode fallback.
    Skips if ffmpeg is not available.
    """
    from shutil import which
    if which("ffmpeg") is None:
        print("ffmpeg not found on PATH; aborting.")
        return None

    # Ensure all files exist
    missing = [p for p in scene_files if not os.path.exists(p)]
    if missing:
        print(f"Missing input files: {missing}")
        return None

    # Build a temporary concat list file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    concat_list_path = output_path + ".list"
    try:
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in scene_files:
                f.write(f"file '{os.path.abspath(sf)}'\n")

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
            # Fallback: re-encode
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
                print(f"ffmpeg failed to concatenate scenes: {e}.")
                return None
    finally:
        try:
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
        except Exception:
            pass


def _load_scene_files_from_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    outputs = manifest.get("outputs", [])
    return [entry.get("file") for entry in outputs if isinstance(entry, dict) and entry.get("file")]


def _discover_scene_files_by_prefix(directory: str, prefix: str) -> List[str]:
    candidates = [
        os.path.join(directory, f) for f in sorted(os.listdir(directory))
        if f.startswith(prefix) and f.endswith('.mp4')
    ]
    return candidates


def main(args: argparse.Namespace) -> None:
    scene_files: List[str] = []

    if args.manifest:
        scene_files = _load_scene_files_from_manifest(args.manifest)
    elif args.files:
        scene_files = args.files
    elif args.prefix:
        directory = args.directory or os.getcwd()
        scene_files = _discover_scene_files_by_prefix(directory, args.prefix)
    else:
        raise SystemExit("Provide one of: --manifest, --files, or --prefix.")

    if not scene_files:
        raise SystemExit("No scene files found to stitch.")

    output_path = args.output
    if not output_path:
        # Default to current directory
        output_path = os.path.join(os.getcwd(), "final_movie.mp4")

    result = _concat_videos_ffmpeg(scene_files, output_path)
    if result:
        print(f"Saved final movie to {result}")
    else:
        raise SystemExit("Failed to stitch scenes. See messages above.")


if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Stitch scene MP4s into a single movie using ffmpeg")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON containing outputs list")
    parser.add_argument("--files", nargs='*', default=None, help="Explicit list of scene files in order")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix to discover ordered scene files in a directory")
    parser.add_argument("--directory", type=str, default=None, help="Directory to search when using --prefix (defaults to cwd)")
    parser.add_argument("--output", type=str, default=None, help="Output movie path (defaults to ./final_movie.mp4)")
    main(parser.parse_args())


