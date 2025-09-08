import os
import json
import argparse
import subprocess
from typing import List, Optional, Tuple
import tempfile
import glob


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


def _ffprobe_duration_seconds(video_path: str) -> Optional[float]:
    """Return the duration of a video file in seconds using ffprobe, or None if unavailable."""
    from shutil import which
    if which("ffprobe") is None:
        print("ffprobe not found on PATH; cannot determine duration for trimming.")
        return None

    try:
        # Query format duration in seconds
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nk=1:nw=1",
                video_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration_str = result.stdout.strip()
        return float(duration_str)
    except Exception as e:
        print(f"ffprobe failed for {video_path}: {e}")
        return None


def _trim_scene_file(input_path: str, clip_start_ms: int, clip_end_ms: int) -> Optional[str]:
    """Trim a single scene by removing clip_start_ms from start and clip_end_ms from end.

    Returns path to a temporary trimmed MP4 file on success, else None.
    """
    if clip_start_ms <= 0 and clip_end_ms <= 0:
        return input_path

    from shutil import which
    if which("ffmpeg") is None:
        print("ffmpeg not found on PATH; cannot trim scenes.")
        return None

    duration_seconds = _ffprobe_duration_seconds(input_path)
    if duration_seconds is None:
        return None

    clip_start_seconds = max(0.0, clip_start_ms / 1000.0)
    clip_end_seconds = max(0.0, clip_end_ms / 1000.0)
    effective_duration = duration_seconds - clip_start_seconds - clip_end_seconds
    if effective_duration <= 0:
        print(
            f"Requested trimming ({clip_start_ms}ms start, {clip_end_ms}ms end) "
            f"exceeds or equals duration ({duration_seconds:.3f}s) for {input_path}"
        )
        return None

    # Create a temp file alongside the input for better write permissions
    tmp_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    tmp_output = os.path.join(tmp_dir, f"{base_name}.trimmed.{clip_start_ms}.{clip_end_ms}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(clip_start_seconds),
        "-i",
        input_path,
        "-t",
        str(effective_duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        tmp_output,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_output
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed to trim {input_path}: {e}")
        return None


def _parse_clip_start_end(pairs_str: str) -> List[Tuple[int, int]]:
    """Parse a comma-separated list of start:end millisecond pairs.

    Example input: "20:20,50:30,60:40"
    Returns list of (start_ms, end_ms).
    """
    if not pairs_str:
        return []
    pairs: List[Tuple[int, int]] = []
    for raw_pair in pairs_str.split(","):
        token = raw_pair.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid clip pair '{token}'. Expected 'start:end' in milliseconds.")
        start_str, end_str = token.split(":", 1)
        try:
            start_ms = int(start_str.strip())
            end_ms = int(end_str.strip())
        except ValueError as e:
            raise ValueError(f"Non-integer clip pair '{token}': {e}") from e
        if start_ms < 0 or end_ms < 0:
            raise ValueError(f"Negative values are not allowed in clip pair '{token}'.")
        pairs.append((start_ms, end_ms))
    return pairs


def _load_scene_files_from_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    outputs = manifest.get("outputs", [])
    return [entry.get("file") for entry in outputs if isinstance(entry, dict) and entry.get("file")]


def _expand_scene_file_args(items: List[str]) -> List[str]:
    """Expand any glob patterns in provided file items; preserve non-pattern entries.

    Patterns are expanded and sorted lexicographically to provide stable order.
    """
    expanded: List[str] = []
    for item in items:
        if any(ch in item for ch in ("*", "?", "[")):
            matches = sorted(glob.glob(item))
            if not matches:
                print(f"No files matched pattern: {item}")
            expanded.extend(matches)
        else:
            expanded.append(item)
    return expanded


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

    # Expand any glob patterns in scene_files to guard against unexpanded shell globs
    scene_files = _expand_scene_file_args(scene_files)
    if not scene_files:
        raise SystemExit("No scene files found after expanding patterns.")

    output_path = args.output
    if not output_path:
        # Default to final-clipped.mp4 in the common directory of the input scenes
        abs_inputs = [os.path.abspath(sf) for sf in scene_files]
        try:
            common_dir = os.path.commonpath(abs_inputs)
            if not os.path.isdir(common_dir):
                # Fallback to directory of the first input
                common_dir = os.path.dirname(abs_inputs[0]) if abs_inputs else os.getcwd()
        except Exception:
            common_dir = os.path.dirname(abs_inputs[0]) if abs_inputs else os.getcwd()
        output_path = os.path.join(common_dir, "final-clipped.mp4")

    # Ensure we do not include the intended output file among inputs (e.g., final-clipped.mp4)
    output_abs = os.path.abspath(output_path)
    scene_files = [sf for sf in scene_files if os.path.abspath(sf) != output_abs]
    if not scene_files:
        raise SystemExit("No scene files remain after excluding output file from inputs.")

    # Optionally trim each scene before concatenation using --clip-start-end pairs
    temp_trimmed_files: List[str] = []
    if args.clip_start_end:
        try:
            clip_pairs = _parse_clip_start_end(args.clip_start_end)
        except ValueError as e:
            raise SystemExit(f"Invalid --clip-start-end value: {e}")

        if not clip_pairs:
            raise SystemExit("--clip-start-end provided but no valid pairs parsed.")

        if len(clip_pairs) == 1 and len(scene_files) > 1:
            # Broadcast single pair to all scenes
            clip_pairs = clip_pairs * len(scene_files)

        if len(clip_pairs) != len(scene_files):
            raise SystemExit(
                f"Number of clip pairs ({len(clip_pairs)}) does not match number of scenes ({len(scene_files)})."
            )

        trimmed_list: List[str] = []
        for sf, (start_ms, end_ms) in zip(scene_files, clip_pairs):
            trimmed = _trim_scene_file(sf, start_ms, end_ms)
            if trimmed is None:
                # Cleanup any created temp files before exiting
                for tmp in temp_trimmed_files:
                    try:
                        if tmp and os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass
                raise SystemExit("Failed to trim scenes. See messages above.")
            if trimmed != sf:
                temp_trimmed_files.append(trimmed)
            trimmed_list.append(trimmed)
        scene_files = trimmed_list

    result = _concat_videos_ffmpeg(scene_files, output_path)
    if result:
        print(f"Saved final movie to {result}")
        # Cleanup temp trimmed files
        for tmp in temp_trimmed_files:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
    else:
        # Cleanup temp trimmed files
        for tmp in temp_trimmed_files:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        raise SystemExit("Failed to stitch scenes. See messages above.")


if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Stitch scene MP4s into a single movie using ffmpeg")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON containing outputs list")
    parser.add_argument("--files", nargs='*', default=None, help="Explicit list of scene files in order")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix to discover ordered scene files in a directory")
    parser.add_argument("--directory", type=str, default=None, help="Directory to search when using --prefix (defaults to cwd)")
    parser.add_argument("--output", type=str, default=None, help="Output movie path (defaults to final-clipped.mp4 in input directory)")
    parser.add_argument(
        "--clip-start-end",
        type=str,
        default=None,
        help="Comma-separated start:end ms pairs per scene (e.g., '20:20,50:30'). "
             "Provide one pair to apply to all scenes or N pairs for N scenes.",
    )
    main(parser.parse_args())


