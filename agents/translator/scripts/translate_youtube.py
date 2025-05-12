#!/usr/bin/env python3
# coding=utf-8
"""
YouTube Video Subtitle Translator

This script:
1. Downloads a YouTube video using yt-dlp
2. Extracts the subtitles
3. Translates them to English using Gemini 2.5
4. Opens the video in MPV with translated subtitles
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
import re
import shlex
import time
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dotenv import load_dotenv

# Determine script location and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")

# Load environment variables from .env file in agents directory
ENV_PATH = os.path.join(AGENTS_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print(f"Loaded environment variables from {ENV_PATH}")
else:
    print(f"Warning: No .env file found at {ENV_PATH}")

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

from agents.translator.agents import TranslatorAgent

# Constants
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos")
SUBTITLE_FORMATS = ["vtt", "srt"]
DEFAULT_SUBTITLE_LANGS = "en"  # Default: download English subtitles
GEMINI_MODEL = "gemini-2.0-flash"  # Default Gemini model to use

# Ensure the Google API key is set
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set. Translation will fail.")
    print(f"Please add your API key to {ENV_PATH}")

def ensure_directory_exists(directory: str) -> None:
    """Ensure the specified directory exists."""
    os.makedirs(directory, exist_ok=True)


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    try:
        # Check for yt-dlp
        yt_dlp_version = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        print(f"Found yt-dlp version: {yt_dlp_version}")

        # Check for MPV
        mpv_version = subprocess.run(
            ["mpv", "--version"], capture_output=True, text=True
        )
        if mpv_version.returncode == 0:
            mpv_info = mpv_version.stdout.splitlines()[0] if mpv_version.stdout else "MPV player"
            print(f"Found MPV: {mpv_info}")
        else:
            print("MPV not found. Please install MPV media player.")
            print("  sudo apt install mpv  # On Ubuntu/Debian")
            print("  brew install mpv      # On macOS with Homebrew")
            return False

        return True
    except FileNotFoundError as e:
        print(f"Dependency check failed: {e}")
        print("Please make sure yt-dlp and MPV are installed:")
        print("  pip install -U yt-dlp")
        print("  sudo apt install mpv  # On Ubuntu/Debian")
        print("  brew install mpv      # On macOS with Homebrew")
        return False


def get_terminal_width() -> int:
    """Get the width of the terminal."""
    try:
        columns, _ = shutil.get_terminal_size()
        return columns
    except (AttributeError, ValueError, OSError):
        return 80  # Default width


def create_progress_bar(percentage: float, width: int = 40) -> str:
    """Create a progress bar string.
    
    Args:
        percentage: Progress percentage (0-100)
        width: Width of the progress bar in characters
        
    Returns:
        String representation of the progress bar
    """
    filled_width = int(width * percentage / 100)
    bar = '█' * filled_width + '-' * (width - filled_width)
    return f"[{bar}] {percentage:.1f}%"


def get_video_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from a URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID or None if not found
    """
    # Check various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def find_existing_video(output_dir: str, video_id: str) -> Optional[str]:
    """
    Check if a video with the given ID is already downloaded.
    
    Args:
        output_dir: Directory to check for existing videos
        video_id: YouTube video ID
        
    Returns:
        Path to existing video file or None if not found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Add a marker to the filename to identify this video ID
    marker = f"_vid_{video_id}_"
    
    # Check all files in the output directory
    for filename in os.listdir(output_dir):
        if marker in filename:
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                    return file_path
    
    return None


def get_language_code(language_name: str) -> str:
    """
    Convert a language name to its ISO language code.
    
    Args:
        language_name: Name of the language (e.g., "English", "Japanese")
        
    Returns:
        ISO language code (e.g., "en", "ja")
    """
    language_codes = {
        "english": "en",
        "spanish": "es", 
        "french": "fr",
        "german": "de",
        "italian": "it",
        "japanese": "ja",
        "auto": "auto"
    }
    
    return language_codes.get(language_name.lower(), language_name.lower())


def download_video(url: str, output_dir: str, subtitle_lang: str = DEFAULT_SUBTITLE_LANGS, force_download: bool = False) -> Tuple[Optional[str], List[str]]:
    """
    Download video and subtitles using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the downloaded files
        subtitle_lang: Language code(s) for subtitles to download (comma-separated, or "all" for all available languages)
        force_download: Force download even if video already exists
        
    Returns:
        Tuple containing (video_path, list_of_subtitle_paths)
    """
    ensure_directory_exists(output_dir)
    
    # Check if video is already downloaded
    video_id = get_video_id(url)
    if not force_download and video_id:
        existing_video = find_existing_video(output_dir, video_id)
        if existing_video:
            print(f"Video already downloaded: {os.path.basename(existing_video)}")
            
            # Find existing subtitle files
            video_base = os.path.splitext(existing_video)[0]
            subtitle_paths = []
            
            for ext in SUBTITLE_FORMATS:
                # Get all subtitle files for this video
                for file in os.listdir(os.path.dirname(existing_video)):
                    if file.startswith(os.path.basename(video_base)) and file.endswith(f".{ext}"):
                        full_path = os.path.join(os.path.dirname(existing_video), file)
                        subtitle_paths.append(full_path)
            
            if subtitle_paths:
                print(f"Found existing subtitle files: {', '.join(os.path.basename(p) for p in subtitle_paths)}")
                
                # Check if we already have the requested subtitle languages
                requested_langs = subtitle_lang.split(",") if subtitle_lang != "all" else []
                missing_langs = []
                
                if subtitle_lang != "all" and requested_langs:
                    for lang in requested_langs:
                        if not any(f".{lang}." in os.path.basename(p).lower() for p in subtitle_paths):
                            missing_langs.append(lang)
                    
                    if missing_langs:
                        print(f"Missing subtitle files for languages: {', '.join(missing_langs)}. Will download additional subtitles.")
                    else:
                        return existing_video, subtitle_paths
                else:
                    return existing_video, subtitle_paths
            else:
                print("No subtitle files found for existing video. Will download subtitles only.")
    
    # Create unique output template with timestamp and video_id
    timestamp = int(time.time())
    output_template = os.path.join(output_dir, f"%(title)s_vid_{video_id}_{timestamp}.%(ext)s")
    
    # Set subtitle languages to download
    sub_langs = subtitle_lang
    
    # Download command with subtitle extraction
    cmd = [
        "yt-dlp",
        "--newline",  # Force progress output to contain newlines
        "--progress",  # Show progress bar
        "--write-auto-sub",  # Write automatically generated subtitles
        "--write-sub",  # Write available subtitles
        "--sub-format", "vtt,srt",  # Prefer VTT format, fallback to SRT
        "--sub-langs", sub_langs,  # Download specified subtitle languages
        "--embed-subs",  # Embed subtitles in the video file
        "--restrict-filenames",  # Restrict filenames to ASCII
        "-o", output_template,  # Output filename template
        url  # YouTube URL
    ]
    
    # Add option for subtitle only download if we have an existing video
    if not force_download and existing_video:
        print("Downloading subtitles only...")
        cmd.insert(1, "--skip-download")  # Skip downloading the video
        # For subtitle-only downloads, make sure they match the existing video filename
        cmd.extend([
            "--output-na-placeholder", "",  # Don't add placeholders to filenames
            "--paths", os.path.dirname(existing_video),  # Save to the same directory
            "--output", os.path.splitext(os.path.basename(existing_video))[0],  # Use existing filename base
        ])
    else:
        print(f"Downloading video from: {url}")
        print("This may take a while depending on the video size...")
        
    # Before downloading, try to list available subtitles
    list_cmd = ["yt-dlp", "--list-subs", url]
    try:
        print("Checking available subtitle languages...")
        result = subprocess.run(list_cmd, capture_output=True, text=True)
        if "Available subtitles" in result.stdout:
            print(result.stdout.split("Available subtitles")[1].strip())
    except Exception as e:
        print(f"Error listing subtitles: {e}")
    
    try:
        # Use Popen to get real-time output for progress display
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Process output in real-time for progress display
        terminal_width = get_terminal_width()
        last_line = ""
        downloaded_files = []
        video_path = None
        
        # Process the output
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            
            # Capture the file path from output
            if '[Merger] Merging formats into' in line:
                potential_path = line.split('into "')[-1].rstrip('"')
                if os.path.exists(potential_path):
                    video_path = potential_path
                    downloaded_files.append(potential_path)
            
            # Track other downloaded files (subtitles)
            elif 'Destination:' in line:
                potential_path = line.split('Destination: ')[-1]
                if os.path.exists(potential_path):
                    downloaded_files.append(potential_path)
            
            # Only update terminal if we have actual progress info
            if '[download]' in line and '%' in line:
                # Extract percentage using regex
                percentage_match = re.search(r'(\d+\.\d+)%', line)
                if percentage_match:
                    percentage = float(percentage_match.group(1))
                    
                    # Extract download speed and ETA if available
                    speed_match = re.search(r'at\s+([^ ]+)', line)
                    speed = speed_match.group(1) if speed_match else "unknown speed"
                    
                    eta_match = re.search(r'ETA\s+([^ ]+)', line)
                    eta = eta_match.group(1) if eta_match else "unknown ETA"
                    
                    # Create a progress bar
                    progress_bar = create_progress_bar(percentage, width=min(50, terminal_width - 30))
                    status = f"{progress_bar} {speed} ETA: {eta}"
                    
                    # Clear previous line and print new progress
                    print(f"\r{' ' * len(last_line)}\r{status}", end='', flush=True)
                    last_line = status
            elif line and not line.startswith('[debug]'):
                # Regular output - print on new line
                print(f"\r{' ' * len(last_line)}\r{line}")
                last_line = line
        
        # Wait for process to complete
        process.wait()
        
        # Print a newline after progress bar completes
        if last_line:
            print()
            
        # Check return code
        if process.returncode != 0:
            print(f"Error: yt-dlp exited with code {process.returncode}")
            return None, []
        
        # If we haven't found the video path from the output, try to find it based on the template
        if not video_path:
            # Look for video files in the output directory matching our timestamp
            for filename in os.listdir(output_dir):
                if str(timestamp) in filename and os.path.isfile(os.path.join(output_dir, filename)):
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                        video_path = os.path.join(output_dir, filename)
                        break
        
        if not video_path:
            print("Warning: Could not determine video path from output")
            # Return the first file that looks like a video if we found any downloaded files
            for file_path in downloaded_files:
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                    video_path = file_path
                    break
            
            if not video_path:
                print("Error: No video file found in downloads")
                return None, []
        
        # Find subtitle files (they'll have same base name as video)
        video_base = os.path.splitext(video_path)[0]
        subtitle_paths = []
        
        for ext in SUBTITLE_FORMATS:
            # Look for both direct and language-coded subtitle files
            possible_paths = [
                f"{video_base}.{ext}",
                f"{video_base}.{subtitle_lang}.{ext}",
                f"{video_base}.auto.{ext}",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    subtitle_paths.append(path)
        
        print(f"Download complete: {os.path.basename(video_path)}")
        return video_path, subtitle_paths
    
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, []


def parse_vtt_subtitles(vtt_path: str) -> List[Dict[str, Any]]:
    """
    Parse VTT subtitle file into a list of subtitle entries.
    
    Args:
        vtt_path: Path to VTT subtitle file
        
    Returns:
        List of subtitle entries with timing and text
    """
    if not os.path.exists(vtt_path):
        return []
        
    subtitles = []
    current_subtitle = None
    
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            start_idx = i + 1
            break
    
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        
        # Empty line marks the end of a subtitle entry
        if not line:
            i += 1
            continue
            
        # Check if line contains timestamps
        timestamp_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", line)
        if timestamp_match:
            start_time, end_time = timestamp_match.groups()
            
            # Create new subtitle entry
            current_subtitle = {
                "start": start_time,
                "end": end_time,
                "text": []
            }
            subtitles.append(current_subtitle)
            
            # Move to next line to get the text
            i += 1
            
            # Collect all text lines until empty line
            while i < len(lines) and lines[i].strip():
                text_line = lines[i].strip()
                # Skip lines with timestamps or cue identifiers
                if not re.match(r"^\d+$", text_line) and not re.match(r"^\d{2}:\d{2}:", text_line):
                    current_subtitle["text"].append(text_line)
                i += 1
        else:
            i += 1
    
    # Create final format with joined text
    for subtitle in subtitles:
        subtitle["text"] = " ".join(subtitle["text"])
    
    return subtitles


def parse_srt_subtitles(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse SRT subtitle file into a list of subtitle entries.
    
    Args:
        srt_path: Path to SRT subtitle file
        
    Returns:
        List of subtitle entries with timing and text
    """
    if not os.path.exists(srt_path):
        return []
        
    subtitles = []
    current_subtitle = None
    subtitle_text = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Empty line marks the end of a subtitle entry
        if not line:
            if current_subtitle and subtitle_text:
                current_subtitle["text"] = " ".join(subtitle_text)
                subtitles.append(current_subtitle)
                current_subtitle = None
                subtitle_text = []
            i += 1
            continue
            
        # Check if line is a subtitle number
        if re.match(r"^\d+$", line):
            i += 1
            if i < len(lines):
                # Next line should contain timestamps
                timestamp_line = lines[i].strip()
                timestamp_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
                if timestamp_match:
                    start_time, end_time = timestamp_match.groups()
                    # Convert comma to period for consistent format
                    start_time = start_time.replace(',', '.')
                    end_time = end_time.replace(',', '.')
                    
                    # Create new subtitle entry
                    current_subtitle = {
                        "start": start_time,
                        "end": end_time,
                        "text": ""
                    }
                    
                    # Move to next line to collect text
                    i += 1
                    while i < len(lines) and lines[i].strip():
                        subtitle_text.append(lines[i].strip())
                        i += 1
        else:
            i += 1
    
    # Add the last subtitle if present
    if current_subtitle and subtitle_text:
        current_subtitle["text"] = " ".join(subtitle_text)
        subtitles.append(current_subtitle)
    
    return subtitles


def parse_subtitles(subtitle_path: str) -> List[Dict[str, Any]]:
    """
    Parse subtitle file based on its extension.
    
    Args:
        subtitle_path: Path to subtitle file
        
    Returns:
        List of subtitle entries
    """
    _, ext = os.path.splitext(subtitle_path)
    ext = ext.lower()[1:]  # Remove the dot
    
    if ext == 'vtt':
        return parse_vtt_subtitles(subtitle_path)
    elif ext == 'srt':
        return parse_srt_subtitles(subtitle_path)
    else:
        print(f"Unsupported subtitle format: {ext}")
        return []


def translate_subtitles(subtitles: List[Dict[str, Any]], source_lang: str, model: str, batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Translate subtitle texts using the translator agent.
    
    Args:
        subtitles: List of subtitle entries
        source_lang: Source language code
        model: Gemini model to use for translation
        batch_size: Number of subtitle entries to translate per API call (0 for all at once)
        
    Returns:
        List of subtitle entries with translated text
    """
    if not subtitles:
        return []
    
    print(f"Translating {len(subtitles)} subtitle entries from {source_lang} to English...")
    
    # Initialize translator with specified model
    translator = TranslatorAgent(model=model)
    
    # Option to translate all subtitles at once
    if batch_size <= 0:
        print("Translating all subtitles in a single API call...")
        try:
            # Join all subtitle texts with separators
            all_text = "\n---SUBTITLE_SEPARATOR---\n".join([sub["text"] for sub in subtitles])
            
            translated_text = translator.translate(
                text=all_text,
                source_language=source_lang,
                target_language="English",
                preserve_formatting=True
            )
            
            # Split the translated text back into individual entries
            translations = translated_text.split("\n---SUBTITLE_SEPARATOR---\n")
            
            # Ensure we got expected number of translations
            if len(translations) != len(subtitles):
                print(f"Warning: Got {len(translations)} translations for {len(subtitles)} subtitles.")
                # Try to match them up as best we can
                translations = translations[:len(subtitles)]
                # Pad with empty strings if needed
                translations.extend([""] * (len(subtitles) - len(translations)))
            
            # Create new subtitle entries with translations
            translated_subtitles = []
            for i, sub in enumerate(subtitles):
                new_sub = sub.copy()
                new_sub["text"] = translations[i].strip() if i < len(translations) else sub["text"]
                translated_subtitles.append(new_sub)
            
            print("Translation completed successfully.")
            return translated_subtitles
            
        except Exception as e:
            print(f"Error translating all subtitles at once: {e}")
            print("Falling back to batch translation...")
            # Fall back to batch processing if translating all at once fails
    
    # Group subtitles for more efficient translation (translate in batches)
    batched_results = []
    effective_batch_size = batch_size if batch_size > 0 else 5  # Default to 5 if batch_size is invalid
    
    print(f"Translating in batches of {effective_batch_size} entries...")
    for i in range(0, len(subtitles), effective_batch_size):
        batch = subtitles[i:i+effective_batch_size]
        batch_text = "\n---\n".join([sub["text"] for sub in batch])
        
        try:
            translated_text = translator.translate(
                text=batch_text,
                source_language=source_lang,
                target_language="English",
                preserve_formatting=True
            )
            
            # Split the translated text back into individual entries
            translations = translated_text.split("\n---\n")
            
            # Ensure we got expected number of translations
            if len(translations) != len(batch):
                # If not, match them up as best we can
                translations = translations[:len(batch)]
                # Pad with empty strings if needed
                translations.extend([""] * (len(batch) - len(translations)))
            
            # Create new subtitle entries with translations
            for j, sub in enumerate(batch):
                new_sub = sub.copy()
                new_sub["text"] = translations[j].strip()
                batched_results.append(new_sub)
            
            # Add progress indicator
            print(f"Translated entries {i+1}-{min(i+effective_batch_size, len(subtitles))} of {len(subtitles)}")
            
        except Exception as e:
            print(f"Error translating batch: {e}")
            # Return original entries for this batch if translation fails
            batched_results.extend(batch)
    
    return batched_results


def write_srt_file(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write subtitle entries to an SRT file.
    
    Args:
        subtitles: List of subtitle entries
        output_path: Path to write the SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            # Convert period to comma for SRT format
            start_time = sub["start"].replace('.', ',')
            end_time = sub["end"].replace('.', ',')
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{sub['text']}\n\n")


def open_in_mpv(video_path: str, subtitle_path: Optional[str] = None) -> None:
    """
    Open the video in MPV with subtitles.
    
    Args:
        video_path: Path to the video file
        subtitle_path: Path to the subtitle file
    """
    cmd = ["mpv", video_path]
    
    if subtitle_path:
        # MPV expects --sub-file=path format (with equals sign)
        cmd.append(f"--sub-file={subtitle_path}")
    
    print(f"Opening video in MPV: {os.path.basename(video_path)}")
    if subtitle_path:
        print(f"With subtitles: {os.path.basename(subtitle_path)}")
    
    try:
        # Use Popen to avoid blocking and return immediately
        subprocess.Popen(cmd)
    except Exception as e:
        print(f"Error opening MPV: {e}")


def main(args: argparse.Namespace):
    
    # Check for API key early if translation is requested
    if not args.no_translate and not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print(f"Please add your Google AI API key to {ENV_PATH}")
        print("You can continue with --no-translate to download without translation")
        if input("Continue with download only? (y/n): ").lower() != 'y':
            sys.exit(1)
        args.no_translate = True
    
    # Extract video ID for display
    video_id = get_video_id(args.url)
    if video_id:
        print(f"Processing YouTube video ID: {video_id}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Convert language name to language code if provided
    if args.source_lang != "auto" and args.source_lang != "all":
        original_lang = args.source_lang
        args.source_lang = get_language_code(args.source_lang)
        if args.source_lang != original_lang:
            print(f"Converted language name '{original_lang}' to code '{args.source_lang}'")
    
    # Download video and subtitles
    video_path, subtitle_paths = download_video(
        args.url, 
        args.output_dir, 
        args.sub_lang,
        force_download=args.force_download
    )
    
    if not video_path:
        print("Failed to download video")
        sys.exit(1)
    
    print(f"Video downloaded to: {video_path}")
    
    if not subtitle_paths:
        print("No subtitles found. The video might not have any subtitles.")
        
        if args.play:
            open_in_mpv(video_path)
        
        sys.exit(0)
    
    print(f"Found {len(subtitle_paths)} subtitle file(s)")
    
    # Try to find a subtitle file in the source language first, if specified
    selected_subtitle = None
    if args.source_lang != "auto":
        for path in subtitle_paths:
            # Look for language code in filename
            if f".{args.source_lang}." in os.path.basename(path).lower():
                selected_subtitle = path
                print(f"Found subtitle file matching requested language '{args.source_lang}': {os.path.basename(path)}")
                break
    
    # Interactive subtitle selection if multiple subtitles are available and no specific match was found
    if not selected_subtitle and len(subtitle_paths) > 1 and args.source_lang == "auto":
        print("\nAvailable subtitle files:")
        for i, path in enumerate(subtitle_paths, 1):
            # Extract language code from filename if possible
            filename = os.path.basename(path)
            lang_match = re.search(r'\.([a-z]{2,3})\.(?:vtt|srt)$', filename)
            lang_code = lang_match.group(1) if lang_match else "unknown"
            print(f"{i}. {filename} (Language code: {lang_code})")
        
        # Prompt user to select a subtitle file
        selection = input("\nSelect subtitle file number to use (default: 1): ").strip()
        
        try:
            index = int(selection) - 1 if selection else 0
            if 0 <= index < len(subtitle_paths):
                selected_subtitle = subtitle_paths[index]
                # Try to extract language from filename for automatic source language
                lang_match = re.search(r'\.([a-z]{2,3})\.(?:vtt|srt)$', os.path.basename(selected_subtitle))
                if lang_match and args.source_lang == "auto":
                    args.source_lang = lang_match.group(1)
                    print(f"Detected subtitle language code from filename: {args.source_lang}")
            else:
                print("Invalid selection, using the first subtitle file")
                selected_subtitle = subtitle_paths[0]
        except ValueError:
            print("Invalid input, using the first subtitle file")
            selected_subtitle = subtitle_paths[0]
    elif not selected_subtitle:
        # Use the first subtitle file if only one is available or no interactive selection
        selected_subtitle = subtitle_paths[0]
        
    subtitle_path = selected_subtitle
    print(f"Using subtitle file: {subtitle_path}")
    
    # Parse subtitles
    subtitles = parse_subtitles(subtitle_path)
    
    if not subtitles:
        print("Failed to parse subtitles")
        
        if args.play:
            open_in_mpv(video_path)
        
        sys.exit(0)
    
    print(f"Parsed {len(subtitles)} subtitle entries")
    
    # Skip translation if requested
    if args.no_translate:
        print("Skipping translation as requested")
        
        if args.play:
            open_in_mpv(video_path, subtitle_path)
        
        sys.exit(0)
    
    # Translate subtitles
    source_lang = args.source_lang
    if source_lang == "auto":
        # Use the first subtitle to detect language
        sample_text = subtitles[0]["text"]
        print("Auto-detecting subtitle language...")
        try:
            translator = TranslatorAgent(model=args.model)
            detected_lang = translator.detect_language(sample_text)
            source_lang = detected_lang
            print(f"Detected subtitle language: {source_lang}")
        except Exception as e:
            print(f"Error detecting language: {e}")
            print("Falling back to 'auto' as source language")
    
    translated_subtitles = translate_subtitles(
        subtitles, 
        source_lang, 
        args.model, 
        batch_size=args.batch_size
    )
    
    if not translated_subtitles:
        print("Translation failed")
        
        if args.play:
            open_in_mpv(video_path, subtitle_path)
        
        sys.exit(0)
    
    # Write translated subtitles to SRT file
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    translated_subtitle_path = os.path.join(video_dir, f"{video_name}_translated_en.srt")
    
    write_srt_file(translated_subtitles, translated_subtitle_path)
    print(f"Translated subtitles written to: {translated_subtitle_path}")
    
    # Open in MPV
    if args.play:
        open_in_mpv(video_path, translated_subtitle_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube video and translate subtitles")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--source-lang", default="auto", 
                       help="Source language code or name (default: auto-detect, will prompt if multiple subtitle files found)")
    parser.add_argument("--sub-lang", default=DEFAULT_SUBTITLE_LANGS, 
                       help=f"Subtitle language(s) to download (default: {DEFAULT_SUBTITLE_LANGS}, use 'all' for all available)")
    parser.add_argument("--output-dir", default=DOWNLOAD_DIR, 
                       help=f"Output directory (default: {os.path.relpath(DOWNLOAD_DIR)})")
    parser.add_argument("--play", action="store_true", 
                       help="Open video in MPV after processing (default: do not play)")
    parser.add_argument("--model", default=GEMINI_MODEL, 
                       help=f"Gemini model to use for translation (default: {GEMINI_MODEL})")
    parser.add_argument("--no-translate", action="store_true", 
                       help="Skip translation step (download only)")
    parser.add_argument("--force-download", action="store_true", 
                       help="Force download even if video already exists")
    parser.add_argument("--batch-size", type=int, default=0, 
                       help="Number of subtitle entries to translate per API call (default: 0 for all)")
    
    main(parser.parse_args()) 