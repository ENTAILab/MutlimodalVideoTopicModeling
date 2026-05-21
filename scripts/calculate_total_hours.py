#!/usr/bin/env python3
"""Calculate total hours of video files in the tagesschau directory and subdirectories."""

import json
import subprocess
from pathlib import Path

VIDEO_DIR = Path("/storage/projects/verma/tagesschau")

def get_video_duration(filepath: str) -> float:
    """Get duration of video in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse "duration=XXX.XXX" format
        output = result.stdout.strip()
        if output.startswith("duration="):
            return float(output.split("=")[1])
        return 0.0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0

def main():
    if not VIDEO_DIR.exists():
        print(f"Video directory not found: {VIDEO_DIR}")
        return

    video_files = sorted(VIDEO_DIR.glob("**/*.mp4")) + sorted(VIDEO_DIR.glob("**/*.mkv"))
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return

    total_seconds = 0
    print(f"Processing {len(video_files)} videos...\n")
    
    for i, video_file in enumerate(video_files, 1):
        duration_seconds = get_video_duration(str(video_file))
        total_seconds += duration_seconds
        hours = duration_seconds / 3600
        print(f"{i:3d}. {video_file.name:50s} {hours:8.2f}h ({duration_seconds:10.0f}s)")
    
    total_hours = total_seconds / 3600
    total_minutes = (total_seconds % 3600) / 60
    
    print("\n" + "=" * 80)
    print(f"Total: {total_hours:.2f} hours ({int(total_hours)}h {int(total_minutes)}m)")
    print("=" * 80)

if __name__ == "__main__":
    main()
