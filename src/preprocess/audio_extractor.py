from __future__ import annotations

import subprocess
from pathlib import Path


def extract_wav(
    video_path: str | Path,
    wav_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    src = Path(video_path)
    dst = Path(wav_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    command = [
    "ffmpeg",
    "-y",
    "-i", str(src),
    "-vn",              # This MUST be present to skip video
    "-acodec", codec,
    "-ar", str(sample_rate),
    "-ac", str(channels),
    str(dst),
    ]
    try: 
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error Output:", e.stderr) # This reveals the real issue
        raise
    return dst
