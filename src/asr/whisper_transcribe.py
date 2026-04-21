from __future__ import annotations

from pathlib import Path
from typing import Any

import whisper


def transcribe_with_whisper(
    wav_path: str | Path,
    model_name: str = "large-v3",
    device: str = "cuda",
    language: str | None = None,
) -> list[dict[str, Any]]:
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(str(wav_path), language=language, verbose=False)

    segments: list[dict[str, Any]] = []
    for i, seg in enumerate(result.get("segments", [])):
        text = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if not text or end <= start:
            continue
        segments.append(
            {
                "segment_id": i,
                "start_s": start,
                "end_s": end,
                "text": text,
                "lang": result.get("language"),
            }
        )

    segments.sort(key=lambda item: item["start_s"])
    return segments
