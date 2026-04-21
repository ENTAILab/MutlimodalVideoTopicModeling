from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def extract_segment_frames(
    video_path: str | Path,
    segments: list[dict[str, Any]],
    out_dir: str | Path,
    top_k: int = 3,
    image_format: str = "jpg",
    max_width: int = 640,
) -> dict[int, list[str]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    segment_to_frames: dict[int, list[str]] = {}
    for seg in segments:
        seg_id = int(seg["segment_id"])
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        times = [start_s + (i + 1) * (end_s - start_s) / (top_k + 1) for i in range(top_k)]

        paths: list[str] = []
        for j, t in enumerate(times):
            frame_idx = int(max(0, t) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if w > max_width:
                ratio = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)

            frame_name = f"seg_{seg_id:05d}_{j}.{image_format}"
            frame_path = out / frame_name
            cv2.imwrite(str(frame_path), frame)
            paths.append(str(frame_path))

        segment_to_frames[seg_id] = paths

    cap.release()
    return segment_to_frames
