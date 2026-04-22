from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _sample_times(start_s: float, end_s: float, sample_count: int) -> list[float]:
    if sample_count <= 0 or end_s <= start_s:
        return []
    span = end_s - start_s
    return [start_s + (i + 1) * span / (sample_count + 1) for i in range(sample_count)]


def _safe_l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 1e-8:
        return vec
    return vec / denom


def _frame_feature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    texture = _safe_l2_normalize(small)

    color_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).astype(np.float32)
    color_hist = _safe_l2_normalize(color_hist.reshape(-1))
    return np.concatenate([texture, color_hist], axis=0)


def _frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Laplacian variance is a strong and cheap proxy for blur.
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rank_representative_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    start_s: float,
    end_s: float,
    dedup_similarity_threshold: float,
    diversity_lambda: float,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not candidates:
        return []

    times = np.asarray([float(c["time_s"]) for c in candidates], dtype=np.float32)
    sharpness = np.asarray([float(c["sharpness"]) for c in candidates], dtype=np.float32)

    if float(sharpness.max()) > float(sharpness.min()):
        sharp_norm = (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min())
    else:
        sharp_norm = np.zeros_like(sharpness)

    mid = 0.5 * (start_s + end_s)
    half_span = max(1e-6, 0.5 * (end_s - start_s))
    center_relevance = 1.0 - np.clip(np.abs(times - mid) / half_span, 0.0, 1.0)
    relevance = 0.7 * center_relevance + 0.3 * sharp_norm

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx: int | None = None
        best_score = -1e9

        for idx in remaining:
            duplicate = False
            max_similarity = 0.0

            for picked in selected:
                similarity = _cosine_similarity(candidates[idx]["feature"], candidates[picked]["feature"])
                max_similarity = max(max_similarity, similarity)
                if similarity >= dedup_similarity_threshold:
                    duplicate = True
                    break

            if duplicate:
                continue

            score = float((1.0 - diversity_lambda) * relevance[idx] - diversity_lambda * max_similarity)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining = [idx for idx in remaining if idx != best_idx]

    if len(selected) < top_k:
        leftovers = [idx for idx in range(len(candidates)) if idx not in selected]
        leftovers.sort(key=lambda idx: float(relevance[idx]), reverse=True)
        selected.extend(leftovers[: max(0, top_k - len(selected))])

    return [candidates[idx] for idx in selected[:top_k]]


def extract_segment_frames(
    video_path: str | Path,
    segments: list[dict[str, Any]],
    out_dir: str | Path,
    top_k: int = 3,
    candidate_multiplier: int = 4,
    min_candidate_frames: int = 8,
    dedup_similarity_threshold: float = 0.96,
    diversity_lambda: float = 0.35,
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
        sample_count = max(top_k, top_k * candidate_multiplier, min_candidate_frames)
        times = _sample_times(start_s, end_s, sample_count)

        candidates: list[dict[str, Any]] = []
        for t in times:
            frame_idx = int(max(0, t) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if w > max_width:
                ratio = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)

            candidates.append(
                {
                    "time_s": t,
                    "frame": frame,
                    "feature": _frame_feature(frame),
                    "sharpness": _frame_sharpness(frame),
                }
            )

        ranked = _rank_representative_candidates(
            candidates=candidates,
            top_k=top_k,
            start_s=start_s,
            end_s=end_s,
            dedup_similarity_threshold=dedup_similarity_threshold,
            diversity_lambda=diversity_lambda,
        )

        paths: list[str] = []
        for j, item in enumerate(ranked):
            frame_name = f"seg_{seg_id:05d}_{j}.{image_format}"
            frame_path = out / frame_name
            cv2.imwrite(str(frame_path), item["frame"])
            paths.append(str(frame_path))

        segment_to_frames[seg_id] = paths

    cap.release()
    return segment_to_frames
