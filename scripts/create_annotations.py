#!/usr/bin/env python3
"""Create cross-video annotation artifacts for human validation tests.

This builds a single diversified annotation dataset across all videos instead of
one dataset per video. Topics are selected in a round-robin pass across videos,
limited by --top_n, and each annotation point uses spaced segment timestamps so
the images are not near-duplicates.

Outputs are written under data/output/annotation/cross_video by default.
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def link_or_copy(src: Path, dst: Path):
    safe_mkdir(dst.parent)
    try:
        if dst.exists():
            return
        os.symlink(src, dst)
    except Exception:
        shutil.copy(src, dst)


def select_spaced_items(items, k):
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]

    chosen_indexes = []
    for i in range(k):
        idx = round(i * (len(items) - 1) / (k - 1))
        if idx not in chosen_indexes:
            chosen_indexes.append(idx)

    if len(chosen_indexes) < k:
        for idx in range(len(items)):
            if idx not in chosen_indexes:
                chosen_indexes.append(idx)
            if len(chosen_indexes) >= k:
                break

    return [items[idx] for idx in chosen_indexes[:k]]


def pick_frame_from_segment(segment_frames, segment_id):
    frames = segment_frames.get(str(segment_id), [])
    if not frames:
        return None
    return frames[min(len(frames) // 2, len(frames) - 1)]


def normalize_topic_name(value):
    if not value:
        return ""
    return " ".join(str(value).strip().lower().split())


def normalized_word_set(words):
    normalized = set()
    for word in words or []:
        token = normalize_topic_name(word)
        if token:
            normalized.add(token)
    return normalized


def find_similar_topic_candidates(candidate, pool):
    candidate_name = normalize_topic_name(candidate.get("name"))
    candidate_words = normalized_word_set(candidate.get("representation", []))

    scored = []
    for other in pool:
        if other["video"] == candidate["video"]:
            continue

        other_name = normalize_topic_name(other.get("name"))
        other_words = normalized_word_set(other.get("representation", []))

        score = 0
        if candidate_name and other_name and candidate_name == other_name:
            score += 4

        overlap = len(candidate_words & other_words)
        if overlap >= 2:
            score += overlap

        if score > 0:
            scored.append((score, other))

    scored.sort(key=lambda item: (-item[0], -item[1].get("count", 0), item[1].get("video", "")))
    return [item[1] for item in scored]


def build_video_topics(output_root: Path, processed_root: Path, video_dir: str):
    out_dir = output_root / video_dir
    proc_dir = processed_root / video_dir
    # Try to load cleaned topics first, fall back to topic_info.json
    cleaned_topic_info_path = out_dir / "cleaned_topics_info.json"
    topic_info_path = out_dir / "topic_info.json"
    segments_enriched_path = out_dir / "segments_enriched.json"
    segment_frames_path = proc_dir / "segment_frames.json"

    if not segments_enriched_path.exists() or not segment_frames_path.exists():
        return []

    if cleaned_topic_info_path.exists():
        topic_info = load_json(cleaned_topic_info_path)
    elif topic_info_path.exists():
        topic_info = load_json(topic_info_path)
    else:
        return []
    segments_enriched = load_json(segments_enriched_path)
    segment_frames = load_json(segment_frames_path)

    topic_to_segments = defaultdict(list)
    segment_meta = {}
    for seg in segments_enriched:
        topic = seg.get("topic")
        segment_id = seg.get("segment_id")
        topic_to_segments[topic].append(segment_id)
        segment_meta[segment_id] = {
            "start_s": seg.get("start_s"),
            "end_s": seg.get("end_s"),
            "text": seg.get("text"),
        }

    candidates = []
    for topic in topic_info:
        topic_id = topic.get("Topic")
        if topic_id == -1:
            continue

        seg_ids = topic_to_segments.get(topic_id, [])
        if len(seg_ids) < 4:
            continue

        seg_ids = sorted(
            seg_ids,
            key=lambda sid: (
                segment_meta.get(sid, {}).get("start_s") is None,
                segment_meta.get(sid, {}).get("start_s", 0.0),
            ),
        )

        candidates.append(
            {
                "video": video_dir,
                "topic": topic_id,
                "count": topic.get("Count", 0),
                "name": topic.get("label") or topic.get("Name"),  # Prefer label from cleaned topics
                "representation": topic.get("Representation", []),
                "representative_docs": topic.get("Representative_Docs", []),
                "segments": seg_ids,
                "segment_meta": segment_meta,
                "segment_frames": segment_frames,
            }
        )

    return candidates


def round_robin_topics(candidates, top_n):
    by_video = defaultdict(list)
    for candidate in candidates:
        by_video[candidate["video"]].append(candidate)

    for video in by_video:
        by_video[video].sort(key=lambda item: (-item["count"], item["topic"]))

    # Ensure selected points are from different videos.
    # If top_n exceeds available videos, only one point per video is returned.
    ordered = []
    videos = sorted(by_video.keys())
    selected_topics = set()

    for video in videos:
        if len(ordered) >= top_n:
            break

        candidates_for_video = by_video[video]
        preferred = None

        for candidate in candidates_for_video:
            if candidate["topic"] not in selected_topics:
                preferred = candidate
                break

        if preferred is None and candidates_for_video:
            preferred = candidates_for_video[0]

        if preferred is not None:
            ordered.append(preferred)
            selected_topics.add(preferred["topic"])

    return ordered


def build_annotation_point(candidate, pool, intrusion_k, matching_k):
    selected_segments_intrusion = select_spaced_items(candidate["segments"], intrusion_k)
    selected_segments_matching = select_spaced_items(candidate["segments"], matching_k)
    if len(selected_segments_intrusion) < intrusion_k or len(selected_segments_matching) < matching_k:
        return None

    intrusion_images = []
    intrusion_segments = []
    intrusion_segments_meta = []
    for segment_id in selected_segments_intrusion:
        frame_path = pick_frame_from_segment(candidate["segment_frames"], segment_id)
        if frame_path is None:
            return None
        intrusion_images.append(frame_path)
        intrusion_segments.append(segment_id)
        meta = candidate.get("segment_meta", {}).get(str(segment_id)) or candidate.get("segment_meta", {}).get(segment_id)
        intrusion_segments_meta.append({
            "segment_id": segment_id,
            "start_s": meta.get("start_s") if meta else None,
            "end_s": meta.get("end_s") if meta else None,
            "text": meta.get("text") if meta else None,
            "frame": frame_path,
        })

    matching_images = []
    matching_segments = []
    matching_segments_meta = []

    # Prefer one matching image per different video when a similar cleaned
    # topic name exists across videos.
    similar_candidates = find_similar_topic_candidates(candidate, pool)
    matching_sources = [candidate] + similar_candidates
    matching_sources = matching_sources[:matching_k]

    for source in matching_sources:
        source_selected = select_spaced_items(source["segments"], 1)
        if not source_selected:
            continue
        segment_id = source_selected[0]
        frame_path = pick_frame_from_segment(source["segment_frames"], segment_id)
        if frame_path is None:
            continue
        matching_images.append(frame_path)
        matching_segments.append(segment_id)
        meta = source.get("segment_meta", {}).get(str(segment_id)) or source.get("segment_meta", {}).get(segment_id)
        matching_segments_meta.append({
            "video": source["video"],
            "segment_id": segment_id,
            "start_s": meta.get("start_s") if meta else None,
            "end_s": meta.get("end_s") if meta else None,
            "text": meta.get("text") if meta else None,
            "frame": frame_path,
        })

    # If there are not enough similar cross-video matches, complete with this
    # topic's own segments to keep the task usable.
    if len(matching_images) < matching_k:
        for segment_id in selected_segments_matching:
            if len(matching_images) >= matching_k:
                break
            if segment_id in matching_segments:
                continue
            frame_path = pick_frame_from_segment(candidate["segment_frames"], segment_id)
            if frame_path is None:
                continue
            matching_images.append(frame_path)
            matching_segments.append(segment_id)
            meta = candidate.get("segment_meta", {}).get(str(segment_id)) or candidate.get("segment_meta", {}).get(segment_id)
            matching_segments_meta.append({
                "video": candidate["video"],
                "segment_id": segment_id,
                "start_s": meta.get("start_s") if meta else None,
                "end_s": meta.get("end_s") if meta else None,
                "text": meta.get("text") if meta else None,
                "frame": frame_path,
            })

    if len(matching_images) < matching_k:
        return None

    return {
        "video": candidate["video"],
        "topic": candidate["topic"],
        "name": candidate["name"],
        "count": candidate["count"],
        "representation": candidate["representation"],
        "representative_docs": candidate.get("representative_docs", []),
        "intrusion_segments": intrusion_segments,
        "intrusion_segments_meta": intrusion_segments_meta,
        "intrusion_images": intrusion_images,
        "matching_segments": matching_segments,
        "matching_segments_meta": matching_segments_meta,
        "matching_images": matching_images,
        "matching_videos": [item.get("video") for item in matching_segments_meta],
    }


def choose_intruder(selected_point, pool):
    other_points = [
        point for point in pool
        if point["topic"] != selected_point["topic"] and point["video"] != selected_point["video"]
    ]
    if not other_points:
        other_points = [
            point for point in pool
            if point["topic"] != selected_point["topic"]
        ]
    if not other_points:
        return None

    intruder_source = random.choice(other_points)
    intruder_segment = random.choice(intruder_source["segments"])
    intruder_frame = pick_frame_from_segment(intruder_source["segment_frames"], intruder_segment)
    if intruder_frame is None:
        return None
    return {
        "video": intruder_source["video"],
        "topic": intruder_source["topic"],
        "segment": intruder_segment,
        "frame": intruder_frame,
    }


def build_dataset(output_root: Path, processed_root: Path, dest_root: Path, top_n: int, intrusion_k: int, matching_k: int, seed: int):
    random.seed(seed)
    safe_mkdir(dest_root)

    all_candidates = []
    for video_dir in sorted(p.name for p in output_root.iterdir() if p.is_dir()):
        all_candidates.extend(build_video_topics(output_root, processed_root, video_dir))

    selected_candidates = round_robin_topics(all_candidates, top_n)

    image_intrusion_dir = dest_root / "image_intrusion"
    topic_matching_dir = dest_root / "topic_matching"
    safe_mkdir(image_intrusion_dir)
    safe_mkdir(topic_matching_dir)

    points_manifest = []
    intrusion_manifest = []
    matching_manifest = []

    for idx, candidate in enumerate(selected_candidates, start=1):
        point_id = f"ann_{idx:04d}"
        point = build_annotation_point(candidate, all_candidates, intrusion_k, matching_k)
        if point is None:
            continue

        intruder = choose_intruder(point, all_candidates)
        if intruder is None:
            continue

        intrusion_set_dir = image_intrusion_dir / point_id
        matching_set_dir = topic_matching_dir / point_id
        safe_mkdir(intrusion_set_dir)
        safe_mkdir(matching_set_dir)

        intrusion_image_paths = []
        for image_index, src_path in enumerate(point["intrusion_images"]):
            dst_path = intrusion_set_dir / f"img_{image_index:02d}.jpg"
            link_or_copy(Path(src_path), dst_path)
            intrusion_image_paths.append(str(dst_path.relative_to(dest_root)))

        intruder_dst = intrusion_set_dir / f"img_{intrusion_k:02d}_intruder.jpg"
        link_or_copy(Path(intruder["frame"]), intruder_dst)
        intrusion_image_paths.append(str(intruder_dst.relative_to(dest_root)))

        matching_image_paths = []
        for image_index, src_path in enumerate(point["matching_images"]):
            dst_path = matching_set_dir / f"img_{image_index:02d}.jpg"
            link_or_copy(Path(src_path), dst_path)
            matching_image_paths.append(str(dst_path.relative_to(dest_root)))

        write_json(
            intrusion_set_dir / "metadata.json",
            {
                "id": point_id,
                "video": point["video"],
                "topic": point["topic"],
                "topic_name": point["name"],
                "words": point["representation"],
                "topic_representation": point["name"],
                "representative_docs": point.get("representative_docs", []),
                "selected_segments": point["intrusion_segments"],
                "selected_segments_meta": point.get("intrusion_segments_meta", []),
                "intruder": intruder,
                "images": intrusion_image_paths,
            },
        )
        write_json(
            matching_set_dir / "metadata.json",
            {
                "id": point_id,
                "video": point["video"],
                "topic": point["topic"],
                "topic_name": point["name"],
                "words": point["representation"],
                "topic_representation": point["name"],
                "representative_docs": point.get("representative_docs", []),
                "selected_segments": point["matching_segments"],
                "selected_segments_meta": point.get("matching_segments_meta", []),
                "matching_videos": point.get("matching_videos", []),
                "images": matching_image_paths,
            },
        )

        with open(intrusion_set_dir / "words.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(point["representation"]))
        with open(matching_set_dir / "words.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(point["representation"]))

        points_manifest.append(
            {
                "id": point_id,
                "video": point["video"],
                "topic": point["topic"],
                "topic_name": point["name"],
                "count": point["count"],
                "words": point["representation"],
                "topic_representation": point["name"],
                "representative_docs": point.get("representative_docs", []),
                "intrusion_segments": point["intrusion_segments"],
                "intrusion_segments_meta": point.get("intrusion_segments_meta", []),
                "matching_segments": point["matching_segments"],
                "matching_segments_meta": point.get("matching_segments_meta", []),
                "matching_videos": point.get("matching_videos", []),
            }
        )
        intrusion_manifest.append(
            {
                "id": point_id,
                "video": point["video"],
                "topic": point["topic"],
                "intruder_video": intruder["video"],
                "intruder_topic": intruder["topic"],
            }
        )
        matching_manifest.append(
            {
                "id": point_id,
                "video": point["video"],
                "topic": point["topic"],
            }
        )

    write_json(dest_root / "annotation_points.json", points_manifest)
    write_json(dest_root / "image_intrusion_manifest.json", intrusion_manifest)
    write_json(dest_root / "topic_matching_manifest.json", matching_manifest)
    write_json(
        dest_root / "annotation_summary.json",
        {
            "selected_points": len(points_manifest),
            "intrusion_sets": len(intrusion_manifest),
            "matching_sets": len(matching_manifest),
            "top_n_requested": top_n,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create cross-video annotation dataset with configurable task count"
    )
    parser.add_argument("--output_root", default="data/output", help="Root output folder")
    parser.add_argument("--processed_root", default="data/processed", help="Root processed folder")
    parser.add_argument("--dest", default="data/output/annotation/cross_video", help="Cross-video annotation destination root")
    parser.add_argument("--top_n", type=int, default=None, help="Number of annotation points to keep (overrides --num_tasks)")
    parser.add_argument("--num_tasks", type=int, default=150, help="Total number of annotation tasks (default: 100; expands to 400 with dual assignment)")
    parser.add_argument("--intrusion_k", type=int, default=7, help="Number of in-topic images for intrusion tests")
    parser.add_argument("--matching_k", type=int, default=6, help="Number of images for topic matching tests")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for intruder selection")
    args = parser.parse_args()

    # Calculate top_n from num_tasks if not explicitly provided
    # num_tasks = top_n × 2 (for image_intrusion + topic_matching)
    # So top_n = num_tasks / 2
    if args.top_n is None:
        top_n = args.num_tasks // 2
    else:
        top_n = args.top_n

    output_root = Path(args.output_root)
    processed_root = Path(args.processed_root)
    dest_root = Path(args.dest)

    print(f"Creating annotation dataset:")
    print(f"  Annotation points: {top_n}")
    print(f"  Task types per point: 2 (image_intrusion, topic_matching)")
    print(f"  Assignments per task: 2 (for dual annotation)")
    print(f"  Total tasks in database: {top_n * 2 * 2}")

    build_dataset(
        output_root=output_root,
        processed_root=processed_root,
        dest_root=dest_root,
        top_n=top_n,
        intrusion_k=args.intrusion_k,
        matching_k=args.matching_k,
        seed=args.seed,
    )
    print(f"Annotation creation complete. Summary saved to {dest_root / 'annotation_summary.json'}")


if __name__ == "__main__":
    main()
