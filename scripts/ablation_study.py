#!/usr/bin/env python3
"""Ablation study runner: process videos within a date range.

Example:
    python3 scripts/ablation_study.py \
        --video-dir /storage/projects/verma/tagesschau \
        --from-date 2026-01-01 \
        --to-date 2026-01-15 \
        --stages clip,visual_cluster,fusion,topic,merge,summary,metrics \
        --config configs/default.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
import shutil

ROOT = Path(__file__).parent.parent
DATA_OUTPUT = ROOT / "data" / "output"


def parse_date(date_str: str) -> datetime:
    """Parse date string YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def extract_date_from_dirname(dirname: str) -> datetime | None:
    """Extract date from video directory name like '2026-01-01_video_0_SD_540p'."""
    parts = dirname.split("_")
    if len(parts) >= 1:
        try:
            # Try to parse the first part as YYYY-MM-DD
            return datetime.strptime(parts[0], "%Y-%m-%d")
        except ValueError:
            pass
    return None


def discover_videos_in_date_range(
    video_dir: Path, from_date: datetime, to_date: datetime
) -> list[Path]:
    """Find all video files in the date range."""
    if not video_dir.exists():
        raise RuntimeError(f"Video directory not found: {video_dir}")

    videos = []
    for video_path in sorted(video_dir.glob("**/*.mp4")):
        # Try to extract date from parent directory name
        parent_name = video_path.parent.name
        video_date = extract_date_from_dirname(parent_name)

        if video_date and from_date <= video_date <= to_date:
            videos.append(video_path)

    return videos


def build_ablation_metadata(
    videos: list[Path],
    from_date: datetime,
    to_date: datetime,
    stages: str,
    config_path: str,
    clip_backend: str | None = None,
    args: argparse.Namespace | None = None,
) -> dict:
    """Build metadata dict for the ablation study."""
    return {
        "ablation_type": "date_range",
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "num_videos": len(videos),
        "stages": stages,
        "config": config_path,
        "clip_backend": clip_backend,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline_on_videos(
    videos: list[Path],
    stages: str,
    config_path: str,
    skip_existing: bool = False,
    clip_backend: str | None = None,
    signlip_model_name: str | None = None,
    signlip_pretrained: str | None = None,
    args=None,
) -> int:
    """Run the pipeline on the selected videos."""
    if not videos:
        print("No videos found in date range.")
        return 0

    print(f"Found {len(videos)} videos in date range:")
    for v in videos[:5]:
        print(f"  - {v.parent.name}/{v.name}")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")

    effective_config = config_path
    temp_config_path: Path | None = None
    if clip_backend or signlip_model_name or signlip_pretrained:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        clip_cfg = cfg.setdefault("clip", {})
        if clip_backend:
            clip_cfg["backend"] = clip_backend
        if signlip_model_name:
            clip_cfg["signlip_model_name"] = signlip_model_name
        if signlip_pretrained:
            clip_cfg["signlip_pretrained"] = signlip_pretrained

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            temp_config_path = Path(tmp.name)
            effective_config = str(temp_config_path)
        print(f"Using temporary config override: {effective_config}")

    # Instead of copying per-video artifacts, write an allowed-videos file and run pipeline once.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmpf:
        for v in videos:
            tmpf.write(str(v.resolve()) + "\n")
        allowed_path = Path(tmpf.name)

    # Prepare effective config (possibly with clip overrides)
    effective_config = config_path
    temp_config_path: Path | None = None
    if clip_backend or signlip_model_name or signlip_pretrained:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        clip_cfg = cfg.setdefault("clip", {})
        if clip_backend:
            clip_cfg["backend"] = clip_backend
        if signlip_model_name:
            clip_cfg["signlip_model_name"] = signlip_model_name
        if signlip_pretrained:
            clip_cfg["signlip_pretrained"] = signlip_pretrained

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            temp_config_path = Path(tmp.name)
            effective_config = str(temp_config_path)
        print(f"Using temporary config override: {effective_config}")

    cmd = [
        sys.executable,
        "-m",
        "src.pipeline_run",
        "--all-mp4",
        "--video-dir",
        str(Path(args.video_dir).resolve()),
        "--stages",
        stages,
        "--config",
        effective_config,
        "--allowed-videos-file",
        str(allowed_path),
    ]

    if skip_existing:
        cmd.append("--skip-existing")
        cmd.append("true")

    print(f"\nRunning pipeline on {len(videos)} selected videos using allowed-videos-file {allowed_path}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    returncode = result.returncode

    try:
        allowed_path.unlink(missing_ok=True)
    except Exception:
        pass
    if temp_config_path and temp_config_path.exists():
        try:
            temp_config_path.unlink(missing_ok=True)
        except Exception:
            pass
    # finally:
    #     if temp_config_path and temp_config_path.exists():
    #         temp_config_path.unlink(missing_ok=True)

    return returncode


def main():
    parser = argparse.ArgumentParser(description="Ablation study: process videos in date range")
    parser.add_argument(
        "--video-dir",
        default="/storage/projects/verma/tagesschau",
        help="Directory containing videos",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--stages",
        default="clip,visual_cluster,fusion,topic,merge,summary,metrics",
        help="Comma-separated pipeline stages",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to pipeline config file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos already processed",
    )
    parser.add_argument(
        "--output-meta",
        default=None,
        help="Path to save ablation metadata (optional)",
    )
    parser.add_argument(
        "--clip-backend",
        default=None,
        choices=["clip", "vllm_api", "signlip"],
        help="Override clip.backend for this ablation run",
    )
    parser.add_argument(
        "--signlip-model-name",
        default=None,
        help="Override clip.signlip_model_name for this run",
    )
    parser.add_argument(
        "--signlip-pretrained",
        default=None,
        help="Override clip.signlip_pretrained for this run",
    )
    args = parser.parse_args()

    # Parse dates
    from_date = parse_date(args.from_date)
    to_date = parse_date(args.to_date)

    if from_date > to_date:
        print("Error: from-date must be <= to-date")
        return 1

    # Discover videos
    video_dir = Path(args.video_dir)
    videos = discover_videos_in_date_range(video_dir, from_date, to_date)

    if not videos:
        print(f"No videos found between {args.from_date} and {args.to_date}")
        return 1

    # Save ablation metadata if requested
    if args.output_meta:
        meta = build_ablation_metadata(
            videos,
            from_date,
            to_date,
            args.stages,
            args.config,
            clip_backend=args.clip_backend,
            args=args,
        )
        meta_path = Path(args.output_meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved ablation metadata to {meta_path}\n")

    # Run pipeline
    returncode = run_pipeline_on_videos(
        videos,
        stages=args.stages,
        config_path=args.config,
        skip_existing=args.skip_existing,
        clip_backend=args.clip_backend,
        signlip_model_name=args.signlip_model_name,
        signlip_pretrained=args.signlip_pretrained,
        args=args,
    )

    return returncode


if __name__ == "__main__":
    sys.exit(main())
