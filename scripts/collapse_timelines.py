#!/usr/bin/env python3
"""Collapse topic timelines across videos in data/output by grouping similar topics.

Outputs: data/output/collapsed_timeline.json
"""

import json
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict, Counter

ROOT = Path(__file__).parent.parent / "data" / "output"
OUT_PATH = ROOT / "collapsed_timeline.json"

def load_topic_files(root: Path):
    results = []
    for topic_file in sorted(root.glob("**/topic_info.json")):
        # assume the parent directory is the video folder
        video_dir = topic_file.parent
        video_name = video_dir.name
        try:
            data = json.loads(topic_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read {topic_file}: {e}")
            continue
        # data is expected to be list of topic dicts
        for topic in data:
            results.append({
                "video_dir": str(video_dir.relative_to(root)),
                "video_name": video_name,
                "topic": topic.get("Topic"),
                "name": topic.get("Name"),
                "count": int(topic.get("Count") or 0),
                "representation": topic.get("Representation") or [],
                "rep_doc": (topic.get("Representative_Docs") or [])[:1],
            })
    return results


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a, b):
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cluster_topics(items, name_thresh=0.75, jaccard_thresh=0.4):
    clusters = []
    for item in items:
        assigned = False
        for cluster in clusters:
            # compare to cluster representative name/words
            rep_name = cluster["rep_name"]
            rep_words = cluster["rep_words"]
            if item["name"] and rep_name and item["name"] == rep_name:
                match = True
            else:
                name_sim = name_similarity(item.get("name", ""), rep_name or "")
                word_sim = jaccard(item.get("representation", []), rep_words or [])
                match = (name_sim >= name_thresh) or (word_sim >= jaccard_thresh)
            if match:
                cluster["items"].append(item)
                # update representative words and name heuristically
                combined_words = Counter()
                for it in cluster["items"]:
                    combined_words.update([w for w in (it.get("representation") or [])])
                cluster["rep_words"] = [w for w, _ in combined_words.most_common(10)]
                # rep_name: choose most common name
                names = [it.get("name") for it in cluster["items"] if it.get("name")]
                if names:
                    cluster["rep_name"] = Counter(names).most_common(1)[0][0]
                assigned = True
                break
        if not assigned:
            clusters.append({
                "rep_name": item.get("name"),
                "rep_words": list(item.get("representation") or [])[:10],
                "items": [item],
            })
    return clusters


def build_collapsed_timeline(clusters):
    out = []
    for cid, c in enumerate(clusters, 1):
        total_count = sum(it.get("count", 0) for it in c["items"]) if c.get("items") else 0
        occurrences = []
        for it in c["items"]:
            occurrences.append({
                "video_dir": it.get("video_dir"),
                "video_name": it.get("video_name"),
                "topic": it.get("topic"),
                "name": it.get("name"),
                "count": it.get("count"),
                "top_words": (it.get("representation") or [])[:6],
                "rep_doc": it.get("rep_doc") or [],
            })
        out.append({
            "cluster_id": cid,
            "cluster_name": c.get("rep_name"),
            "rep_words": c.get("rep_words"),
            "total_count": total_count,
            "occurrences": occurrences,
            "num_occurrences": len(occurrences),
        })
    # sort clusters by total_count desc
    out.sort(key=lambda x: x["total_count"], reverse=True)
    return out


def main():
    items = load_topic_files(ROOT)
    if not items:
        print(f"No topic_info.json files found under {ROOT}")
        return
    print(f"Loaded {len(items)} topic entries from {ROOT}")
    clusters = cluster_topics(items)
    collapsed = build_collapsed_timeline(clusters)
    OUT_PATH.write_text(json.dumps(collapsed, ensure_ascii=False, indent=2))
    print(f"Wrote collapsed timeline to {OUT_PATH} with {len(collapsed)} clusters")


if __name__ == "__main__":
    main()
