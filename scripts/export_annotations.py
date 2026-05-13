#!/usr/bin/env python3
"""
Export annotations from the SQLite database and compute inter-annotator agreement and coherence metrics.

Usage:
    python export_annotations.py --output <path> [--db <path>]

Output includes:
    - Per-annotation-pair agreement metrics (agreement rate, score correlation, etc.)
    - Per-topic aggregated coherence and agreement
    - Overall metrics
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, pearsonr


def get_annotations_by_point(db_path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Fetch all completed annotations grouped by (point_id, task_type).
    
    Returns:
        Dictionary mapping (point_id, task_type) to list of annotation dicts.
        Each annotation dict contains point_id, task_type, assignment_index, topic, topic_name, 
        video, response, completed_by, completed_at.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = """
    SELECT 
        t.point_id, t.task_type, t.assignment_index,
        json_extract(t.payload_json, '$.topic') as topic,
        json_extract(t.payload_json, '$.topic_name') as topic_name,
        json_extract(t.payload_json, '$.video') as video,
        t.response_json, u.username as completed_by, t.completed_at
    FROM tasks t
    LEFT JOIN users u ON u.id = t.completed_by_user_id
    WHERE t.status = 'completed'
    ORDER BY t.point_id, t.task_type, t.assignment_index
    """
    
    rows = conn.execute(query).fetchall()
    conn.close()
    
    result = defaultdict(list)
    for row in rows:
        key = (row["point_id"], row["task_type"])
        result[key].append({
            "point_id": row["point_id"],
            "task_type": row["task_type"],
            "assignment_index": row["assignment_index"],
            "topic": int(row["topic"]) if row["topic"] else None,
            "topic_name": row["topic_name"],
            "video": row["video"],
            "response": json.loads(row["response_json"]) if row["response_json"] else None,
            "completed_by": row["completed_by"],
            "completed_at": row["completed_at"],
        })
    
    return result


def compute_image_intrusion_agreement(pair: list[dict]) -> dict[str, Any]:
    """
    Compute agreement for image intrusion task (selected_image_index).
    
    Both annotators selected the same intruder image → agreement = True.
    """
    if len(pair) != 2:
        return {"agreement": None, "disagreement_reason": "incomplete_pair"}
    
    resp1 = pair[0].get("response", {})
    resp2 = pair[1].get("response", {})
    
    idx1 = resp1.get("selected_image_index")
    idx2 = resp2.get("selected_image_index")
    
    if idx1 is None or idx2 is None:
        return {"agreement": None, "disagreement_reason": "missing_response"}
    
    return {
        "agreement": idx1 == idx2,
        "annotator_1_index": idx1,
        "annotator_2_index": idx2,
        "annotator_1": pair[0]["completed_by"],
        "annotator_2": pair[1]["completed_by"],
    }


def compute_topic_matching_agreement(pair: list[dict]) -> dict[str, Any]:
    """
    Compute agreement for topic matching task (match_score 1-5).
    
    Returns correlation, ICC, and agreement (same score).
    """
    if len(pair) != 2:
        return {"agreement": None, "disagreement_reason": "incomplete_pair"}
    
    resp1 = pair[0].get("response", {})
    resp2 = pair[1].get("response", {})
    
    score1 = resp1.get("match_score")
    score2 = resp2.get("match_score")
    
    if score1 is None or score2 is None:
        return {"agreement": None, "disagreement_reason": "missing_response"}
    
    # Exact agreement (same rating)
    exact_agreement = score1 == score2
    
    # Absolute difference (0-4 scale)
    abs_diff = abs(score1 - score2)
    
    # Compute ICC(2,1) - two-way mixed, consistency, single rater
    scores = np.array([[score1, score2]])
    mean_score = scores.mean()
    mean_per_rater = scores.mean(axis=0)
    
    # Between-targets variance
    bms = 1 * np.sum((mean_score - mean_per_rater) ** 2)
    
    # Within-targets variance
    wms = np.sum((scores - mean_score) ** 2)
    
    # ICC formula (simplified for 2 raters)
    if wms == 0:
        icc = 1.0 if bms > 0 else None
    else:
        icc = (bms - wms) / (bms + wms)
    
    # Correlation (Pearson and Spearman)
    try:
        pearson_corr, _ = pearsonr([score1], [score2])
    except:
        pearson_corr = None
    
    try:
        spearman_corr, _ = spearmanr([score1], [score2])
    except:
        spearman_corr = None
    
    return {
        "agreement": exact_agreement,
        "score_1": score1,
        "score_2": score2,
        "absolute_difference": abs_diff,
        "icc": icc if icc is not None else None,
        "pearson_correlation": pearson_corr if pearson_corr is not None else None,
        "annotator_1": pair[0]["completed_by"],
        "annotator_2": pair[1]["completed_by"],
        "coherence_score": (score1 + score2) / 2,
    }


def export_annotations(db_path: Path, output_path: Path) -> None:
    """
    Export annotations and compute metrics.
    """
    annotations_by_point = get_annotations_by_point(db_path)
    
    image_intrusion_pairs = []
    topic_matching_pairs = []
    topic_scores = defaultdict(lambda: {"coherence_scores": [], "agreement_count": 0, "disagreement_count": 0})
    
    for (point_id, task_type), pair in annotations_by_point.items():
        if task_type == "image_intrusion":
            agreement_data = compute_image_intrusion_agreement(pair)
            agreement_data["point_id"] = point_id
            agreement_data["topic"] = pair[0]["topic"] if pair else None
            agreement_data["topic_name"] = pair[0]["topic_name"] if pair else None
            agreement_data["video"] = pair[0]["video"] if pair else None
            image_intrusion_pairs.append(agreement_data)
            
            # Update topic-level stats
            if pair and pair[0]["topic"] is not None:
                topic = pair[0]["topic"]
                if agreement_data.get("agreement") is True:
                    topic_scores[topic]["agreement_count"] += 1
                elif agreement_data.get("agreement") is False:
                    topic_scores[topic]["disagreement_count"] += 1
        
        elif task_type == "topic_matching":
            agreement_data = compute_topic_matching_agreement(pair)
            agreement_data["point_id"] = point_id
            agreement_data["topic"] = pair[0]["topic"] if pair else None
            agreement_data["topic_name"] = pair[0]["topic_name"] if pair else None
            agreement_data["video"] = pair[0]["video"] if pair else None
            topic_matching_pairs.append(agreement_data)
            
            # Update topic-level stats
            if pair and pair[0]["topic"] is not None:
                topic = pair[0]["topic"]
                if "coherence_score" in agreement_data:
                    topic_scores[topic]["coherence_scores"].append(agreement_data["coherence_score"])
                if agreement_data.get("agreement") is True:
                    topic_scores[topic]["agreement_count"] += 1
                elif agreement_data.get("agreement") is False:
                    topic_scores[topic]["disagreement_count"] += 1
    
    # Compute topic-level summaries
    topic_summaries = {}
    for topic, stats in sorted(topic_scores.items()):
        total = stats["agreement_count"] + stats["disagreement_count"]
        topic_summaries[topic] = {
            "agreement_rate": stats["agreement_count"] / total if total > 0 else None,
            "disagreement_rate": stats["disagreement_count"] / total if total > 0 else None,
            "coherence": np.mean(stats["coherence_scores"]) if stats["coherence_scores"] else None,
            "coherence_std": np.std(stats["coherence_scores"]) if len(stats["coherence_scores"]) > 1 else None,
        }
    
    # Compute overall metrics
    intrusion_agreements = [p.get("agreement") for p in image_intrusion_pairs if p.get("agreement") is not None]
    matching_agreements = [p.get("agreement") for p in topic_matching_pairs if p.get("agreement") is not None]
    intrusion_agreement_rate = np.mean(intrusion_agreements) if intrusion_agreements else None
    matching_agreement_rate = np.mean(matching_agreements) if matching_agreements else None
    
    matching_coherences = [p.get("coherence_score") for p in topic_matching_pairs if "coherence_score" in p]
    overall_coherence = np.mean(matching_coherences) if matching_coherences else None
    
    overall_metrics = {
        "image_intrusion": {
            "total_pairs": len(image_intrusion_pairs),
            "agreement_rate": float(intrusion_agreement_rate) if intrusion_agreement_rate is not None else None,
            "agreement_count": int(sum(1 for p in image_intrusion_pairs if p.get("agreement") is True)),
            "disagreement_count": int(sum(1 for p in image_intrusion_pairs if p.get("agreement") is False)),
        },
        "topic_matching": {
            "total_pairs": len(topic_matching_pairs),
            "agreement_rate": float(matching_agreement_rate) if matching_agreement_rate is not None else None,
            "agreement_count": int(sum(1 for p in topic_matching_pairs if p.get("agreement") is True)),
            "disagreement_count": int(sum(1 for p in topic_matching_pairs if p.get("agreement") is False)),
            "coherence": float(overall_coherence) if overall_coherence is not None else None,
            "coherence_std": float(np.std(matching_coherences)) if len(matching_coherences) > 1 else None,
        },
        "overall": {
            "total_annotations": len(image_intrusion_pairs) + len(topic_matching_pairs),
            "completed_pairs": sum(1 for p in image_intrusion_pairs if p.get("agreement") is not None) + sum(1 for p in topic_matching_pairs if p.get("agreement") is not None),
        },
    }
    
    output = {
        "metadata": {
            "database": str(db_path),
            "output_file": str(output_path),
        },
        "overall_metrics": overall_metrics,
        "topic_summaries": topic_summaries,
        "image_intrusion_pairs": image_intrusion_pairs,
        "topic_matching_pairs": topic_matching_pairs,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported annotations to {output_path}")
    print(f"\nOverall Metrics:")
    print(f"  Image Intrusion Agreement: {overall_metrics['image_intrusion']['agreement_rate']:.2%}" if overall_metrics['image_intrusion']['agreement_rate'] is not None else "  Image Intrusion Agreement: N/A")
    print(f"  Topic Matching Agreement: {overall_metrics['topic_matching']['agreement_rate']:.2%}" if overall_metrics['topic_matching']['agreement_rate'] is not None else "  Topic Matching Agreement: N/A")
    print(f"  Topic Matching Coherence: {overall_metrics['topic_matching']['coherence']:.2f}" if overall_metrics['topic_matching']['coherence'] is not None else "  Topic Matching Coherence: N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotations and compute metrics")
    parser.add_argument("--db", type=Path, default=Path("data/annotation.sqlite3"))
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"Error: Database not found at {args.db}")
        return
    
    export_annotations(args.db, args.output)


if __name__ == "__main__":
    main()
