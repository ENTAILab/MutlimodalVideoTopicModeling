from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def summarize_topics_extractively(
    segments: list[dict[str, Any]],
    topics: list[int],
    max_sentences_per_topic: int = 4,
) -> dict[str, str]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for seg, topic in zip(segments, topics):
        grouped[int(topic)].append(seg["text"])

    summaries: dict[str, str] = {}
    for topic_id, sents in grouped.items():
        if topic_id == -1:
            continue
        if len(sents) <= max_sentences_per_topic:
            summaries[str(topic_id)] = " ".join(sents)
            continue

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.95)
        mat = vec.fit_transform(sents)
        centroid = mat.mean(axis=0)
        scores = np.asarray((mat @ centroid.T)).reshape(-1)
        idx = np.argsort(scores)[::-1][:max_sentences_per_topic]
        chosen = [sents[i] for i in idx]
        summaries[str(topic_id)] = " ".join(chosen)

    return summaries
