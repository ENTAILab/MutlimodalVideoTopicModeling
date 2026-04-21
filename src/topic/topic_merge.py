from __future__ import annotations

from typing import Any


def reduce_similar_topics(topic_model: Any, docs: list[str]) -> Any:
    try:
        return topic_model.reduce_topics(docs, nr_topics="auto")
    except Exception:
        return topic_model
