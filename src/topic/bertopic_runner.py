from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bertopic import BERTopic


def run_bertopic(
    segments: list[dict[str, Any]],
    sentence_model_name: str = "all-mpnet-base-v2",
    min_topic_size: int = 5,
    seed_topic_list: list[list[str]] | None = None,
    precomputed_embeddings: np.ndarray | None = None,
) -> tuple[BERTopic, list[int], list[list[float]]]:
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sentence_transformers import SentenceTransformer
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to import BERTopic dependencies. This is usually caused by an incompatible "
            "torchcodec/ffmpeg stack pulled in by sentence-transformers. "
            "Use sentence-transformers<5 or install a torchcodec version compatible with your "
            "current PyTorch and FFmpeg runtime."
        ) from exc

    docs = [str(seg.get("text", "")).strip() or "[EMPTY]" for seg in segments]
    doc_count = len(docs)
    min_df = 1 if doc_count < 10 else 1
    max_df = 1.0 if doc_count < 10 else 0.95
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    guided_seed_topic_list = seed_topic_list or None
    embedding_model = None
    if precomputed_embeddings is None:
        embedding_model = SentenceTransformer(sentence_model_name)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        seed_topic_list=guided_seed_topic_list,
        min_topic_size=min_topic_size,
        verbose=False,
    )
    if precomputed_embeddings is not None and precomputed_embeddings.shape[0] != len(docs):
        raise ValueError("precomputed_embeddings rows must match number of segments")

    topics, probs = topic_model.fit_transform(docs, embeddings=precomputed_embeddings)
    return topic_model, topics, probs


def encode_text_segments(
    segments: list[dict[str, Any]],
    sentence_model_name: str = "all-mpnet-base-v2",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    docs = [str(seg.get("text", "")).strip() or "[EMPTY]" for seg in segments]
    model = SentenceTransformer(sentence_model_name)
    embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)
