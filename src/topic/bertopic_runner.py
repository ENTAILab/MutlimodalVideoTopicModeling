from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


import numpy as np
# from pyarrow import cuda

if TYPE_CHECKING:
    from bertopic import BERTopic


def run_bertopic(
    segments: list[dict[str, Any]],
    sentence_model_name: str = "all-mpnet-base-v2",
    min_topic_size: int = 5,
    seed_topic_list: list[list[str]] | None = None,
    precomputed_embeddings: np.ndarray | None = None,
    device: str = "cuda",
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

    # Ensure min_df / max_df are compatible for small corpora.
    # Use integer min_df=1 by default. Use a float max_df for larger corpora
    # but guard that max_df * doc_count >= min_df.
    min_df = 1
    preferred_max_df = 0.95 if doc_count >= 10 else 1.0
    if isinstance(preferred_max_df, float) and doc_count > 0 and preferred_max_df * doc_count < min_df:
        max_df = 1.0
    else:
        max_df = preferred_max_df

    # vectorizer_model = CountVectorizer(
    #     stop_words="english",
    #     ngram_range=(1, 2),
    #     min_df=min_df,
    #     max_df=max_df,
    # )
    from nltk.corpus import stopwords
    german_stopwords = stopwords.words('german')
    vectorizer_model = CountVectorizer(stop_words=german_stopwords)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    guided_seed_topic_list = seed_topic_list or None
    embedding_model = None
    if precomputed_embeddings is None:
        embedding_model = SentenceTransformer(sentence_model_name, device=device)

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

    try:
        topics, probs = topic_model.fit_transform(docs, embeddings=precomputed_embeddings)
    except ValueError as exc:
        message = str(exc).lower()
        # Handle common vectorizer pruning errors including max_df/min_df mismatches
        # and the "after pruning, no terms remain" case by falling back to a
        # minimal vectorizer configuration.
        if (
            "after pruning, no terms remain" in message
            or "max_df corresponds to" in message
            or "max_df is <= min_df" in message
            or "max_df .* < min_df" in message
        ):
            # Fallback for sparse or repetitive corpora where stopwords/max_df remove all terms.
            fallback_vectorizer = CountVectorizer(
                stop_words=None,
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0,
                token_pattern=r"(?u)\b\w+\b",
            )
            fallback_ctfidf = ClassTfidfTransformer(reduce_frequent_words=False)
            topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=fallback_vectorizer,
                ctfidf_model=fallback_ctfidf,
                seed_topic_list=guided_seed_topic_list,
                min_topic_size=min_topic_size,
                verbose=False,
            )
            topics, probs = topic_model.fit_transform(docs, embeddings=precomputed_embeddings)
        else:
            raise

    return topic_model, topics, probs


def encode_text_segments(
    segments: list[dict[str, Any]],
    sentence_model_name: str = "all-mpnet-base-v2",
    device: str = "cuda",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    docs = [str(seg.get("text", "")).strip() or "[EMPTY]" for seg in segments]
    model = SentenceTransformer(sentence_model_name, device=device)
    embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)
