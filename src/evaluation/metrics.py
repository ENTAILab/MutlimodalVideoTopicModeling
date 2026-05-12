from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_json


def _normalized_entropy(counts: list[int]) -> float:
    if not counts:
        return 0.0
    probs = np.asarray(counts, dtype=np.float64)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
    return float(entropy / max_entropy)


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    x = np.asarray(sorted(values), dtype=np.float64)
    n = x.size
    if n == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def _cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    unique = np.unique(valid_labels)
    if valid_labels.size < 3 or unique.size < 2:
        return {
            "silhouette_cosine": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }

    valid_embeddings = embeddings[valid_mask]
    try:
        silhouette = float(silhouette_score(valid_embeddings, valid_labels, metric="cosine"))
    except Exception:
        silhouette = None

    try:
        calinski = float(calinski_harabasz_score(valid_embeddings, valid_labels))
    except Exception:
        calinski = None

    try:
        davies = float(davies_bouldin_score(valid_embeddings, valid_labels))
    except Exception:
        davies = None

    return {
        "silhouette_cosine": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
    }


def _topic_word_stats(topic_info: list[dict[str, Any]]) -> dict[str, float]:
    words: list[str] = []
    for row in topic_info:
        topic_id = int(row.get("Topic", -1))
        if topic_id < 0:
            continue
        representation = [str(w).strip().lower() for w in row.get("Representation", []) if str(w).strip()]
        words.extend(representation[:10])

    unique_words = len(set(words))
    total_words = len(words)
    topic_diversity = float(unique_words / total_words) if total_words else 0.0
    return {
        "topic_top_words_total": float(total_words),
        "topic_top_words_unique": float(unique_words),
        "topic_diversity": topic_diversity,
    }


def _npmi_topic_coherence(topic_info: list[dict[str, Any]], docs: list[str]) -> float | None:
    cleaned_docs = [d for d in docs if d.strip()]
    if len(cleaned_docs) < 2:
        return None

    vectorizer = CountVectorizer(stop_words="english", binary=True)
    try:
        X = vectorizer.fit_transform(cleaned_docs)
    except ValueError:
        return None

    vocab = vectorizer.vocabulary_
    doc_count = X.shape[0]
    if doc_count == 0:
        return None

    X_bool = X.astype(bool).astype(np.int32)
    col_sums = np.asarray(X_bool.sum(axis=0)).reshape(-1)

    topic_scores: list[float] = []
    for row in topic_info:
        topic_id = int(row.get("Topic", -1))
        if topic_id < 0:
            continue

        words = [str(w).strip().lower() for w in row.get("Representation", []) if str(w).strip()]
        ids = [vocab[w] for w in words[:10] if w in vocab]
        if len(ids) < 2:
            continue

        pair_scores: list[float] = []
        for i, j in combinations(ids, 2):
            pi = col_sums[i] / doc_count
            pj = col_sums[j] / doc_count
            pij = float(X_bool[:, i].multiply(X_bool[:, j]).sum()) / doc_count
            if pi <= 0.0 or pj <= 0.0 or pij <= 0.0:
                continue
            pmi = np.log(pij / (pi * pj))
            npmi = pmi / (-np.log(pij))
            pair_scores.append(float(npmi))

        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    if not topic_scores:
        return None
    return float(np.mean(topic_scores))


def calculate_we_score(topic_embeddings_list, aggregate_model_score=True):
    """
        Word Embedding (WE) Score: Calculates pairwise cosine similarity.

        Args:
            topic_embeddings_list (list): A list where each element is a 2D numpy array
                                          of shape (num_words, embedding_dim) for a single topic.
            aggregate_model_score (bool): If True, returns the mean score across all topics (model level).
                                          If False, returns a list of scores for each individual topic.
        Returns:
            float or list: The overall model score, or a list of individual topic scores.
        """
    topic_scores = []

    for embeddings in topic_embeddings_list:
        if len(embeddings) < 2:
            topic_scores.append(0.0)
            continue

        sim_matrix = cosine_similarity(embeddings)
        pairwise_sims = sim_matrix[np.triu_indices(len(embeddings), k=1)]
        topic_scores.append(float(np.mean(pairwise_sims)))

    return float(np.mean(topic_scores)) if aggregate_model_score else topic_scores


def calculate_iec_score(topic_embeddings_list, aggregate_model_score=True):
    """
    Image Embedding-based Coherence (IEC): Calculates average pairwise visual similarity.

    Args:
        topic_embeddings_list (list): A list of 2D numpy arrays of multimodal embeddings.
        aggregate_model_score (bool): If True, returns the overall model mean (Eq 4).
                                      If False, returns individual topic scores (Eq 3).
    """
    topic_scores = []

    for embeddings in topic_embeddings_list:
        if len(embeddings) < 2:
            topic_scores.append(0.0)
            continue

        sim_matrix = cosine_similarity(embeddings)
        pairwise_sims = sim_matrix[np.triu_indices(len(embeddings), k=1)]
        topic_scores.append(float(np.mean(pairwise_sims)))

    return float(np.mean(topic_scores)) if aggregate_model_score else topic_scores


def calculate_ieps_score(topic_embeddings_list, return_pairwise_scores=False):
    """
    Image Embedding-based Pairwise Similarity (IEPS): Measures the diversity of a
    topic model by computing similarity BETWEEN all pairs of topics.
    (Lower score = Higher diversity).

    Args:
        topic_embeddings_list (list): A list of 2D numpy arrays of multimodal embeddings.
        return_pairwise_scores (bool): If True, returns both the overall score and the
                                       list of individual topic-pair scores.
    Returns:
        float: The overall IEPS score for the model.
    """
    num_topics = len(topic_embeddings_list)

    if num_topics < 2:
        print("Warning: Need at least 2 topics to calculate IEPS.")
        return 0.0

    pairwise_topic_scores = []
    for i, j in combinations(range(num_topics), 2):
        emb_i = topic_embeddings_list[i]
        emb_j = topic_embeddings_list[j]

        if len(emb_i) == 0 or len(emb_j) == 0:
            pairwise_topic_scores.append(0.0)
            continue

        cross_sim_matrix = cosine_similarity(emb_i, emb_j)
        ieps_pair_score = float(np.mean(cross_sim_matrix))
        pairwise_topic_scores.append(ieps_pair_score)

    overall_ieps = float(np.mean(pairwise_topic_scores))

    if return_pairwise_scores:
        return overall_ieps, pairwise_topic_scores
    return overall_ieps


def _compute_embedding_coherence_scores(
    embeddings: np.ndarray,
    topics: np.ndarray,
    topic_info: list[dict[str, Any]] | None = None,
    word_embedding_model_name: str = "all-mpnet-base-v2",
) -> dict[str, float | None]:
    """
    Compute IEC and WE scores for embeddings grouped by topic.

    IEC (Image Embedding-based Coherence) is computed from the provided segment-level
    embeddings grouped by topic (suitable for visual embeddings).

    WE (Word Embedding score) is computed from top topic words (from topic_info)
    by embedding the words with a sentence-transformer and computing intra-topic
    pairwise similarity.

    Args:
        embeddings: 2D numpy array of shape (num_segments, embedding_dim)
        topics: 1D numpy array of topic labels for each segment
        topic_info: Optional list of topic dictionaries containing 'Topic' and
                    'Representation' (top words) for each topic.
        word_embedding_model_name: sentence-transformers model to use for word embeddings.

    Returns:
        Dictionary with 'iec_score' and 'we_score' keys
    """
    if embeddings.shape[0] != topics.shape[0]:
        return {"iec_score": None, "we_score": None}

    valid_mask = topics >= 0
    valid_topics = topics[valid_mask]
    valid_embeddings = embeddings[valid_mask]

    if len(valid_topics) == 0:
        return {"iec_score": None, "we_score": None}

    unique_topics = np.unique(valid_topics)
    topic_embeddings_list = []
    for topic_id in sorted(unique_topics):
        topic_mask = valid_topics == topic_id
        topic_embs = valid_embeddings[topic_mask]
        if len(topic_embs) > 0:
            topic_embeddings_list.append(topic_embs)

    if not topic_embeddings_list:
        return {"iec_score": None, "we_score": None}

    # IEC: visual / segment-level embedding coherence
    try:
        iec = calculate_iec_score(topic_embeddings_list, aggregate_model_score=True)
    except Exception:
        iec = None

    # WE: compute from topic top-words (if provided); fall back to segment-level
    we = None
    if topic_info is not None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(word_embedding_model_name)

            topic_word_embs = []
            # Build a mapping from topic id -> representation words
            topic_repr_map: dict[int, list[str]] = {}
            for row in topic_info:
                tid = int(row.get("Topic", -1))
                if tid < 0:
                    continue
                reps = [str(w).strip() for w in row.get("Representation", []) if str(w).strip()]
                if reps:
                    topic_repr_map[tid] = reps[:20]

            for topic_id in sorted(unique_topics):
                words = topic_repr_map.get(int(topic_id), [])
                if not words:
                    # If no words for this topic, append empty array placeholder
                    topic_word_embs.append(np.zeros((0, model.get_sentence_embedding_dimension())))
                    continue
                emb = model.encode(words, convert_to_numpy=True)
                topic_word_embs.append(emb)

            we = calculate_we_score(topic_word_embs, aggregate_model_score=True)
        except Exception:
            # If sentence-transformers is unavailable or encoding fails, fall back
            # to computing WE on segment-level embeddings (previous behavior).
            try:
                we = calculate_we_score(topic_embeddings_list, aggregate_model_score=True)
            except Exception:
                we = None
    else:
        try:
            we = calculate_we_score(topic_embeddings_list, aggregate_model_score=True)
        except Exception:
            we = None

    return {"iec_score": iec, "we_score": we}

def compute_numeric_metrics(
    segments: list[dict[str, Any]],
    processed_dir: str | Path,
    output_dir: str | Path,
    topic_info_override: list[dict[str, Any]] | None = None,
    topic_labels_override: list[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)

    docs = [str(seg.get("text", "")).strip() for seg in segments]
    durations = [float(seg.get("end_s", 0.0)) - float(seg.get("start_s", 0.0)) for seg in segments]
    if topic_labels_override is None:
        topics = np.asarray([int(seg.get("topic", -1)) for seg in segments], dtype=np.int32)
    else:
        topics = np.asarray(topic_labels_override, dtype=np.int32)
        if topics.shape[0] != len(segments):
            raise ValueError("topic_labels_override must have one label per segment")

    topic_counts = Counter(int(t) for t in topics.tolist())
    non_noise_counts = [count for topic, count in topic_counts.items() if topic >= 0]

    topic_info: list[dict[str, Any]] = []
    if topic_info_override is not None:
        topic_info = topic_info_override
    else:
        topic_info_path = output_path / "topic_info.json"
        if topic_info_path.exists():
            topic_info = load_json(topic_info_path)

    total_segments = int(len(segments))
    topic_changes = int(np.sum(topics[1:] != topics[:-1])) if total_segments > 1 else 0

    token_counts = [len(doc.split()) for doc in docs if doc]
    vocab_size = len(set(word.lower() for doc in docs for word in doc.split()))

    metrics: dict[str, Any] = {
        "num_segments": total_segments,
        "total_duration_s": float(np.sum(durations)) if durations else 0.0,
        "avg_segment_duration_s": float(np.mean(durations)) if durations else 0.0,
        "avg_tokens_per_segment": float(np.mean(token_counts)) if token_counts else 0.0,
        "vocabulary_size": int(vocab_size),
        "num_topics_excluding_noise": int(len([t for t in topic_counts if t >= 0])),
        "noise_segment_count": int(topic_counts.get(-1, 0)),
        "noise_ratio": float(topic_counts.get(-1, 0) / total_segments) if total_segments else 0.0,
        "topic_entropy_normalized": _normalized_entropy(non_noise_counts),
        "topic_size_gini": _gini(non_noise_counts),
        "topic_transition_rate": float(topic_changes / max(total_segments - 1, 1)),
        "topic_word_metrics": _topic_word_stats(topic_info),
        "topic_coherence_npmi": _npmi_topic_coherence(topic_info, docs),
        "embedding_cluster_metrics": {},
    }

    for name in ("audio_embeddings", "visual_embeddings", "fused_embeddings", "multimodal_topic_embeddings"):
        emb_path = processed_path / f"{name}.npy"
        if not emb_path.exists():
            continue
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != topics.shape[0]:
            continue
        cluster_metrics = _cluster_metrics(embeddings, topics)
        coherence_scores = _compute_embedding_coherence_scores(embeddings, topics, topic_info=topic_info)
        metrics["embedding_cluster_metrics"][name] = {**cluster_metrics, **coherence_scores}

    return metrics
