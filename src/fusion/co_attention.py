from __future__ import annotations

import numpy as np

from src.utils import l2_normalize


def _align_to_common_dim(*arrays: np.ndarray) -> list[np.ndarray]:
    min_dim = min(arr.shape[1] for arr in arrays)
    return [arr[:, :min_dim] for arr in arrays]


def co_attention_fuse(
    audio_vectors: np.ndarray,
    visual_vectors: np.ndarray,
    weight_audio: float = 0.5,
    weight_visual: float = 0.5,
) -> np.ndarray:
    if len(audio_vectors) != len(visual_vectors):
        raise ValueError("Audio and visual vectors must have the same segment count")

    audio_n = l2_normalize(audio_vectors)
    visual_n = l2_normalize(visual_vectors)

    a_cut, v_cut = _align_to_common_dim(audio_n, visual_n)

    sim = (a_cut * v_cut).sum(axis=1, keepdims=True)
    sim_scale = (sim + 1.0) / 2.0

    a_weighted = a_cut * (weight_audio * sim_scale)
    v_weighted = v_cut * (weight_visual * sim_scale)
    interaction = a_cut * v_cut

    fused = np.concatenate([a_weighted, v_weighted, interaction], axis=1)
    return l2_normalize(fused)


def co_attention_fuse_multimodal(
    text_vectors: np.ndarray,
    audio_vectors: np.ndarray,
    visual_vectors: np.ndarray,
    weight_text: float = 0.34,
    weight_audio: float = 0.33,
    weight_visual: float = 0.33,
) -> np.ndarray:
    if len(text_vectors) != len(audio_vectors) or len(text_vectors) != len(visual_vectors):
        raise ValueError("Text, audio, and visual vectors must have the same segment count")

    text_n = l2_normalize(text_vectors)
    audio_n = l2_normalize(audio_vectors)
    visual_n = l2_normalize(visual_vectors)
    t_cut, a_cut, v_cut = _align_to_common_dim(text_n, audio_n, visual_n)

    sim_ta = (t_cut * a_cut).sum(axis=1, keepdims=True)
    sim_tv = (t_cut * v_cut).sum(axis=1, keepdims=True)
    sim_av = (a_cut * v_cut).sum(axis=1, keepdims=True)
    sim_scale = (sim_ta + sim_tv + sim_av + 3.0) / 6.0

    t_weighted = t_cut * (weight_text * sim_scale)
    a_weighted = a_cut * (weight_audio * sim_scale)
    v_weighted = v_cut * (weight_visual * sim_scale)

    inter_ta = t_cut * a_cut
    inter_tv = t_cut * v_cut
    inter_av = a_cut * v_cut
    inter_tav = t_cut * a_cut * v_cut

    fused = np.concatenate(
        [
            t_weighted,
            a_weighted,
            v_weighted,
            inter_ta,
            inter_tv,
            inter_av,
            inter_tav,
        ],
        axis=1,
    )
    return l2_normalize(fused)
