from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


def _load_wav_tensor(wav_path: str | Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 1:
        arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        arr = arr.reshape(-1, channels).T
    else:
        arr = arr.reshape(1, -1)
    return torch.from_numpy(arr), sample_rate


def _mfcc_fallback(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float, bins: int = 64) -> np.ndarray:
    s0 = max(0, int(start_s * sample_rate))
    s1 = max(s0 + 1, int(end_s * sample_rate))
    chunk = waveform[:, s0:s1]

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=min(40, bins),
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": max(64, bins)},
    )(chunk)
    emb = mfcc.mean(dim=-1).flatten().detach().cpu().numpy().astype(np.float32)
    return emb


def embed_segments(
    wav_path: str | Path,
    segments: list[dict[str, Any]],
    model_name: str = "pyannote/embedding",
    hf_token_env: str = "HF_TOKEN",
    fallback_bins: int = 64,
) -> np.ndarray:
    waveform, sample_rate = _load_wav_tensor(wav_path)

    try:
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment

        token = os.getenv(hf_token_env)
        model = Model.from_pretrained(model_name, token=token)
        infer = Inference(model, window="whole")
        audio = {"waveform": waveform, "sample_rate": sample_rate}
        vectors: list[np.ndarray] = []
        for seg in segments:
            emb = infer.crop(audio, Segment(seg["start_s"], seg["end_s"]))
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            vectors.append(arr)
        return np.vstack(vectors)
    except Exception:
        vectors = [
            _mfcc_fallback(waveform, sample_rate, seg["start_s"], seg["end_s"], bins=fallback_bins)
            for seg in segments
        ]
        return np.vstack(vectors)
