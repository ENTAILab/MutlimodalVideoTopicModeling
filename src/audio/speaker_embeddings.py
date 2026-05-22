from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm


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


def _segment_waveform(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float) -> torch.Tensor:
    s0 = max(0, int(start_s * sample_rate))
    s1 = max(s0 + 1, int(end_s * sample_rate))
    return waveform[:, s0:s1]


def _resample_waveform(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)


def _mono_numpy_audio(waveform: torch.Tensor) -> np.ndarray:
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _mfcc_fallback(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float, bins: int = 64) -> np.ndarray:
    chunk = _segment_waveform(waveform, sample_rate, start_s, end_s)

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=min(40, bins),
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": max(64, bins)},
    )(chunk)
    emb = mfcc.mean(dim=-1).flatten().detach().cpu().numpy().astype(np.float32)
    return emb


def _pyannote_embeddings(
    waveform: torch.Tensor,
    sample_rate: int,
    segments: list[dict[str, Any]],
    model_name: str,
    hf_token_env: str,
) -> np.ndarray:
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    token = os.getenv(hf_token_env)
    model = Model.from_pretrained(model_name, token=token)
    infer = Inference(model, window="whole")
    audio = {"waveform": waveform, "sample_rate": sample_rate}

    vectors: list[np.ndarray] = []
    for seg in tqdm(segments, desc="pyannote audio embeddings", unit="seg"):
        emb = infer.crop(audio, Segment(seg["start_s"], seg["end_s"]))
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        vectors.append(arr)
    return np.vstack(vectors)


def _clap_embeddings(
    waveform: torch.Tensor,
    sample_rate: int,
    segments: list[dict[str, Any]],
    model_name: str,
    device: str,
    target_sample_rate: int,
) -> np.ndarray:
    from transformers import ClapModel, ClapProcessor

    dev = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(dev)
    model.eval()

    vectors: list[np.ndarray] = []
    for seg in tqdm(segments, desc="CLAP audio embeddings", unit="seg"):
        chunk = _segment_waveform(waveform, sample_rate, seg["start_s"], seg["end_s"])
        chunk = _resample_waveform(chunk, sample_rate, target_sample_rate)
        audio_np = _mono_numpy_audio(chunk)
        inputs = processor(audio=audio_np, sampling_rate=target_sample_rate, return_tensors="pt")
        inputs = {key: value.to(dev) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.inference_mode():
            emb = model.get_audio_features(**inputs)
        vectors.append(emb.squeeze(0).detach().cpu().numpy().astype(np.float32))

    return np.vstack(vectors)


def embed_segments(
    wav_path: str | Path,
    segments: list[dict[str, Any]],
    backend: str = "pyannote_fallback",
    model_name: str = "pyannote/embedding",
    hf_token_env: str = "HF_TOKEN",
    clap_model_name: str = "laion/clap-htsat-unfused",
    clap_device: str = "cuda",
    clap_sampling_rate: int = 48000,
    fallback_bins: int = 64,
) -> np.ndarray:
    waveform, sample_rate = _load_wav_tensor(wav_path)

    selected_backend = backend.strip().lower()
    valid_backends = {"pyannote", "pyannote_fallback", "pyannote-fallback", "pyannote_with_fallback", "fallback", "fallback_pyannote", "clap", "clap_audio"}
    if selected_backend not in valid_backends:
        raise ValueError(f"Unknown speaker embedding backend '{backend}'. Use 'pyannote_fallback' or 'clap'.")

    try:
        if selected_backend in {"pyannote", "pyannote_fallback", "pyannote-fallback", "fallback", "fallback_pyannote"}:
            return _pyannote_embeddings(
                waveform=waveform,
                sample_rate=sample_rate,
                segments=segments,
                model_name=model_name,
                hf_token_env=hf_token_env,
            )

        if selected_backend in {"clap", "clap_audio"}:
            return _clap_embeddings(
                waveform=waveform,
                sample_rate=sample_rate,
                segments=segments,
                model_name=clap_model_name,
                device=clap_device,
                target_sample_rate=clap_sampling_rate,
            )
    except Exception as e:
        print(f"Error occurred while computing speaker embeddings with backend '{backend}', fallback mechanism used, error: {e}")
        vectors = [
            _mfcc_fallback(waveform, sample_rate, seg["start_s"], seg["end_s"], bins=fallback_bins)
            for seg in tqdm(segments, desc="MFCC fallback embeddings", unit="seg")
        ]
        return np.vstack(vectors)
