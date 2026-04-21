from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
import open_clip
import torch
from PIL import Image


def _embed_frames_openclip(
    segment_to_frames: dict[int, list[str]],
    model_name: str,
    pretrained: str,
    device: str,
    fallback_embedding_dim: int,
) -> dict[int, np.ndarray]:
    dev = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=dev)
    model.eval()

    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for seg_id, frame_paths in segment_to_frames.items():
            frame_embeddings: list[np.ndarray] = []
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(dev)
                emb = model.encode_image(tensor)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                frame_embeddings.append(emb.squeeze(0).cpu().numpy().astype(np.float32))

            if frame_embeddings:
                out[seg_id] = np.mean(np.vstack(frame_embeddings), axis=0)
            else:
                out[seg_id] = np.zeros((fallback_embedding_dim,), dtype=np.float32)

    return out


def _image_to_data_uri(image_path: str | Path) -> str:
    with Path(image_path).open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_embedding_response(response_payload: dict[str, Any]) -> np.ndarray:
    if isinstance(response_payload.get("data"), list) and response_payload["data"]:
        first = response_payload["data"][0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return np.asarray(first["embedding"], dtype=np.float32)

    if isinstance(response_payload.get("embedding"), list):
        return np.asarray(response_payload["embedding"], dtype=np.float32)

    if isinstance(response_payload.get("vector"), list):
        return np.asarray(response_payload["vector"], dtype=np.float32)

    raise RuntimeError("Unable to parse embedding from vLLM API response")


def _fetch_vllm_embedding(
    image_path: str | Path,
    base_url: str,
    endpoint: str,
    model: str,
    timeout_s: float,
    api_key_env: str | None,
) -> np.ndarray:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    data_uri = _image_to_data_uri(image_path)
    payload = {
        "model": model,
        "input": data_uri,
        "encoding_format": "float",
    }

    headers = {"Content-Type": "application/json"}
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(url=url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    return _parse_embedding_response(parsed)


def _embed_frames_vllm_api(
    segment_to_frames: dict[int, list[str]],
    base_url: str,
    endpoint: str,
    model: str,
    timeout_s: float,
    api_key_env: str | None,
    fallback_embedding_dim: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    inferred_dim: int | None = None

    for seg_id, frame_paths in segment_to_frames.items():
        frame_embeddings: list[np.ndarray] = []
        for frame_path in frame_paths:
            emb = _fetch_vllm_embedding(
                image_path=frame_path,
                base_url=base_url,
                endpoint=endpoint,
                model=model,
                timeout_s=timeout_s,
                api_key_env=api_key_env,
            )
            frame_embeddings.append(emb)
            inferred_dim = emb.shape[0]

        if frame_embeddings:
            out[seg_id] = np.mean(np.vstack(frame_embeddings), axis=0).astype(np.float32)
        else:
            dim = inferred_dim if inferred_dim is not None else fallback_embedding_dim
            out[seg_id] = np.zeros((dim,), dtype=np.float32)

    return out


def embed_frames_per_segment(
    segment_to_frames: dict[int, list[str]],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda",
    backend: str = "clip",
    vllm_base_url: str = "http://localhost:8000/v1",
    vllm_endpoint: str = "/embeddings",
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_api_key_env: str | None = "VLLM_API_KEY",
    vllm_timeout_s: float = 60.0,
    fallback_embedding_dim: int = 512,
) -> dict[int, np.ndarray]:
    selected_backend = backend.strip().lower()
    if selected_backend in {"clip", "open_clip", "openclip"}:
        return _embed_frames_openclip(
            segment_to_frames=segment_to_frames,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            fallback_embedding_dim=fallback_embedding_dim,
        )

    if selected_backend in {"vllm", "vllm_api"}:
        return _embed_frames_vllm_api(
            segment_to_frames=segment_to_frames,
            base_url=vllm_base_url,
            endpoint=vllm_endpoint,
            model=vllm_model,
            timeout_s=vllm_timeout_s,
            api_key_env=vllm_api_key_env,
            fallback_embedding_dim=fallback_embedding_dim,
        )

    raise ValueError(f"Unknown visual embedding backend '{backend}'. Use 'clip' or 'vllm_api'.")


def to_segment_matrix(segments: list[dict[str, Any]], visual_map: dict[int, np.ndarray]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for seg in segments:
        vectors.append(visual_map[int(seg["segment_id"])])
    return np.vstack(vectors)
