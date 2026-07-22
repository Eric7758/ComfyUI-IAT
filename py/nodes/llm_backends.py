from __future__ import annotations

"""Offline-friendly text/vision generation adapters.

Ollama uses its native local API; vLLM uses the OpenAI-compatible API; Local
delegates to the existing Transformers runtime and its in-process model cache.
"""

import base64
import json
import socket
from http.client import RemoteDisconnected
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, request

from PIL import Image

class BackendError(RuntimeError):
    """A generation backend could not fulfill a request."""


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(_pil_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pil_to_ollama_image(image: Image.Image) -> str:
    return base64.b64encode(_pil_to_png_bytes(image)).decode("ascii")


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        choice_message = choices[0].get("message") or {}
        content = choice_message.get("content") if isinstance(choice_message, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts).strip()
    response = payload.get("response")
    return response.strip() if isinstance(response, str) else ""


def _request_json(url: str, payload: Dict[str, Any], timeout: int, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=max(5, int(timeout))) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendError(f"[IAT] Backend HTTP {exc.code} from `{url}`: {detail[:500]}") from exc
    except (error.URLError, RemoteDisconnected, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise BackendError(f"[IAT] Cannot connect to generation backend `{url}`: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendError(f"[IAT] Generation backend returned invalid JSON from `{url}`.") from exc
    if not isinstance(value, dict):
        raise BackendError(f"[IAT] Generation backend returned an unexpected response from `{url}`.")
    return value


def _normalize_ollama_url(base_url: str) -> str:
    root = (base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    if root.endswith("/api"):
        return root
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return f"{root}/api"


def _normalize_vllm_url(base_url: str) -> str:
    root = (base_url or "http://127.0.0.1:8000/v1").strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    return f"{root}/chat/completions"


def _generate_ollama(
    *,
    model: str,
    base_url: str,
    prompt: str,
    images: Optional[List[Image.Image]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    timeout: int,
    keep_alive: Any,
    think: bool,
    system_prompt: str = "",
) -> str:
    if not model.strip():
        raise BackendError("[IAT] Ollama model name is empty.")
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = [_pil_to_ollama_image(image) for image in images]
    options: Dict[str, Any] = {
        "num_predict": int(max_tokens),
        "temperature": max(0.0, float(temperature)),
        "top_p": max(0.0, min(1.0, float(top_p))),
        "repeat_penalty": float(repetition_penalty),
        "seed": int(seed),
    }
    messages = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append(message)
    payload = {
        "model": model.strip(),
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "think": bool(think),
        "options": options,
    }
    response_payload = _request_json(f"{_normalize_ollama_url(base_url)}/chat", payload, timeout)
    text = _extract_text(response_payload)
    if not text:
        raise BackendError("[IAT] Ollama returned an empty response.")
    return text


def _generate_vllm(
    *,
    model: str,
    base_url: str,
    prompt: str,
    images: Optional[List[Image.Image]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    timeout: int,
    api_key: str,
    system_prompt: str = "",
) -> str:
    if not model.strip():
        raise BackendError("[IAT] vLLM model name is empty.")
    content: Any = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}})
    messages = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})
    payload: Dict[str, Any] = {
        "model": model.strip(),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": max(0.0, float(temperature)),
        "top_p": max(0.0, min(1.0, float(top_p))),
        "seed": int(seed),
        "repetition_penalty": float(repetition_penalty),
    }
    api_key_text = (api_key or "").strip()
    headers = {"Authorization": f"Bearer {api_key_text}"} if api_key_text else {}
    response_payload = _request_json(_normalize_vllm_url(base_url), payload, timeout, headers=headers)
    text = _extract_text(response_payload)
    if not text:
        raise BackendError("[IAT] vLLM returned an empty response.")
    return text


def generate_with_backend(
    *,
    backend: str,
    model: str,
    base_url: str,
    prompt: str,
    images: Optional[List[Image.Image]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    timeout: int,
    local_device: str = "cuda",
    local_attention_backend: Optional[str] = None,
    keep_local_model_loaded: bool = True,
    ollama_keep_alive: Any = -1,
    ollama_think: bool = False,
    vllm_api_key: str = "",
    system_prompt: str = "",
) -> str:
    system_text = (system_prompt or "").strip()
    normalized = (backend or "Local").strip()
    if normalized == "Ollama":
        return _generate_ollama(
            model=model,
            base_url=base_url,
            prompt=prompt,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            timeout=timeout,
            keep_alive=ollama_keep_alive,
            think=ollama_think,
            system_prompt=system_text,
        )
    if normalized == "vLLM":
        return _generate_vllm(
            model=model,
            base_url=base_url,
            prompt=prompt,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            timeout=timeout,
            api_key=vllm_api_key,
            system_prompt=system_text,
        )

    if normalized != "Local":
        raise BackendError(f"[IAT] Unsupported generation backend: `{backend}`")
    try:
        # Keep remote backends usable in lightweight environments where the
        # optional Local Transformers/Torch stack is not installed.
        from .qwen35_runtime import generate_text, generate_vision_text, unload_all_models

        if images:
            text = generate_vision_text(
                variant=model,
                device=local_device,
                attention_backend=local_attention_backend,
                images=images,
                text_prompt=prompt,
                system_prompt=system_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        else:
            text = generate_text(
                variant=model,
                device=local_device,
                attention_backend=local_attention_backend,
                messages=(
                    ([{"role": "system", "content": system_text}] if system_text else [])
                    + [{"role": "user", "content": prompt}]
                ),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
    except Exception as exc:
        raise BackendError(f"[IAT] Local Transformers generation failed: {exc}") from exc
    if not keep_local_model_loaded:
        unload_all_models()
    if not text.strip():
        raise BackendError("[IAT] Local Transformers returned an empty response.")
    return text.strip()
