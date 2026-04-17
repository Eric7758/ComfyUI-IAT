import asyncio
import base64
import json
import os
import re
import socket
import sys
from io import BytesIO
from typing import Optional
from urllib import error, request

from PIL import Image

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

from .qwen35_runtime import (
    ATTENTION_OPTIONS,
    DEFAULT_ATTENTION_BACKEND,
    DEVICE_OPTIONS,
    QUANT_OPTIONS,
    TEXT_MODEL_CANDIDATES,
    VL_MODEL_CANDIDATES,
    generate_text,
    generate_vision_text,
    unload_all_models,
)

_CFG = getattr(sys.modules.get("comfyui_iat_config"), "data", {}) or {}
_CFG_PATH = getattr(sys.modules.get("comfyui_iat_config"), "path", "config.yaml")
_MODEL_CFG = (_CFG.get("model") or {}) if isinstance(_CFG, dict) else {}
_OPENAI_CFG = (_CFG.get("openai") or {}) if isinstance(_CFG, dict) else {}
_GEMINI_CFG = (_CFG.get("gemini") or {}) if isinstance(_CFG, dict) else {}
_QWEN_COMPAT_CFG = (_CFG.get("qwen_compatible") or {}) if isinstance(_CFG, dict) else {}
_DEFAULT_VARIANT = _MODEL_CFG.get("default_variant", "Qwen3.5-2B")
_DEFAULT_QUANT = _MODEL_CFG.get("quantization", "无")
_DEFAULT_DEVICE = _MODEL_CFG.get("device", "cuda")
_DEFAULT_OPENAI_BASE_URL = (_OPENAI_CFG.get("base_url") or "https://api.openai.com/v1").strip()
_DEFAULT_OPENAI_MODEL = (_OPENAI_CFG.get("model") or "gpt-4.1-mini").strip()
_DEFAULT_OPENAI_API_KEY = (_OPENAI_CFG.get("api_key") or "").strip()
_DEFAULT_OPENAI_API_KEY_ENV = (_OPENAI_CFG.get("api_key_env") or "OPENAI_API_KEY").strip()
_DEFAULT_HTTP_USER_AGENT = (_OPENAI_CFG.get("user_agent") or "ComfyUI-IAT/1.22").strip()
_CFG_FILE_NAME = os.path.basename(_CFG_PATH) or "config.yaml"
TRANSLATION_TARGET_OPTIONS = ["English", "中文"]
GPT_IMAGE_DETAIL_OPTIONS = ["auto", "low", "high"]
_GPT_API_ROUTES_REGISTERED = False
VISION_API_PROVIDER_OPTIONS = [
    "OpenAI-Compatible",
    "Gemini",
    "Qwen OpenAI-Compatible",
]

VISION_API_PROVIDER_CONFIGS = {
    "OpenAI-Compatible": {
        "config_key": "openai",
        "default_base_url": _DEFAULT_OPENAI_BASE_URL,
        "default_model": _DEFAULT_OPENAI_MODEL,
        "default_api_key": _DEFAULT_OPENAI_API_KEY,
        "default_api_key_env": _DEFAULT_OPENAI_API_KEY_ENV,
        "mode": "openai_compatible",
    },
    "Gemini": {
        "config_key": "gemini",
        "default_base_url": (_GEMINI_CFG.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").strip(),
        "default_model": (_GEMINI_CFG.get("model") or "gemini-2.5-flash").strip(),
        "default_api_key": (_GEMINI_CFG.get("api_key") or "").strip(),
        "default_api_key_env": (_GEMINI_CFG.get("api_key_env") or "GEMINI_API_KEY").strip(),
        "mode": "gemini",
    },
    "Qwen OpenAI-Compatible": {
        "config_key": "qwen_compatible",
        "default_base_url": (_QWEN_COMPAT_CFG.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
        "default_model": (_QWEN_COMPAT_CFG.get("model") or "qwen-vl-plus").strip(),
        "default_api_key": (_QWEN_COMPAT_CFG.get("api_key") or "").strip(),
        "default_api_key_env": (_QWEN_COMPAT_CFG.get("api_key_env") or "DASHSCOPE_API_KEY").strip(),
        "mode": "openai_compatible",
    },
}

PROMPT_STYLES = {
    "Enhance": "Expand and enrich this prompt with vivid details while keeping the original intent.",
    "Refine": "Rewrite this prompt to be clear, concise, and production-ready for image generation.",
    "Creative Rewrite": "Rewrite this prompt creatively with stronger visual storytelling.",
    "Detailed Visual": "Convert this prompt into a highly detailed visual description for text-to-image models.",
}

REVERSE_PRESETS = {
    "Detailed Description": "Describe this image in detail and output an English image generation prompt only.",
    "Prompt Reverse": "Infer a compact production-grade positive prompt from this image. English only.",
    "Style Focus": "Describe style, camera, lighting, color, composition, and materials as a reusable generation prompt.",
}

KONTEXT_SYSTEM_PROMPT = """You are a prompt engineer specialized in image editing prompts.
Rewrite user requests into clean English prompts that are explicit, editable, and production-ready.
Rules:
1) Be concrete and visual, avoid vague words.
2) Preserve identity and composition when user requires consistency.
3) For replacement tasks, use exact text/object names.
4) Output only one final English prompt, no explanations.
"""


def _tensor_to_pil_list(image):
    if image is None:
        return []
    if image.dim() == 3:
        image = image.unsqueeze(0)

    pil_images = []
    for i in range(image.shape[0]):
        array = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        pil_images.append(Image.fromarray(array))
    return pil_images


def _collect_pil_images(*image_inputs):
    merged = []
    for img in image_inputs:
        merged.extend(_tensor_to_pil_list(img))
    return merged


def _detect_language(text: str):
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text):
        return "ja" if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text) else "zh"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return None


def _coerce_positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class GPTAPIError(RuntimeError):
    def __init__(self, message: str, category: str = "unknown_error", status_code: Optional[int] = None):
        super().__init__(message)
        self.category = category
        self.status_code = status_code


def _get_provider_config(provider: str):
    return VISION_API_PROVIDER_CONFIGS.get(provider, VISION_API_PROVIDER_CONFIGS["OpenAI-Compatible"])


def _resolve_provider_api_key(provider: str, api_key: str):
    provider_cfg = _get_provider_config(provider)
    explicit = (api_key or "").strip() or provider_cfg.get("default_api_key", "")
    if explicit:
        return explicit
    env_name = provider_cfg.get("default_api_key_env", "")
    if env_name:
        return (os.getenv(env_name) or "").strip()
    return ""


def _normalize_openai_api_root(base_url: str, default_base_url: str):
    normalized = (base_url or "").strip().rstrip("/") or default_base_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _build_chat_completions_url(base_url: str, default_base_url: str):
    return f"{_normalize_openai_api_root(base_url, default_base_url)}/chat/completions"


def _build_models_url(base_url: str, default_base_url: str):
    return f"{_normalize_openai_api_root(base_url, default_base_url)}/models"


def _normalize_gemini_api_root(base_url: str, default_base_url: str):
    normalized = (base_url or "").strip().rstrip("/") or default_base_url.rstrip("/")
    if "/models/" in normalized:
        normalized = normalized.split("/models/", 1)[0]
    if normalized.endswith(":generateContent"):
        normalized = normalized.rsplit("/", 1)[0]
    if normalized.endswith("/v1beta") or normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1beta"


def _normalize_gemini_model_name(model: str, default_model: str):
    value = (model or "").strip() or default_model
    if value.startswith("models/"):
        return value[len("models/") :]
    return value


def _build_gemini_generate_url(base_url: str, model: str, default_base_url: str, default_model: str):
    root = _normalize_gemini_api_root(base_url, default_base_url)
    model_name = _normalize_gemini_model_name(model, default_model)
    return f"{root}/models/{model_name}:generateContent"


def _build_gemini_models_url(base_url: str, default_base_url: str):
    return f"{_normalize_gemini_api_root(base_url, default_base_url)}/models"


def _pil_to_data_url(image: Image.Image):
    safe_image = image.convert("RGB") if image.mode not in {"RGB", "RGBA"} else image
    buffer = BytesIO()
    safe_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_chat_text(payload):
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            text_value = item.get("text")
            if isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                parts.append(text_value["value"])
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()

    refusal = message.get("refusal")
    if isinstance(refusal, str):
        return refusal.strip()
    return ""


def _extract_api_error_details(raw_text: str):
    result = {
        "message": raw_text.strip(),
        "code": None,
        "type": None,
    }
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return result

    err = payload.get("error")
    if isinstance(err, dict):
        if isinstance(err.get("message"), str):
            result["message"] = err["message"].strip()
        if isinstance(err.get("code"), str):
            result["code"] = err["code"].strip()
        if isinstance(err.get("type"), str):
            result["type"] = err["type"].strip()
        return result

    if isinstance(err, str) and err.strip():
        result["message"] = err.strip()
    return result


def _classify_http_error(status_code: int, message: str, base_url: str, endpoint_kind: str):
    normalized_message = (message or "").strip()
    lowered = normalized_message.lower()
    normalized_url = (base_url or _DEFAULT_OPENAI_BASE_URL).strip()

    if status_code == 400:
        if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
            return GPTAPIError(
                f"Invalid model name. The upstream API could not find the specified model. Raw upstream message: {normalized_message}",
                "invalid_model",
                status_code,
            )
        return GPTAPIError(
            f"Vision API rejected the request as invalid (400). Check model, URL compatibility, and request parameters. Raw upstream message: {normalized_message}",
            "invalid_request",
            status_code,
        )

    if status_code == 401 or "invalid api key" in lowered or "incorrect api key" in lowered or "unauthorized" in lowered:
        return GPTAPIError(
            f"Invalid API key or unauthorized request. Check api_key and provider account permissions. Raw upstream message: {normalized_message}",
            "invalid_api_key",
            status_code,
        )

    if status_code == 402 or "insufficient balance" in lowered or "余额不足" in lowered or "payment required" in lowered:
        return GPTAPIError(
            f"Upstream API balance is insufficient. Recharge the provider account or switch to another key. Raw upstream message: {normalized_message}",
            "insufficient_balance",
            status_code,
        )

    if status_code == 403 and (
        "insufficient balance" in lowered
        or "quota" in lowered
        or "balance" in lowered
        or "billing" in lowered
        or "credit" in lowered
    ):
        return GPTAPIError(
            f"Upstream API rejected the request because the account has no remaining balance or quota. Raw upstream message: {normalized_message}",
            "insufficient_balance",
            status_code,
        )

    if status_code == 403 and (
        "1010" in lowered
        or "1020" in lowered
        or "cloudflare" in lowered
        or "access denied" in lowered
    ):
        if "api.openai.com" in normalized_url:
            return GPTAPIError(
                "Vision API access was denied by the upstream gateway (403 / Cloudflare-style block). "
                "If this request is going to the official OpenAI endpoint, check whether your egress IP is in a supported country/territory, "
                f"or switch to a compliant proxy/base_url. Raw upstream message: {normalized_message}",
                "gateway_blocked",
                status_code,
            )
        return GPTAPIError(
            "Vision API access was denied by the configured gateway (403 / Cloudflare-style block). "
            "This usually means the proxy/base_url is rejecting the current client IP, request signature, or account. "
            f"Try another compatible gateway or inspect its WAF rules. Raw upstream message: {normalized_message}",
            "gateway_blocked",
            status_code,
        )

    if status_code == 403:
        return GPTAPIError(
            f"Upstream API denied access (403). Check key permissions, model permissions, or gateway policy. Raw upstream message: {normalized_message}",
            "access_denied",
            status_code,
        )

    if status_code == 404:
        if endpoint_kind == "models":
            return GPTAPIError(
                "Model list endpoint was not found. Check base_url or whether this provider exposes a compatible /models endpoint. "
                f"Raw upstream message: {normalized_message}",
                "endpoint_not_found",
                status_code,
            )
        return GPTAPIError(
            f"Vision API endpoint was not found. Check base_url. Raw upstream message: {normalized_message}",
            "endpoint_not_found",
            status_code,
        )

    if status_code in {408, 504}:
        return GPTAPIError(
            f"Vision API request timed out upstream. Increase timeout_seconds or check provider latency. Raw upstream message: {normalized_message}",
            "timeout",
            status_code,
        )

    if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return GPTAPIError(
            f"Vision API rate limit exceeded. Slow down requests or upgrade the provider quota. Raw upstream message: {normalized_message}",
            "rate_limited",
            status_code,
        )

    if status_code in {500, 502, 503}:
        return GPTAPIError(
            f"Vision API service is temporarily unavailable ({status_code}). Try again later or switch provider. Raw upstream message: {normalized_message}",
            "upstream_unavailable",
            status_code,
        )

    return GPTAPIError(
        f"Vision API request failed ({status_code}): {normalized_message}",
        "unknown_error",
        status_code,
    )


def _classify_connection_error(exc: Exception, base_url: str):
    reason = getattr(exc, "reason", exc)
    normalized_message = str(reason or exc).strip() or str(exc)
    lowered = normalized_message.lower()

    if isinstance(reason, socket.timeout) or isinstance(exc, socket.timeout) or "timed out" in lowered:
        return GPTAPIError(
            "Vision API request timed out. Increase timeout_seconds or check network/provider latency.",
            "timeout",
        )

    if (
        "unknown url type" in lowered
        or "no host given" in lowered
        or "invalid url" in lowered
        or "unsupported url" in lowered
    ):
        return GPTAPIError(
            f"Invalid Vision API URL: {base_url}. Check the base_url format.",
            "invalid_url",
        )

    if (
        "name or service not known" in lowered
        or "temporary failure in name resolution" in lowered
        or "nodename nor servname" in lowered
        or "getaddrinfo failed" in lowered
    ):
        return GPTAPIError(
            f"Could not resolve the Vision API host from base_url: {base_url}. Check the URL or DNS configuration.",
            "invalid_url",
        )

    if "connection refused" in lowered or "failed to establish a new connection" in lowered:
        return GPTAPIError(
            f"Could not connect to the Vision API service at {base_url}. Check whether the service is online and reachable.",
            "connection_failed",
        )

    if "ssl" in lowered or "certificate" in lowered:
        return GPTAPIError(
            f"TLS/SSL handshake failed while connecting to the Vision API. Check HTTPS configuration or provider certificates. Details: {normalized_message}",
            "ssl_error",
        )

    return GPTAPIError(
        f"Vision API connection failed: {normalized_message}",
        "connection_failed",
    )


def _extract_model_ids(payload):
    candidates = None
    for key in ("data", "models", "result"):
        if isinstance(payload.get(key), list):
            candidates = payload[key]
            break

    if candidates is None:
        return []

    values = []
    for item in candidates:
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
            continue

        if not isinstance(item, dict):
            continue

        for key in ("id", "name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
                break

    return sorted(set(values), key=str.lower)


def _request_json(
    *,
    url,
    headers,
    timeout_seconds,
    method="GET",
    payload=None,
    endpoint_kind="chat",
    base_url="",
):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        details = _extract_api_error_details(detail)
        raise _classify_http_error(exc.code, details["message"], base_url, endpoint_kind) from exc
    except ValueError as exc:
        raise GPTAPIError(
            f"Invalid Vision API URL: {base_url}. Check the base_url format.",
            "invalid_url",
        ) from exc
    except error.URLError as exc:
        raise _classify_connection_error(exc, base_url) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise GPTAPIError(
            "Vision API request timed out. Increase timeout_seconds or check network/provider latency.",
            "timeout",
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GPTAPIError(
            "Vision API returned a non-JSON response. Check whether base_url points to a supported provider endpoint.",
            "invalid_response",
        ) from exc


def _request_openai_json(
    *,
    url,
    api_key,
    timeout_seconds,
    method="GET",
    payload=None,
    endpoint_kind="chat",
    base_url="",
):
    return _request_json(
        url=url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _DEFAULT_HTTP_USER_AGENT,
        },
        timeout_seconds=timeout_seconds,
        method=method,
        payload=payload,
        endpoint_kind=endpoint_kind,
        base_url=base_url,
    )


def _request_gemini_json(
    *,
    url,
    api_key,
    timeout_seconds,
    method="GET",
    payload=None,
    endpoint_kind="chat",
    base_url="",
):
    return _request_json(
        url=url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "User-Agent": _DEFAULT_HTTP_USER_AGENT,
        },
        timeout_seconds=timeout_seconds,
        method=method,
        payload=payload,
        endpoint_kind=endpoint_kind,
        base_url=base_url,
    )


def _call_gpt_reverse_prompt(
    *,
    model,
    api_key,
    base_url,
    default_base_url,
    images,
    text_prompt,
    image_detail,
    max_tokens,
    temperature,
    top_p,
    timeout_seconds,
):
    user_content = [{"type": "text", "text": text_prompt}]
    for pil_image in images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _pil_to_data_url(pil_image),
                    "detail": image_detail,
                },
            }
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You analyze images for prompt reverse engineering. Output only the final English image generation prompt with no explanation or markdown.",
            },
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response_payload = _request_openai_json(
        url=_build_chat_completions_url(base_url, default_base_url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="POST",
        payload=payload,
        endpoint_kind="chat",
        base_url=base_url,
    )

    text = _extract_chat_text(response_payload)
    if not text:
        raise GPTAPIError("Vision API returned an empty response.", "empty_response")
    return text


def _fetch_gpt_models(
    *,
    api_key,
    base_url,
    default_base_url,
    timeout_seconds,
):
    payload = _request_openai_json(
        url=_build_models_url(base_url, default_base_url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="GET",
        endpoint_kind="models",
        base_url=base_url,
    )
    model_ids = _extract_model_ids(payload)
    if not model_ids:
        raise GPTAPIError(
            "The upstream /models endpoint returned no usable model IDs.",
            "models_empty",
        )
    return model_ids


def _extract_gemini_text(payload):
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""

    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    values = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            values.append(part["text"].strip())
    return "\n".join(value for value in values if value).strip()


def _call_gemini_reverse_prompt(
    *,
    model,
    api_key,
    base_url,
    default_base_url,
    images,
    text_prompt,
    max_tokens,
    temperature,
    top_p,
    timeout_seconds,
):
    parts = []
    for pil_image in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": _pil_to_data_url(pil_image).split("base64,", 1)[1],
                }
            }
        )
    parts.append({"text": text_prompt})

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }

    response_payload = _request_gemini_json(
        url=_build_gemini_generate_url(base_url, model, default_base_url, model),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="POST",
        payload=payload,
        endpoint_kind="chat",
        base_url=base_url or default_base_url,
    )

    text = _extract_gemini_text(response_payload)
    if not text:
        raise GPTAPIError("Vision API returned an empty response.", "empty_response")
    return text


def _fetch_gemini_models(
    *,
    api_key,
    base_url,
    default_base_url,
    timeout_seconds,
):
    payload = _request_gemini_json(
        url=_build_gemini_models_url(base_url, default_base_url),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="GET",
        endpoint_kind="models",
        base_url=base_url or default_base_url,
    )

    models = payload.get("models") or []
    values = []
    for item in models:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            values.append(_normalize_gemini_model_name(name.strip(), ""))

    values = [value for value in values if value]
    values = sorted(set(values), key=str.lower)
    if not values:
        raise GPTAPIError(
            "The Gemini /models endpoint returned no usable model IDs.",
            "models_empty",
        )
    return values


def _register_gpt_api_routes():
    global _GPT_API_ROUTES_REGISTERED
    if _GPT_API_ROUTES_REGISTERED or web is None or PromptServer is None:
        return

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.post("/iat/gpt/models")
    async def iat_gpt_models(request_obj):
        try:
            body = await request_obj.json()
        except Exception:
            body = {}

        provider = str((body or {}).get("provider", "OpenAI-Compatible")).strip() or "OpenAI-Compatible"
        provider_cfg = _get_provider_config(provider)
        api_key = _resolve_provider_api_key(provider, str((body or {}).get("api_key", "")))
        base_url = str((body or {}).get("base_url", "")).strip() or provider_cfg.get("default_base_url", _DEFAULT_OPENAI_BASE_URL)
        timeout_seconds = _coerce_positive_int((body or {}).get("timeout_seconds"), _coerce_positive_int(_OPENAI_CFG.get("timeout_seconds"), 60))

        if not api_key:
            env_name = provider_cfg.get("default_api_key_env", "") or _DEFAULT_OPENAI_API_KEY_ENV
            config_key = provider_cfg.get("config_key", "openai")
            return web.json_response(
                {
                    "ok": False,
                    "error": f"Missing API key. Fill the api_key field, set {config_key}.api_key in {_CFG_FILE_NAME}, or set {env_name}.",
                    "category": "missing_api_key",
                },
                status=400,
            )

        try:
            if provider_cfg.get("mode") == "gemini":
                models = await asyncio.to_thread(
                    _fetch_gemini_models,
                    api_key=api_key,
                    base_url=base_url,
                    default_base_url=provider_cfg.get("default_base_url", ""),
                    timeout_seconds=timeout_seconds,
                )
            else:
                models = await asyncio.to_thread(
                    _fetch_gpt_models,
                    api_key=api_key,
                    base_url=base_url,
                    default_base_url=provider_cfg.get("default_base_url", ""),
                    timeout_seconds=timeout_seconds,
                )
        except GPTAPIError as exc:
            status = exc.status_code or (400 if exc.category in {"invalid_url", "invalid_api_key", "invalid_request", "invalid_model", "missing_api_key"} else 502)
            return web.json_response(
                {
                    "ok": False,
                    "error": str(exc),
                    "category": exc.category,
                    "status_code": exc.status_code,
                },
                status=status,
            )
        except Exception as exc:
            return web.json_response(
                {
                    "ok": False,
                    "error": f"Unexpected model refresh failure: {exc}",
                    "category": "unknown_error",
                },
                status=500,
            )

        return web.json_response({"ok": True, "models": models})

    _GPT_API_ROUTES_REGISTERED = True


class Qwen35PromptEnhancerNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "无"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "cuda"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "SDPA"
        
        return {
            "required": {
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "enhancement_style": (list(PROMPT_STYLES.keys()), {"default": "Enhance"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def enhance_prompt(
        self,
        model_variant,
        quantization,
        device,
        attention_backend,
        prompt_text,
        enhancement_style,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
    ):
        system_prompt = (custom_system_prompt or "").strip() or PROMPT_STYLES.get(enhancement_style, PROMPT_STYLES["Enhance"])
        user_prompt = (prompt_text or "").strip() or "Describe a cinematic scene in rich visual detail."

        text = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (text,)


class Qwen35ReversePromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in VL_MODEL_CANDIDATES else list(VL_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "无"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "cuda"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "SDPA"
        
        return {
            "required": {
                "model_variant": (list(VL_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "preset_prompt": (list(REVERSE_PRESETS.keys()), {"default": "Detailed Description"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 192, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def reverse_prompt(
        self,
        model_variant,
        quantization,
        device,
        attention_backend,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        pil_images = _collect_pil_images(image, image_2, image_3, image_4)
        if len(pil_images) == 0:
            return ("[IAT] Please connect at least one IMAGE input for reverse prompt.",)

        text_prompt = (custom_prompt or "").strip() or REVERSE_PRESETS.get(preset_prompt, REVERSE_PRESETS["Detailed Description"])

        text = generate_vision_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            images=pil_images,
            text_prompt=text_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (text,)


class GPTReversePromptNode:
    DESCRIPTION = (
        "Reverse prompt through multiple vision APIs. "
        f"Supported providers: {', '.join(VISION_API_PROVIDER_OPTIONS)}. "
        f"For safer key storage, leave api_key empty and set the matching provider section in {_CFG_FILE_NAME}."
    )

    @classmethod
    def INPUT_TYPES(cls):
        _register_gpt_api_routes()
        return {
            "required": {
                "provider": (
                    VISION_API_PROVIDER_OPTIONS,
                    {
                        "default": "OpenAI-Compatible",
                        "tooltip": "Select which upstream API format to use. Qwen currently uses its OpenAI-compatible mode; Gemini uses the native generateContent API.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": _DEFAULT_OPENAI_MODEL,
                        "tooltip": "Provider-specific model name used for image reverse prompting. You can type it manually or use refresh_models to query /models and select one.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            f"Optional inline override. For better safety, leave this blank and set the matching provider's api_key in "
                            f"{_CFG_FILE_NAME}, or use that provider's environment variable."
                        ),
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": _DEFAULT_OPENAI_BASE_URL,
                        "tooltip": "Provider API base URL. For Gemini this should point to the Gemini API root; for OpenAI/Qwen-compatible providers this should point to the OpenAI-compatible /v1 root.",
                    },
                ),
                "preset_prompt": (list(REVERSE_PRESETS.keys()), {"default": "Detailed Description"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_detail": (
                    GPT_IMAGE_DETAIL_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Only used by OpenAI-compatible providers. Gemini may ignore this setting.",
                    },
                ),
                "max_tokens": ("INT", {"default": 192, "min": 32, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": _coerce_positive_int(_OPENAI_CFG.get("timeout_seconds"), 60),
                        "min": 5,
                        "max": 300,
                        "tooltip": "HTTP timeout for the upstream vision API request.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "IAT/Vision API"

    def reverse_prompt(
        self,
        provider,
        model,
        api_key,
        base_url,
        preset_prompt,
        custom_prompt,
        image_detail,
        max_tokens,
        temperature,
        top_p,
        timeout_seconds,
        image=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        pil_images = _collect_pil_images(image, image_2, image_3, image_4)
        if len(pil_images) == 0:
            return ("[IAT] Please connect at least one IMAGE input for reverse prompt.",)

        provider_cfg = _get_provider_config(provider)
        resolved_api_key = _resolve_provider_api_key(provider, api_key)
        if not resolved_api_key:
            env_hint = provider_cfg.get("default_api_key_env", "") or _DEFAULT_OPENAI_API_KEY_ENV or "OPENAI_API_KEY"
            config_key = provider_cfg.get("config_key", "openai")
            return (
                f"[IAT] Missing API key. Fill the api_key field, set {config_key}.api_key in "
                f"{_CFG_FILE_NAME}, or set {env_hint}.",
            )

        resolved_model = (model or "").strip() or provider_cfg.get("default_model", _DEFAULT_OPENAI_MODEL)
        if not resolved_model:
            return ("[IAT] Please provide a model name.",)

        resolved_base_url = (base_url or "").strip() or provider_cfg.get("default_base_url", _DEFAULT_OPENAI_BASE_URL)

        text_prompt = (custom_prompt or "").strip() or REVERSE_PRESETS.get(preset_prompt, REVERSE_PRESETS["Detailed Description"])

        try:
            if provider_cfg.get("mode") == "gemini":
                text = _call_gemini_reverse_prompt(
                    model=resolved_model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    default_base_url=provider_cfg.get("default_base_url", _DEFAULT_OPENAI_BASE_URL),
                    images=pil_images,
                    text_prompt=text_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_seconds=timeout_seconds,
                )
            else:
                text = _call_gpt_reverse_prompt(
                    model=resolved_model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    default_base_url=provider_cfg.get("default_base_url", _DEFAULT_OPENAI_BASE_URL),
                    images=pil_images,
                    text_prompt=text_prompt,
                    image_detail=image_detail,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_seconds=timeout_seconds,
                )
        except Exception as exc:
            return (f"[IAT] {exc}",)

        return (text,)


class QwenTranslatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "无"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "cuda"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "SDPA"
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Please input text to translate."}),
                "target_language": (TRANSLATION_TARGET_OPTIONS, {"default": "English"}),
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.5}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "IAT/Qwen3.5"

    def translate(self, text, target_language, model_variant, quantization, device, attention_backend, max_tokens, temperature, keep_model_loaded, seed):
        src = (text or "").strip()
        if not src:
            return ("",)

        lang = _detect_language(src)
        target_code = "en" if target_language in {"English", "英文"} else "zh"
        if lang == target_code:
            return (src,)

        target_prompt = "natural English" if target_code == "en" else "自然流畅的中文"
        response = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the user text to {target_prompt}. Return only the translation.",
                },
                {"role": "user", "content": src},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repetition_penalty=1.0,
            seed=seed,
        )

        if not keep_model_loaded:
            unload_all_models()
        return (response.strip(),)


class QwenKontextTranslatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        # 获取默认模型，如果不存在则使用第一个可用模型
        default_variant = _DEFAULT_VARIANT if _DEFAULT_VARIANT in TEXT_MODEL_CANDIDATES else list(TEXT_MODEL_CANDIDATES.keys())[0]
        default_quant = _DEFAULT_QUANT if _DEFAULT_QUANT in QUANT_OPTIONS else "无"
        default_device = _DEFAULT_DEVICE if _DEFAULT_DEVICE in DEVICE_OPTIONS else "cuda"
        default_attention_backend = DEFAULT_ATTENTION_BACKEND if DEFAULT_ATTENTION_BACKEND in ATTENTION_OPTIONS else "SDPA"
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Please input editing instruction."}),
                "model_variant": (list(TEXT_MODEL_CANDIDATES.keys()), {"default": default_variant}),
                "quantization": (QUANT_OPTIONS, {"default": default_quant}),
                "device": (DEVICE_OPTIONS, {"default": default_device}),
                "attention_backend": (ATTENTION_OPTIONS, {"default": default_attention_backend}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "IAT/Qwen3.5"

    def optimize_prompt(self, text, model_variant, quantization, device, attention_backend, max_tokens, temperature, keep_model_loaded, seed):
        src = (text or "").strip()
        if not src:
            return ("",)

        response = generate_text(
            variant=model_variant,
            quantization=quantization,
            device=device,
            attention_backend=attention_backend,
            messages=[
                {"role": "system", "content": KONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": src},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            seed=seed,
        )

        if "```" in response:
            parts = response.split("```")
            response = parts[1] if len(parts) > 1 else parts[0]

        if not keep_model_loaded:
            unload_all_models()
        return (response.strip(),)


_register_gpt_api_routes()


NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": Qwen35PromptEnhancerNode,
    "Qwen35ReversePrompt by IAT": Qwen35ReversePromptNode,
    "GPTReversePrompt by IAT": GPTReversePromptNode,
    "QwenTranslator by IAT": QwenTranslatorNode,
    "QwenKontextTranslator by IAT": QwenKontextTranslatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer by IAT": "Qwen3.5 提示词增强器（IAT）",
    "Qwen35ReversePrompt by IAT": "Qwen3.5 反推提示词（IAT）",
    "GPTReversePrompt by IAT": "Vision API 反推提示词（IAT）",
    "QwenTranslator by IAT": "Qwen 翻译器（IAT）",
    "QwenKontextTranslator by IAT": "Qwen 编辑提示词优化（IAT）",
}
