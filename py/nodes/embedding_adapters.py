from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EmbeddingAdapterError(RuntimeError):
    """Raised when a local embedding model cannot be loaded or encoded."""


_ADAPTERS: Dict[Tuple[str, str, str, int], "EmbeddingAdapter"] = {}
_ADAPTER_LOCK = RLock()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EmbeddingAdapterError(f"Could not read embedding model config `{path}`: {exc}") from exc
    if not isinstance(value, dict):
        raise EmbeddingAdapterError(f"Embedding model config `{path}` must contain a JSON object.")
    return value


def detect_embedding_provider(model_path: str, requested: str = "auto") -> str:
    requested = (requested or "auto").strip().lower().replace("-", "_")
    aliases = {
        "chineseclip": "chinese_clip",
        "qwen3vl": "qwen3_vl",
        "qwen3_vl_embedding": "qwen3_vl",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "chinese_clip", "qwen3_vl"}:
        raise EmbeddingAdapterError(f"Unsupported embedding provider: `{requested}`")

    path = Path(model_path).expanduser()
    if not path.is_dir():
        raise EmbeddingAdapterError(f"Local embedding model does not exist: `{path}`")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise EmbeddingAdapterError(f"Local embedding model is missing config.json: `{path}`")
    model_type = str(_read_json(config_path).get("model_type") or "").lower()

    detected = ""
    if model_type == "chinese_clip":
        detected = "chinese_clip"
    elif model_type == "qwen3_vl" and (path / "config_sentence_transformers.json").is_file():
        detected = "qwen3_vl"
    if requested == "auto":
        if detected:
            return detected
        raise EmbeddingAdapterError(
            f"Could not auto-detect a supported embedding provider from `{config_path}` (model_type={model_type!r})."
        )
    if detected and detected != requested:
        raise EmbeddingAdapterError(
            f"Embedding provider `{requested}` does not match model_type `{model_type}` in `{config_path}`."
        )
    return requested


def embedding_model_signature(model_path: str, provider: str = "auto") -> str:
    path = Path(model_path).expanduser().resolve()
    resolved_provider = detect_embedding_provider(str(path), provider)
    digest = hashlib.sha256(resolved_provider.encode("utf-8"))
    metadata_names = {
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "preprocessor_config.json",
        "sentence_bert_config.json",
        "tokenizer_config.json",
    }
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path).as_posix()
        suffix = candidate.suffix.lower()
        if candidate.name in metadata_names or relative == "1_Pooling/config.json":
            digest.update(relative.encode("utf-8"))
            digest.update(candidate.read_bytes())
        elif suffix in {".safetensors", ".bin"}:
            digest.update(relative.encode("utf-8"))
            digest.update(str(candidate.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _normalized_lists(values: Any, dimension: int = 0) -> List[List[float]]:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise EmbeddingAdapterError(f"Embedding model returned an invalid shape: {array.shape}")
    if not np.isfinite(array).all():
        raise EmbeddingAdapterError("Embedding model returned non-finite values.")
    if dimension:
        if dimension > array.shape[1]:
            raise EmbeddingAdapterError(
                f"Requested embedding dimension {dimension} exceeds model output dimension {array.shape[1]}."
            )
        array = array[:, :dimension]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise EmbeddingAdapterError("Embedding model returned a zero-length vector.")
    return (array / norms).tolist()


class EmbeddingAdapter:
    provider = ""

    def __init__(self, model_path: str, device: str, batch_size: int, dimension: int = 0):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.dimension = max(0, int(dimension))

    def encode_texts(self, texts: Sequence[str], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        raise NotImplementedError

    def encode_images(self, images: Sequence[Any], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        raise NotImplementedError

    def unload(self) -> None:
        return None


class ChineseCLIPEmbeddingAdapter(EmbeddingAdapter):
    provider = "chinese_clip"

    def __init__(self, model_path: str, device: str, batch_size: int, dimension: int = 0):
        super().__init__(model_path, device, batch_size, dimension)
        try:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            self.processor = ChineseCLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.model = ChineseCLIPModel.from_pretrained(self.model_path, local_files_only=True)
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            raise EmbeddingAdapterError(
                f"Failed to load local Chinese CLIP embedding model `{self.model_path}`: {exc}"
            ) from exc

    def encode_texts(self, texts: Sequence[str], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        import torch

        vectors: List[List[float]] = []
        batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=list(texts[start : start + batch_size]), padding=True, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            vectors.extend(_normalized_lists(features.detach().float().cpu(), self.dimension))
        return vectors

    def encode_images(self, images: Sequence[Any], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        import torch

        vectors: List[List[float]] = []
        batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        for start in range(0, len(images), batch_size):
            inputs = self.processor(images=list(images[start : start + batch_size]), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            vectors.extend(_normalized_lists(features.detach().float().cpu(), self.dimension))
        return vectors

    def unload(self) -> None:
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        self.processor = None


class Qwen3VLEmbeddingAdapter(EmbeddingAdapter):
    provider = "qwen3_vl"

    def __init__(self, model_path: str, device: str, batch_size: int, dimension: int = 0):
        super().__init__(model_path, device, batch_size, dimension)
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device == "cuda":
                try:
                    from comfy import model_management

                    model_management.unload_all_models()
                    model_management.soft_empty_cache()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
            )
        except ImportError as exc:
            raise EmbeddingAdapterError(
                "Qwen3-VL-Embedding requires sentence-transformers>=5.4.0 in the active ComfyUI Python environment."
            ) from exc
        except Exception as exc:
            raise EmbeddingAdapterError(
                f"Failed to load local Qwen3-VL embedding model `{self.model_path}`: {exc}"
            ) from exc

    def _encode(self, inputs: Sequence[Any], instruction: str, batch_size: Optional[int] = None) -> List[List[float]]:
        if not inputs:
            return []
        try:
            values = self.model.encode(
                list(inputs),
                prompt=(instruction or None),
                batch_size=self.batch_size if batch_size is None else max(1, int(batch_size)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingAdapterError(f"Qwen3-VL embedding failed: {exc}") from exc
        return _normalized_lists(values, self.dimension)

    def encode_texts(self, texts: Sequence[str], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        return self._encode(list(texts), instruction, batch_size)

    def encode_images(self, images: Sequence[Any], instruction: str = "", *, batch_size: Optional[int] = None) -> List[List[float]]:
        return self._encode([{"image": image} for image in images], instruction, batch_size)

    def unload(self) -> None:
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None


def get_embedding_adapter(
    model_path: str,
    device: str,
    batch_size: int,
    provider: str = "auto",
    dimension: int = 0,
) -> EmbeddingAdapter:
    path = str(Path(model_path).expanduser().resolve())
    resolved_provider = detect_embedding_provider(path, provider)
    key = (resolved_provider, path, device, max(0, int(dimension)))
    with _ADAPTER_LOCK:
        adapter = _ADAPTERS.get(key)
        if adapter is None:
            cls = Qwen3VLEmbeddingAdapter if resolved_provider == "qwen3_vl" else ChineseCLIPEmbeddingAdapter
            adapter = cls(path, device, batch_size, dimension)
            _ADAPTERS[key] = adapter
        return adapter


def unload_embedding_adapters() -> None:
    with _ADAPTER_LOCK:
        adapters = list(_ADAPTERS.values())
        _ADAPTERS.clear()
        for adapter in adapters:
            adapter.unload()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
