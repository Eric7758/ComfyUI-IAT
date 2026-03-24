# ComfyUI-IAT

ComfyUI custom nodes for image/text workflows.

## Current Architecture

- Simple utility nodes are kept as-is:
  - Input helpers (`Text/Int/Float/Seed/Base64`)
  - Image helpers (`Match Size`, `Resize Longest Side`, `ResizeToSDXL`, `Image Size`)
- All LLM-related nodes are fully refactored and unified to a single Qwen3.5 runtime.

## LLM Refactor (Qwen3.5 Unified)

All LLM nodes now share one backend runtime:

- Runtime module: `py/nodes/qwen35_runtime.py`
- Node module: `py/nodes/qwen35_nodes.py`
- Legacy file `py/nodes/llm_nodes.py` is now a shim only.

### Supported LLM Nodes

- `Qwen Translator by IAT`
- `Qwen Kontext Translator by IAT`
- `Qwen3.5 Prompt Enhancer by IAT`
- `Qwen3.5 Reverse Prompt by IAT`

### Model Policy

- Unified model family: **Qwen3.5**.
- Default variant: `Qwen3.5-Latest`.
- If model is missing locally, plugin auto-downloads to:
  - `ComfyUI/models/LLM`
- Download source fallback order:
  1. ModelScope
  2. HuggingFace

## Config

`config.yaml`:

```yaml
model:
  default_variant: "Qwen3.5-Latest"
  quantization: "None (FP16/BF16)"
  device: "auto"
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Notes

- Requires `transformers>=4.57.0`.
- Quantization `4-bit/8-bit` requires `bitsandbytes` in your runtime.
