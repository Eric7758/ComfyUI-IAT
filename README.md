# ComfyUI-IAT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.3.60-blue)](https://github.com/comfyanonymous/ComfyUI)

ComfyUI custom nodes for image/text workflows:
- Multilingual translation (Chinese/Japanese -> English) via Qwen
- Prompt optimization for Flux/Kontext workflows
- Image utilities (SDXL resize, match size, longest side)
- Input helper nodes (text, int, float, seed, base64 image)

## Project Structure

- Entry: `__init__.py`
- Node implementations: `py/nodes/*.py`
- Config: `config.yaml`
- Dependency installers: `install.bat`, `install.sh`, `install.py`
- Registry metadata: `pyproject.toml`

## Nodes

### Input
- `Text Input by IAT`
- `Float Input by IAT`
- `Integer Input by IAT`
- `Seed Generator by IAT`
- `Base64 to Image by IAT`

### Image
- `Image Match Size by IAT`
- `Image Resize Longest Side by IAT`
- `ImageResizeToSDXL by IAT`
- `Image Size by IAT`

### LLM
- `Qwen Translator by IAT`
- `QwenKontextTranslator by IAT`

## Installation

### Method 1: ComfyUI-Manager
Install from the manager, or add this repo as a custom node.

### Method 2: Git clone

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Eric7758/ComfyUI-IAT.git
cd ComfyUI-IAT
```

Install dependencies (pick one):

```bat
install.bat
```

```bash
bash install.sh
```

```bash
python -m pip install -r requirements.txt
```

## Qwen Model Config

Edit `config.yaml`:

- `model.qwen_path` default: `models/LLM/Qwen3.5-4B`
- `model.qwen_path` supports:
  - Absolute paths, e.g. `D:/models/Qwen/...`
  - Paths relative to ComfyUI root, e.g. `models/LLM/Qwen3.5-4B`
  - Paths relative to this plugin root, e.g. `../../models/LLM/...`
- `model.device`: `cuda`, `cpu`, or `auto`
- `logging.verbose`: `true` to print per-module load logs

## Notes

- Implementation code is placed under `py/` to avoid collisions with ComfyUI's top-level `nodes.py` module.
- Chat template calls disable thinking mode when tokenizer supports it, to keep node output clean.