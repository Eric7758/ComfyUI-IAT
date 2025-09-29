# ComfyUI-IAT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.3.60-blue)](https://github.com/comfyanonymous/ComfyUI)

A powerful ComfyUI plugin for:
- 🌐 Multilingual translation (Chinese/Japanese → English) via Qwen
- 🎯 Prompt optimization for Flux/Kontext workflows
- 🖼️ Advanced image resizing (SDXL, match size, longest side)
- 🔢 Flexible input nodes (text, int, float, seed, base64 image)

## Features

### Input Nodes
- `Text Input by IAT` - Multiline text input
- `Float Input by IAT` - Float value input
- `Integer Input by IAT` - Integer value input
- `Seed Generator by IAT` - Random seed generator
- `Base64 to Image by IAT` - Convert Base64 string to image

### Image Nodes
- `Image Match Size by IAT` - Resize image to match reference size
- `Image Resize Longest Side by IAT` - Resize image by longest side
- `ImageResizeToSDXL by IAT` - Resize image for SDXL (1152x1152 target)
- `Image Size by IAT` - Get image dimensions

### LLM Nodes
- `Qwen Translator by IAT` - Translate Chinese/Japanese to English
- `QwenKontextTranslator by IAT` - Optimize prompts for Kontext

## Installation

### Method 1: Git Clone
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Eric7758/ComfyUI-IAT.git
cd ComfyUI-IAT
pip install -r requirements.txt
