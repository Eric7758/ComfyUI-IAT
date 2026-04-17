# Installation Guide

> 中文速览（精简版）
>
> - 推荐环境：`Python 3.10+`、已安装 `ComfyUI`、NVIDIA GPU（可选）。
> - 完整依赖见根目录 `requirements.txt`，其中包含 `torch/numpy/Pillow` 等基础依赖，适配更多 ComfyUI 环境。
> - 首次安装后请重启 ComfyUI；若提示 `qwen3_5` 架构缺失，执行一次 `python install.py` 并重启。

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- Python 3.8+
- ComfyUI installed
- 8GB RAM
- 8GB VRAM (for smaller models)

### Recommended Requirements
- Python 3.10+
- 16GB+ RAM
- 16GB+ VRAM (for 9B+ models)
- CUDA-capable GPU

## Installation Methods

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI
2. Click on "Manager" button
3. Select "Install Custom Nodes"
4. Search for "ComfyUI-IAT"
5. Click "Install"
6. Restart ComfyUI

### Method 2: Manual Installation

```bash
# Navigate to ComfyUI custom nodes directory
cd path/to/ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/Eric7758/ComfyUI-IAT.git

# Navigate to the plugin directory
cd ComfyUI-IAT

# Install dependencies
python install.py
```

### Method 3: Using Install Scripts

#### Windows
```batch
# Double-click or run in CMD
install.bat
```

#### Linux/Mac
```bash
# Make script executable and run
chmod +x install.sh
./install.sh
```

## Configuration

### Basic Configuration

Edit `config.yaml` in the plugin root directory:

```yaml
model:
  default_variant: "Qwen3.5-2B"
  quantization: "无"
  device: "cuda"

runtime:
  default_attention_backend: "SDPA"
  auto_upgrade_transformers: true
  prefer_optimized_attention: true
  enable_torch_compile: false

openai:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4.1-mini"
  api_key: ""
  api_key_env: "OPENAI_API_KEY"
  timeout_seconds: 60
  user_agent: "ComfyUI-IAT/1.22"

logging:
  verbose: false
```

For secrets, you can write the API key directly into `config.yaml`:

```yaml
openai:
  api_key: "sk-your-api-key"
```

Optional provider-specific sections:

```yaml
gemini:
  base_url: "https://generativelanguage.googleapis.com/v1beta"
  model: "gemini-2.5-flash"
  api_key: ""
  api_key_env: "GEMINI_API_KEY"

qwen_compatible:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-vl-plus"
  api_key: ""
  api_key_env: "DASHSCOPE_API_KEY"
```

### Configuration Options

#### Model Variants
- `Qwen3.5-0.8B` - Fastest, lowest quality
- `Qwen3.5-2B` - Balanced speed/quality
- `Qwen3.5-4B` - Good quality for mid-range GPUs
- `Qwen3.5-9B` - High quality
- `Qwen3.5-27B` - Best quality, requires large VRAM or aggressive quantization

#### Quantization Options
- `无` - Best quality, highest VRAM
- `8-bit` - Good quality, medium VRAM
- `4-bit` - Acceptable quality, lowest VRAM

#### Device Options
- `cuda` - Use NVIDIA GPU
- `cpu` - Use CPU (slower)

#### Vision API Reverse Prompt Configuration
- For `OpenAI-Compatible`, configure the `openai` section in `config.yaml`
- For `Gemini`, you can add a `gemini` section with `base_url`, `model`, `api_key`, and `api_key_env`
- For `Qwen OpenAI-Compatible`, you can add a `qwen_compatible` section with `base_url`, `model`, `api_key`, and `api_key_env`
- If the matching provider section and node input are both empty, the node falls back to that provider's environment variable
- Override `openai.user_agent` only if your proxy/WAF requires a custom client signature

### First Run

On first use, the plugin will automatically download required models to:
```
ComfyUI/models/LLM/
```

Download sources (in order):
1. ModelScope (default)
2. HuggingFace (fallback)

## Troubleshooting

### Common Issues

#### Issue: "Module not found" error
**Solution:**
```bash
python install.py
```

#### Issue: `model type 'qwen3_5' but Transformers does not recognize this architecture`
**Solution:**
```bash
python install.py
```
If the plugin auto-upgrades `transformers` during runtime, restart ComfyUI once and run the node again.

#### Issue: "CUDA out of memory"
**Solutions:**
1. Use smaller model variant (e.g., 0.8B or 2B)
2. Enable quantization (8-bit or 4-bit)
3. Close other GPU-intensive applications

#### Issue: Model download fails
**Solutions:**
1. Check internet connection
2. Try changing download source in config
3. Manually download from HuggingFace/ModelScope

#### Issue: Slow generation
**Solutions:**
1. Use smaller model variant
2. Enable quantization
3. Check GPU utilization
4. Reduce max_tokens parameter

### Getting Help

- [GitHub Issues](https://github.com/Eric7758/ComfyUI-IAT/issues)
- [ComfyUI Discord](https://comfy.org/discord)

## Uninstallation

To uninstall:

```bash
# Navigate to custom nodes directory
cd ComfyUI/custom_nodes

# Remove the plugin directory
rm -rf ComfyUI-IAT
```

Or simply delete the `ComfyUI-IAT` folder from your `custom_nodes` directory.
