# Installation Guide

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
- 16GB+ VRAM (for 7B+ models)
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
pip install -r requirements.txt
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
  default_variant: "Qwen3.5-Latest"
  quantization: "None (FP16/BF16)"
  device: "auto"

logging:
  verbose: false
```

### Configuration Options

#### Model Variants
- `Qwen3.5-0.8B` - Fastest, lowest quality
- `Qwen3.5-3B` - Balanced speed/quality
- `Qwen3.5-7B` - Good quality
- `Qwen3.5-14B` - High quality
- `Qwen3.5-32B` - Best quality
- `Qwen3.5-Latest` - Auto-select latest

#### Quantization Options
- `None (FP16/BF16)` - Best quality, highest VRAM
- `8-bit` - Good quality, medium VRAM
- `4-bit` - Acceptable quality, lowest VRAM

#### Device Options
- `auto` - Automatically select best device
- `cuda` - Use NVIDIA GPU
- `cpu` - Use CPU (slower)

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
pip install -r requirements.txt
```

#### Issue: "CUDA out of memory"
**Solutions:**
1. Use smaller model variant (e.g., 0.8B or 3B)
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
