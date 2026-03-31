# Qwen3.5 Models Skill

## Overview

Qwen3.5 is Alibaba's latest generation of natively multimodal language models. Unlike previous versions, all Qwen3.5 models support text, image, and video inputs with text output.

## Model Architecture

### Key Features
- **Native Multimodal**: All models support text + image + video input
- **Gated DeltaNet + Gated Attention**: Hybrid architecture for efficiency
- **Early Fusion**: Multimodal tokens trained together from the start
- **Extended Context**: 262K tokens native, up to 1M with YaRN
- **Global Language Support**: 201 languages and dialects

### Architecture Details
```
Input: Text / Image / Video
  ↓
Vision Encoder (for images/video)
  ↓
Gated DeltaNet (32 heads) + Gated Attention (16 heads)
  ↓
FFN (12,288 intermediate dim)
  ↓
Output: Text (with optional <think>...</think> reasoning)
```

## Available Models

### Dense Models (Consumer GPU Friendly)

| Model | Parameters | VRAM (FP16) | VRAM (4-bit) | Use Case |
|-------|-----------|-------------|--------------|----------|
| Qwen3.5-0.8B | ~0.9B | ~2GB | ~1GB | Edge devices, fast inference |
| Qwen3.5-2B | ~2B | ~5GB | ~2GB | Balanced performance |
| Qwen3.5-4B | ~5B | ~10GB | ~3GB | Mid-range GPUs |
| Qwen3.5-9B | ~10B | ~20GB | ~6GB | **Recommended for most users** |
| Qwen3.5-27B | ~28B | ~56GB | ~16GB | High quality (use GGUF) |

### GGUF Quantized Versions

For consumer GPUs, use GGUF quantized models:

**Recommended GGUF Formats:**
- `Q4_K_M` - Default choice, good balance (5.7GB for 9B model)
- `Q5_K_M` - Higher quality (6.6GB for 9B model)
- `Q6_K` - Near-original quality (7.5GB for 9B model)
- `UD-Q4_K_XL` - Unsloth dynamic, superior accuracy

**Sources:**
- `bartowski/Qwen_Qwen3.5-{size}-GGUF`
- `unsloth/Qwen3.5-{size}-GGUF`

**VRAM Rule of Thumb:**
```
Model File Size < GPU VRAM - 2GB (for overhead)
```

## Usage in ComfyUI-IAT

### Text Generation Nodes
- **Qwen3.5 Prompt Enhancer**: Improves image generation prompts
- **Qwen Translator**: Translates CN/JP to EN
- **Qwen Kontext Translator**: Optimizes editing instructions

### Vision-Language Nodes
- **Qwen3.5 Reverse Prompt**: Generates prompts from images

### Configuration

```yaml
# config.yaml
model:
  default_variant: "Qwen3.5-9B"  # For 12GB+ VRAM
  # default_variant: "Qwen3.5-4B"  # For 8GB VRAM
  # default_variant: "Qwen3.5-2B"  # For 6GB VRAM
  quantization: "4-bit"  # Use for larger models
  device: "auto"
```

## Model Selection Guide

### By VRAM

| Your VRAM | Recommended Model | Quantization |
|-----------|------------------|--------------|
| 4-6 GB | Qwen3.5-2B | FP16 |
| 6-8 GB | Qwen3.5-4B | 4-bit |
| 8-12 GB | Qwen3.5-9B | 4-bit |
| 12-16 GB | Qwen3.5-9B | 8-bit or FP16 |
| 16-24 GB | Qwen3.5-27B | 4-bit (GGUF) |
| 24GB+ | Qwen3.5-27B | 8-bit or FP16 |

### By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Fast prototyping | Qwen3.5-4B | Quick iteration |
| Production quality | Qwen3.5-9B | Best balance |
| Maximum quality | Qwen3.5-27B + GGUF | Best results |
| Low memory | Qwen3.5-2B | Minimal VRAM |

## Thinking Mode

Qwen3.5 models default to thinking mode:

```
<think>
The user wants me to enhance this prompt. I should add details about...
</think>

Final enhanced prompt here...
```

To disable (for faster inference):
```python
chat_template_kwargs = {"enable_thinking": False}
```

## Performance Tips

1. **Use 4-bit quantization** for models > 4B parameters
2. **Enable Flash Attention 2** for faster inference
3. **Use GGUF** for 9B and 27B models on consumer GPUs
4. **Keep model loaded** when processing multiple items
5. **Match model size to VRAM**: Don't use 27B on 8GB GPU

## References

- [Hugging Face Collection](https://huggingface.co/collections/Qwen/qwen35)
- [GGUF Versions](https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF)
- [Official Documentation](https://qwen.readthedocs.io/)

## Version History

- **v1.1.0** - Updated to consumer GPU focus, removed MoE models, added GGUF support info
