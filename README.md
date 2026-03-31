<div align="center">

# 🎨 ComfyUI-IAT

**Image & Text utilities with Qwen3.5 translator and prompt optimizer for ComfyUI**

[![GitHub stars](https://img.shields.io/github/stars/Eric7758/ComfyUI-IAT?style=flat-square)](https://github.com/Eric7758/ComfyUI-IAT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Eric7758/ComfyUI-IAT?style=flat-square)](https://github.com/Eric7758/ComfyUI-IAT/network)
[![License](https://img.shields.io/github/license/Eric7758/ComfyUI-IAT?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Plugin-green.svg?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)

[English](#english) | [中文](#中文)

<img src="docs/images/demo.png" alt="ComfyUI-IAT Demo" width="800"/>

</div>

---

## <a name="english"></a> 🇺🇸 English

### ✨ Features

ComfyUI-IAT provides powerful AI-driven text and image processing nodes for ComfyUI workflows:

| Node | Function | Description |
|------|----------|-------------|
| 📝 **Qwen3.5 Prompt Enhancer** | Prompt Optimization | Enhance your prompts with vivid details and professional quality |
| 🔍 **Qwen3.5 Reverse Prompt** | Image-to-Text | Generate prompts from images using vision-language models |
| 🌐 **Qwen Translator** | Translation | Translate Chinese/Japanese to natural English |
| ✏️ **Qwen Kontext Translator** | Editing Optimization | Optimize editing instructions for image editing models |

### 🚀 Quick Start

#### Installation

**Method 1: Using ComfyUI Manager (Recommended)**
1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "ComfyUI-IAT"
4. Click Install

**Method 2: Manual Installation**

```bash
# Navigate to ComfyUI custom nodes directory
cd ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/Eric7758/ComfyUI-IAT.git

# Install dependencies
cd ComfyUI-IAT
pip install -r requirements.txt
```

**Method 3: Using Install Scripts**

```bash
# Windows
install.bat

# Linux/Mac
bash install.sh
```

#### Configuration

Edit `config.yaml` to customize default settings:

```yaml
model:
  default_variant: "Qwen3.5-Latest"  # Default model variant
  quantization: "None (FP16/BF16)"   # Quantization mode
  device: "auto"                      # Device: auto, cuda, cpu

logging:
  verbose: false                      # Enable verbose logging
```

### 📖 Usage

#### 1. Prompt Enhancement

Use **Qwen3.5 Prompt Enhancer** to improve your prompts:

```
Input: "a girl in forest"
Output: "A young woman standing in a mystical forest, dappled sunlight 
filtering through ancient oak trees, wearing a flowing emerald dress, 
ethereal atmosphere with floating dust particles..."
```

**Enhancement Styles:**
- **Enhance** - Expand with vivid details
- **Refine** - Clear and concise
- **Creative Rewrite** - Stronger visual storytelling
- **Detailed Visual** - Highly detailed description

#### 2. Reverse Prompt (Image-to-Text)

Use **Qwen3.5 Reverse Prompt** to generate prompts from images:

- Upload 1-4 images
- Choose preset or custom prompt
- Get detailed generation prompts

**Presets:**
- **Detailed Description** - Full image description
- **Prompt Reverse** - Compact generation prompt
- **Style Focus** - Style and composition analysis

#### 3. Translation

Use **Qwen Translator** for automatic translation:

- Auto-detects Chinese/Japanese
- Translates to natural English
- Optimized for image generation prompts

#### 4. Editing Optimization

Use **Qwen Kontext Translator** for image editing:

- Optimizes editing instructions
- Produces clean, editable prompts
- Perfect for Kontext-based editing models

### 🛠️ Requirements

- Python 3.8+
- ComfyUI
- PyTorch 2.0+
- Transformers 4.57.0+
- 8GB+ VRAM (16GB+ recommended for larger models)

### 📦 Model Support

**Qwen3.5 - Native Multimodal Models for Consumer GPUs**

All Qwen3.5 models support **text + image + video** input with **text** output:

**Small Models (4-8GB VRAM):**
| Model | Size | VRAM (FP16) | VRAM (4-bit) | Best For |
|-------|------|-------------|--------------|----------|
| Qwen3.5-0.8B | ~0.9B | ~2GB | ~1GB | Fast inference, low memory |
| Qwen3.5-2B | ~2B | ~5GB | ~2GB | Balanced speed/quality |
| Qwen3.5-4B | ~5B | ~10GB | ~3GB | Good quality on mid-range GPUs |

**Medium Models (8-16GB VRAM):**
| Model | Size | VRAM (FP16) | VRAM (4-bit) | Best For |
|-------|------|-------------|--------------|----------|
| Qwen3.5-9B | ~10B | ~20GB | ~6GB | High quality, recommended |

**Large Models (16GB+ VRAM or GGUF):**
| Model | Size | VRAM (FP16) | VRAM (4-bit) | Best For |
|-------|------|-------------|--------------|----------|
| Qwen3.5-27B | ~28B | ~56GB | ~16GB | Best quality, use GGUF for consumer GPUs |

**Quantization Options:**
- None (FP16/BF16) - Best quality
- 8-bit - Reduced memory
- 4-bit - Minimum memory

**GGUF Support:** For larger models (9B, 27B) on consumer GPUs, use GGUF quantized versions from `bartowski/Qwen_Qwen3.5-*-GGUF` or `unsloth/Qwen3.5-*-GGUF`.

### 📝 Changelog

See [UPDATE.md](UPDATE.md) for detailed changelog.

### 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## <a name="中文"></a> 🇨🇳 中文

### ✨ 功能特性

ComfyUI-IAT 为 ComfyUI 工作流提供强大的 AI 驱动的文本和图像处理节点：

| 节点 | 功能 | 描述 |
|------|------|------|
| 📝 **Qwen3.5 提示词增强器** | 提示词优化 | 用生动的细节和专业质量增强提示词 |
| 🔍 **Qwen3.5 反推提示词** | 图像转文本 | 使用视觉语言模型从图像生成提示词 |
| 🌐 **Qwen 翻译器** | 翻译 | 将中文/日文翻译成自然流畅的英文 |
| ✏️ **Qwen 编辑提示词优化** | 编辑优化 | 为图像编辑模型优化编辑指令 |

### 🚀 快速开始

#### 安装

**方法 1：使用 ComfyUI Manager（推荐）**
1. 打开 ComfyUI Manager
2. 点击 "Install Custom Nodes"
3. 搜索 "ComfyUI-IAT"
4. 点击安装

**方法 2：手动安装**

```bash
# 进入 ComfyUI 自定义节点目录
cd ComfyUI/custom_nodes

# 克隆仓库
git clone https://github.com/Eric7758/ComfyUI-IAT.git

# 安装依赖
cd ComfyUI-IAT
pip install -r requirements.txt
```

**方法 3：使用安装脚本**

```bash
# Windows
install.bat

# Linux/Mac
bash install.sh
```

#### 配置

编辑 `config.yaml` 自定义默认设置：

```yaml
model:
  default_variant: "Qwen3.5-Latest"  # 默认模型版本
  quantization: "None (FP16/BF16)"   # 量化模式
  device: "auto"                      # 设备: auto, cuda, cpu

logging:
  verbose: false                      # 启用详细日志
```

### 📖 使用方法

#### 1. 提示词增强

使用 **Qwen3.5 提示词增强器** 改进提示词：

```
输入: "森林中的女孩"
输出: "一位年轻女子站在神秘的森林中，斑驳的阳光透过古老的橡树，
身着飘逸的翡翠色长裙，空灵的氛围中漂浮着尘埃颗粒..."
```

**增强风格：**
- **Enhance** - 扩展生动细节
- **Refine** - 清晰简洁
- **Creative Rewrite** - 更强的视觉叙事
- **Detailed Visual** - 高度详细描述

#### 2. 反推提示词（图像转文本）

使用 **Qwen3.5 反推提示词** 从图像生成提示词：

- 上传 1-4 张图像
- 选择预设或自定义提示词
- 获取详细的生成提示词

**预设选项：**
- **Detailed Description** - 完整图像描述
- **Prompt Reverse** - 紧凑生成提示词
- **Style Focus** - 风格和构图分析

#### 3. 翻译

使用 **Qwen 翻译器** 进行自动翻译：

- 自动检测中文/日文
- 翻译成自然流畅的英文
- 针对图像生成提示词优化

#### 4. 编辑优化

使用 **Qwen 编辑提示词优化** 进行图像编辑：

- 优化编辑指令
- 生成清晰、可编辑的提示词
- 完美适配基于 Kontext 的编辑模型

### 🛠️ 系统要求

- Python 3.8+
- ComfyUI
- PyTorch 2.0+
- Transformers 4.57.0+
- 8GB+ 显存（大模型建议 16GB+）

### 📦 模型支持

**Qwen3.5 - 原生多模态模型**

所有 Qwen3.5 模型都支持**文本 + 图像 + 视频**输入，**文本**输出：

**Dense 模型（标准）：**
- Qwen3.5-0.8B（约0.9B参数）
- Qwen3.5-2B（约2B参数）
- Qwen3.5-4B（约5B参数）
- Qwen3.5-9B（约10B参数）
- Qwen3.5-27B（约28B参数）

**MoE 模型（混合专家）：**
- Qwen3.5-35B-A3B（35B总参数，3B激活）
- Qwen3.5-122B-A10B（122B总参数，10B激活）
- Qwen3.5-397B-A17B（397B总参数，17B激活）

**特殊版本：**
- Qwen3.5-Latest（自动选择最佳可用）
- Base 模型（预训练版）
- FP8 量化模型
- GPTQ-Int4 量化模型

**注意：** 与 Qwen2.5（有单独的文本和VL版本）不同，所有 Qwen3.5 模型都是原生多模态，采用统一的视觉-语言架构。

**量化选项：**
- None (FP16/BF16) - 最佳质量
- 8-bit - 减少内存
- 4-bit - 最小内存

### 📝 更新日志

查看 [UPDATE.md](UPDATE.md) 获取详细更新日志。

### 🤝 贡献

欢迎贡献！请随时提交问题和拉取请求。

### 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件。

---

## 🙏 Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The powerful node-based UI for Stable Diffusion
- [Qwen](https://github.com/QwenLM/Qwen) - Alibaba's large language model series
- [ModelScope](https://www.modelscope.cn/) - Model hosting platform

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

</div>
