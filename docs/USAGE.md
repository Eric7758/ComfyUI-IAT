# Usage Guide

> 中文速览（精简版）
>
> - 节点分类不变：`IAT/Qwen3.5`、`IAT/Vision API`、`IAT/Image`、`IAT/Input`。
> - 本次整理仅做文件与说明精简，不改节点输入输出与执行逻辑。
> - 反推接口优先级：节点输入 > `config.yaml` 对应 provider > 环境变量。
> - 文本与视觉节点仅支持官方原版 `model_variant`。
> - 模型下载目录固定为 `ComfyUI/models/diffusion_models`。
> - 下载源顺序固定为 `ModelScope -> HuggingFace`。
> - `runtime.offline_only: true` 时会完全跳过下载；本地模型即使校验不完整，也会继续尝试加载，直到实际加载时报错。

## Table of Contents
- [Node Overview](#node-overview)
- [Image Color Palette Extractor](#image-color-palette-extractor)
- [Model Variants](#model-variants)
- [Qwen3.5 Prompt Enhancer](#qwen35-prompt-enhancer)
- [Qwen3.5 Reverse Prompt](#qwen35-reverse-prompt)
- [Vision API Reverse Prompt](#vision-api-reverse-prompt)
- [Qwen Translator](#qwen-translator)
- [Qwen Kontext Translator](#qwen-kontext-translator)
- [Best Practices](#best-practices)
- [Example Workflows](#example-workflows)

## Node Overview

ComfyUI-IAT provides 6 nodes for text and image processing:

| Node | Category | Purpose |
|------|----------|---------|
| Qwen3.5 Prompt Enhancer | IAT/Qwen3.5 | Enhance and optimize text prompts |
| Qwen3.5 Reverse Prompt | IAT/Qwen3.5 | Generate prompts from images |
| Vision API Reverse Prompt | IAT/Vision API | Generate prompts from images via OpenAI-compatible APIs, Gemini, and Qwen-compatible providers |
| Qwen Translator | IAT/Qwen3.5 | Translate text to English |
| Qwen Kontext Translator | IAT/Qwen3.5 | Optimize editing instructions |
| Image Color Palette Extractor | IAT/Image | Extract dominant colors and render a ratio-based palette chart |

## Image Color Palette Extractor

### Purpose
Extract dominant colors from one or more input images and output a vertical palette chart whose bar widths follow color ratio.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image | IMAGE | required | Input image or image batch |
| num_colors | Int | 6 | Number of dominant colors to extract (2-20) |
| output_width | Int | 1000 | Output palette width |
| output_height | Int | 400 | Output palette height |
| min_ratio | Float | 0.01 | Ignore colors whose ratio is below this threshold |
| sort_order | Dropdown | ratio_desc | `ratio_desc` / `ratio_asc` / `lightness` |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| palette_image | IMAGE | Palette bar image (batch-aware) |
| color_info | STRING | Color details with HEX/RGB/ratio; batched outputs are separated by `Image N` blocks |

### Example Workflow

```
Load Image → Image Color Palette Extractor by IAT → Save Image
                                  ↓
                              Show Text
```

## Model Variants

Available `model_variant` values:
- `Qwen3.5-0.8B`
- `Qwen3.5-2B`
- `Qwen3.5-4B`
- `Qwen3.5-9B`
- `Qwen3.5-27B`
- `Qwen3.6-35B-A3B`

Runtime behavior:
- Backend: Transformers (official model path only)
- Download path: `ComfyUI/models/diffusion_models`
- Download order: ModelScope first, HuggingFace fallback

## Qwen3.5 Prompt Enhancer

### Purpose
Transform simple prompts into detailed, professional-grade image generation prompts.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_variant | Dropdown | Official model list | Select official model variant |
| device | Dropdown | cuda | Computing device |
| prompt_text | String | "" | Input prompt to enhance |
| enhancement_style | Dropdown | Enhance | Enhancement style |
| custom_system_prompt | String | "" | Custom system prompt |
| max_tokens | Int | 256 | Maximum output length |
| temperature | Float | 0.7 | Creativity (0.0-1.5) |
| top_p | Float | 0.9 | Nucleus sampling |
| repetition_penalty | Float | 1.1 | Repetition penalty |
| keep_model_loaded | Boolean | True | Keep model in memory |
| seed | Int | 1 | Random seed |

### Enhancement Styles

1. **Enhance** - Expand with vivid details
   ```
   Input: "a girl in forest"
   Output: "A young woman standing in a mystical forest, dappled sunlight 
   filtering through ancient oak trees, wearing a flowing emerald dress..."
   ```

2. **Refine** - Clear and concise
   ```
   Input: "make a picture of a cat sitting on a mat"
   Output: "A domestic cat sitting on a woven mat, front view, soft lighting"
   ```

3. **Creative Rewrite** - Stronger visual storytelling
   ```
   Input: "sunset over ocean"
   Output: "Golden hour masterpiece: fiery orange and purple clouds reflect 
   on mirror-calm ocean waters, distant silhouette of a lone sailboat..."
   ```

4. **Detailed Visual** - Highly detailed description
   ```
   Input: "cyberpunk city"
   Output: "Futuristic cyberpunk metropolis at night, towering neon-lit 
   skyscrapers with holographic advertisements, flying vehicles between 
   buildings, wet streets reflecting colorful lights..."
   ```

### Example Workflow

```
[Text Input] → [Qwen3.5 Prompt Enhancer] → [CLIP Text Encode] → [KSampler]
                    ↓
            [Show Text]
```

## Qwen3.5 Reverse Prompt

### Purpose
Generate text prompts from input images using vision-language models.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_variant | Dropdown | Official model list | Select official VL variant |
| device | Dropdown | cuda | Computing device |
| preset_prompt | Dropdown | Detailed Description | Analysis style |
| custom_prompt | String | "" | Custom analysis prompt |
| max_tokens | Int | 192 | Maximum output length |
| temperature | Float | 0.0 | Creativity (0.0-1.5) |
| top_p | Float | 0.9 | Nucleus sampling |
| repetition_penalty | Float | 1.1 | Repetition penalty |
| keep_model_loaded | Boolean | True | Keep model in memory |
| seed | Int | 1 | Random seed |
| image | IMAGE | optional | Primary image |
| image_2 | IMAGE | optional | Second image |
| image_3 | IMAGE | optional | Third image |
| image_4 | IMAGE | optional | Fourth image |

### Preset Prompts

1. **Detailed Description** - Comprehensive image analysis
   - Describes all elements, composition, lighting, style
   - Best for understanding complex images

2. **Prompt Reverse** - Compact generation prompt
   - Outputs prompt suitable for image generation
   - Optimized for reuse in generation workflows

3. **Style Focus** - Style and technique analysis
   - Focuses on artistic style, camera settings, lighting
   - Best for learning and reproducing styles

### Example Workflow

```
[Load Image] → [Qwen3.5 Reverse Prompt] → [Show Text]
                    ↓
            [CLIP Text Encode] → [KSampler] → [Save Image]
```

## Vision API Reverse Prompt

### Purpose
Generate text prompts from input images using multiple vision APIs.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| provider | Dropdown | OpenAI-Compatible | Upstream API provider |
| model | String | gpt-4.1-mini | Provider-specific model name |
| api_key | String | "" | API key override; leave empty to use the matching provider config/env |
| base_url | String | https://api.openai.com/v1 | Provider API base URL |
| preset_prompt | Dropdown | Detailed Description | Analysis style |
| custom_prompt | String | "" | Custom analysis prompt |
| image_detail | Dropdown | auto | Vision detail level |
| max_tokens | Int | 192 | Maximum output length |
| temperature | Float | 0.2 | Creativity (0.0-2.0) |
| top_p | Float | 1.0 | Nucleus sampling |
| timeout_seconds | Int | 60 | HTTP timeout |
| image | IMAGE | optional | Primary image |
| image_2 | IMAGE | optional | Second image |
| image_3 | IMAGE | optional | Third image |
| image_4 | IMAGE | optional | Fourth image |

### Features

- **Multi-provider** - Supports `OpenAI-Compatible`, `Gemini`, and `Qwen OpenAI-Compatible`
- **Preset parity** - Reuses the same reverse prompt presets as the local Qwen node
- **Flexible auth** - Resolution order is node input -> matching provider section in `config.yaml` -> that provider's environment variable
- **Multi-image input** - Supports up to 4 input images
- **Actionable errors** - Returns clearer reasons for invalid API key, insufficient balance, invalid URL, timeout, rate limiting, and upstream failures
- **Model refresh** - Use `refresh_models` to query `/models` with the current `api_key` and `base_url`, then select from `available_models`

### Example Workflow

```
[Load Image] → [Vision API Reverse Prompt] → [Show Text]
                    ↓
            [CLIP Text Encode] → [KSampler] → [Save Image]
```

## Qwen Translator

### Purpose
Automatically translate Chinese or Japanese text to natural English.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | String | "" | Text to translate |
| model_variant | Dropdown | Official model list | Select official model variant |
| device | Dropdown | cuda | Computing device |
| max_tokens | Int | 512 | Maximum output length |
| temperature | Float | 0.1 | Low for accuracy |
| keep_model_loaded | Boolean | True | Keep model in memory |
| seed | Int | 1 | Random seed |

### Features

- **Auto-detection** - Automatically detects Chinese or Japanese
- **English pass-through** - Returns English input unchanged
- **Optimized for prompts** - Translation tailored for image generation

### Example

```
Input:  "一个穿着红色汉服的中国女孩"
Output: "A Chinese girl wearing traditional red Hanfu"

Input:  "美しい日本の庭園"
Output: "A beautiful Japanese garden"
```

### Example Workflow

```
[Text Input (Chinese)] → [Qwen Translator] → [CLIP Text Encode] → [KSampler]
```

## Qwen Kontext Translator

### Purpose
Optimize editing instructions for image editing models (especially Kontext-based).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | String | "" | Editing instruction |
| model_variant | Dropdown | Official model list | Select official model variant |
| device | Dropdown | cuda | Computing device |
| max_tokens | Int | 512 | Maximum output length |
| temperature | Float | 0.0 | Low for consistency |
| keep_model_loaded | Boolean | True | Keep model in memory |
| seed | Int | 1 | Random seed |

### Features

- **Instruction optimization** - Converts vague instructions to precise prompts
- **Consistency preservation** - Maintains identity when required
- **Explicit output** - Clean, editable English prompts

### Example

```
Input:  "把背景换成森林"
Output: "Replace the background with a lush forest scene, maintain the 
subject's position and lighting consistency"

Input:  "add more flowers"
Output: "Add colorful wildflowers in the foreground and midground, 
natural distribution, complementary colors"
```

### Example Workflow

```
[Load Image] → [Qwen Kontext Translator] → [Kontext Edit Model] → [Save Image]
      ↑
[Text Input]
```

## Best Practices

### Model Selection

| VRAM Available | Recommended model_variant |
|----------------|---------------------------|
| 8GB | `Qwen3.5-0.8B` |
| 12GB | `Qwen3.5-2B` |
| 16GB+ | `Qwen3.5-4B` / `Qwen3.5-9B` |
| 24GB+ | `Qwen3.5-9B` / `Qwen3.5-27B` |

### Temperature Guidelines

| Use Case | Temperature | Reason |
|----------|-------------|--------|
| Translation | 0.0-0.2 | Accuracy is key |
| Prompt Enhancement | 0.6-0.8 | Balance creativity |
| Creative Writing | 0.8-1.2 | More variation |
| Reverse Prompt | 0.0-0.3 | Factual description |

### Memory Management

- Use `keep_model_loaded = True` when processing multiple items
- Set to `False` to free VRAM between uses
- Consider using smaller models for batch processing

### Workflow Tips

1. **Chain nodes together** - Use translator → enhancer for non-English prompts
2. **Compare styles** - Try different enhancement styles to find best results
3. **Save outputs** - Use Show Text node to save good prompts for reuse
4. **Batch processing** - Process multiple images with same settings

## Example Workflows

### Workflow 1: Multi-language Prompt Enhancement

```
[Text Input (Chinese)] 
         ↓
[Qwen Translator] → [Show Text: Original EN]
         ↓
[Qwen3.5 Prompt Enhancer] → [Show Text: Enhanced]
         ↓
[CLIP Text Encode] → [KSampler] → [Save Image]
```

### Workflow 2: Image Analysis and Recreation

```
[Load Image]
         ↓
[Qwen3.5 Reverse Prompt] → [Show Text: Prompt]
         ↓
[CLIP Text Encode] → [KSampler] → [Save Image]
```

### Workflow 3: Style Transfer with Editing

```
[Load Image A] [Load Image B]
         ↓              ↓
[Qwen3.5 Reverse Prompt] (Style Focus)
         ↓
[Combine with editing instruction]
         ↓
[Qwen Kontext Translator]
         ↓
[Kontext Edit Model] → [Save Image]
```

### Workflow 4: Batch Prompt Enhancement

```
[Text List] → [Qwen3.5 Prompt Enhancer] → [Text List Output]
                    ↓
            [Iterate] → [KSampler] → [Save Image]
```

## Performance Tips

1. **First load is slow** - Model downloads and loads on first use
2. **Subsequent uses are fast** - Keep models loaded when possible
3. **Use a smaller official variant** - Choose a model size that matches your VRAM
4. **Batch when possible** - Process multiple items in one session
5. **Monitor VRAM** - Use system monitor to track memory usage
