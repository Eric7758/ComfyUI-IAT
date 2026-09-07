# Usage Guide

> 中文速览（精简版）
>
> - 节点分类不变：`IAT/Qwen3.5`、`IAT/Vision API`、`IAT/Image`、`IAT/Input`。
> - Dataset 功能已拆分为 `Dataset Caption Picker` 与 `Dataset RAG Prompt Generator`。
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
- [Dataset Nodes](#dataset-nodes)
- [Vision API Reverse Prompt](#vision-api-reverse-prompt)
- [Qwen Translator](#qwen-translator)
- [Qwen Kontext Translator](#qwen-kontext-translator)
- [Best Practices](#best-practices)
- [Example Workflows](#example-workflows)

## Node Overview

ComfyUI-IAT provides 9 nodes for text and image processing:

| Node | Category | Purpose |
|------|----------|---------|
| Qwen3.5 Prompt Enhancer | IAT/Qwen3.5 | Enhance and optimize text prompts |
| Qwen3.5 Reverse Prompt | IAT/Qwen3.5 | Generate prompts from images |
| Dataset Caption Picker | IAT/Dataset | Deterministically select one original training caption |
| Dataset RAG Prompt Generator | IAT/Dataset | Retrieve paired training examples and generate one prompt |
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

## Dataset Nodes

### Purpose
Use the two Dataset nodes with an external offline dataset. Text is required;
reference image is optional and can be used for structure-aware retrieval.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| user_prompt | String | required | User requirements and hard constraints |
| dataset_name | Dropdown | first dataset | Dataset directory under `datasets.root` |
| backend | Dropdown | Ollama | `Ollama`, `vLLM`, or in-process `Local` |
| model_override | String | config default | Backend-specific model name |
| base_url_override | String | config default | Backend endpoint override |
| retrieval_seed | Int | 1 | Reproducible retrieval seed within the relevant candidate pool |
| generation_seed | Int | 1 | Reproducible backend generation seed |
| exploration_strength | Dropdown | Medium | `Mild`, `Medium`, or `Strong`; controls retrieval diversity and composition variation |
| variation_seed | Int | 1 | Reproducible CMF component/color/material combination seed |
| top_k | Int | 4 | Final MMR-selected captions, 1-8 |
| preserve_reference_color | Boolean | False | Use RGB instead of grayscale reference image |
| custom_instruction | String | "" | Extra generation instruction |
| max_tokens | Int | 512 | Maximum generated prompt length |
| temperature | Float | 0.0 | `0.0` uses the exploration mapping; positive values override it |
| top_p | Float | 1.0 | Nucleus sampling parameter |
| repetition_penalty | Float | 1.05 | Repetition penalty passed to the backend |
| timeout_seconds | Int | 300 | Backend request timeout |
| image | IMAGE | optional | Reference image 1 |
| image_2 | IMAGE | optional | Reference image 2 |
| image_3 | IMAGE | optional | Reference image 3 |
| image_4 | IMAGE | optional | Reference image 4 |
| auxiliary_color_strategy | Enum | none | Fallback auxiliary-color policy when `cmf_request_json` does not set one |
| cmf_request_json | STRING | optional | Structured CMF request; when supplied it overrides the free-text color/material fields |

`CMF Color Reference Image（IAT）` accepts the same `cmf_request_json` and
outputs an exact RGB swatch image plus resolved-color JSON. Connect the image
to the downstream color-reference path while keeping the generated prompt's
HEX/RGB text. It does not run Depth/Lineart itself; `retrieval_debug` exposes
the recommended controls, strengths, and low-denoise handoff for the image
generation workflow.

`CMF Region Color Acceptance（IAT）` takes one generated `IMAGE`, one region
`MASK`, and a target HEX. It reports masked mean RGB and CIEDE2000, so each
region can be accepted or rejected independently after generation.

### Notes

- `Dataset Caption Picker` does not load a model and uses `random.Random(seed)` for reproducible sampling.
- `Dataset RAG Prompt Generator` combines Chinese BM25/character n-gram retrieval with a local multimodal embedding index. Qwen3-VL-Embedding and Chinese CLIP are supported.
- Exploration is deterministic: the same backend, model, dataset fingerprint/version, prompt, and seeds produce the same retrieval/composition inputs. Change `retrieval_seed` to explore nearby training examples, `variation_seed` to explore CMF combinations, or `generation_seed` to vary only backend sampling.
- The default CMF request is deterministic: `temperature` sent to the backend is `0`. Set `deterministic` to `false` in `cmf_request_json` before using a non-zero temperature or exploration mapping.
- User-supplied RGB/HEX values are normalized and locked exactly; the rendered conditioning text includes the exact HEX/RGB and a factual OKLCH-derived descriptor. Decorative color names remain metadata and do not control generation.
- Colors receive stable IDs such as `C1` and `C2`; model assignments use the ID so two colors from the same family cannot overwrite each other.
- Every requested material must be assigned to at least one compatible component. If the selected material count exceeds the current component capacity, the request fails instead of silently dropping a material.
- Retrieval is material-aware: when a requested material exists in the dataset, the selected pool reserves one matching sample for it and also retains a `full_cabin` sample when `top_k` has room. `retrieval_debug.material_coverage` shows requested, reserved, used, and missing materials.
- The fixed component map includes `座椅主面料`, `座椅侧翼`, `门板内衬肌理`, `中控台面上层`, `中控台前饰板`, `马鞍区面板`, and `方向盘`.
- With a reference image, `retrieval_debug.image_conditioning` recommends Depth/Lineart structure controls, low denoise, and masked Lab/Delta-E 2000 color measurement after generation. Actual ControlNet and region masks remain downstream workflow responsibilities.
- Auxiliary colors are resolved locally before model planning. The resolved palette and automatically added colors are exposed as `resolved_colors` and `auxiliary_colors` in `retrieval_debug`.
- The final result is compiled locally through the fixed CMF template `由原状态转变为...，以...为主色调，其中，...。`; model prose, extra colors, and unselected materials are never rendered.
- Source samples are classified as `full_cabin`, `color_material`, or `detail`. The derived type is stored in the JSON index cache and SQLite chunk metadata.
- Multi-view datasets group the same filename stem across `control1`, `control2`, `control3`, and `result`; one `result/<stem>.txt` caption represents the group.
- The generator accepts up to four reference images. A batched IMAGE input is expanded into individual images and sent together to the selected backend.
- The index cache is automatically rebuilt when `dataset.json`, image files, or captions change.
- `retrieval_debug` includes the hybrid weights, candidate pool scores/tie-breakers, selected ranks, MMR profile, image counts, and index version for diagnosing relevance versus exploration.
- Local generation reuses the existing Transformers cache. Ollama uses native `/api/chat`; vLLM uses `/v1/chat/completions`.
- The default configuration is fully offline and points at local Ollama `qwen3.5:122b`.
- The node returns the final prompt, retrieved captions, retrieval scores/debug JSON, and dataset metadata.

### Retrieval performance and model reuse

- Embedding models are reused across text/image calls with different batch sizes.
  Batch size is passed per encoding call; device and output dimension remain part
  of the adapter identity. Query/document instructions are unchanged.
- Dense scores use lazily built contiguous `float32` matrices. BM25 term weights
  and postings are prepared once per index. Fusion weights, material coverage,
  and seeded MMR policy are unchanged; float32 scores may differ slightly from
  previous Python arithmetic, particularly for near-ties.
- Up to two CPU indexes are kept in an LRU cache. Source fingerprints, encoder
  settings, record metadata, and disk-cache contents are checked before reuse.
  Hashing still reads files; this avoids JSON decoding and statistics rebuilding,
  not all disk I/O. Separately decoded `.iatdb` snapshots are not conflated.
- Existing schema-v5 JSON caches and schema-v2 `.iatdb` bundles remain compatible.
  JSON cache replacement is atomic; malformed/non-finite cached vectors rebuild.
- `datasets.embedding_keep_loaded: false` remains the default so retrieval can
  release model memory before generation. Set it to `true` for repeated queries
  when the CPU/GPU has room for both embedding and generation models. It is
  independent of the bounded CPU index cache.
- This update does not change the embedding model, add a reranker, migrate vector
  formats, or alter the CMF planning strategy. Those require dataset evaluation.

Run the synthetic CPU benchmark from the plugin directory with the active Python:

```powershell
python -B scripts/benchmark_retrieval.py --entries 5000 --dimensions 1024 --repeats 5
```

It checks numerical agreement and reports median scoring times plus first-use
matrix/statistics preparation. It excludes embedding inference, dataset I/O,
MMR, and generation; its speedup is not an end-to-end workflow speedup.

### Portable SQLite datasets

Drag one dataset directory, or the dataset root for a batch build, onto
`build_iatdb.cmd`. The builder uses the local embedding settings in `config.yaml`
and writes bundles under `compiled/`. Repeating the build skips unchanged data.
Each `.iatdb` contains metadata, captions, and normalized text/RGB/grayscale
vectors only. It does not contain original images. Qwen3-VL inputs are resized
to a maximum side of 768 pixels before embedding so build and query preprocessing
remain identical on 24 GB GPUs.

### Structured CMF request

Use `cmf_request_json` for the production contract. `user_prompt` may remain
empty when this field is connected:

```json
{
  "style_id": "offroad",
  "scope": "full_cabin",
  "colors": [
    {"family": "orange", "hex": "#c96f3a", "role": "primary"},
    {"family": "gray", "hex": "#2f3f49", "role": "secondary"}
  ],
  "materials": ["suede", "fabric"],
  "preserve_geometry": true,
  "palette_policy": "balanced",
  "auxiliary_color_strategy": "style",
  "material_policy": "strict",
  "trigger_mode": "inline",
  "deterministic": true
}
```

RGB can be used instead of HEX, or both can be supplied and must agree. When
the field is omitted, the node still parses the existing short label format,
for example `越野CMF风格，紫色系（#A077AB）、黑色系、麂皮，织物`.

`auxiliary_color_strategy` supports four deterministic policies:

- `reuse_secondary`: add no new color; missing accent/neutral roles reuse the selected secondary color, then the primary color.
- `dataset`: fill missing roles only from colors found in the retrieved captions. It never falls back to style colors.
- `style`: fill missing roles from the style palette. The off-road defaults are orange/gray/brown/black and the home defaults are beige/gray/brown/black.
- `none`: add no color and synthesize no missing auxiliary role; component assignments use only explicitly selected colors. This is the production default. At least one user color is required.

An explicit value inside `cmf_request_json` overrides the node dropdown. For
legacy requests, `palette_policy: "input_only"` and `palette_policy: "strict"`
map to `none` when `auxiliary_color_strategy` is omitted. Automatically added
accent and neutral colors are rendered only in component clauses, not in the
`以...为主色调` header.

### Example Workflow

```
[Text Input] + [Load Image optional] → [Dataset RAG Prompt Generator]
                                      ↓
                              [CLIP Text Encode] → [KSampler] → [Save Image]
```

For direct caption inspection:

```text
[Dataset Caption Picker] → [Show Text]
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
