# Example Workflows

This directory contains example ComfyUI workflows demonstrating various use cases of ComfyUI-IAT.

## Available Workflows

### 1. Prompt Enhancement (`workflow_prompt_enhancement.json`)

**Purpose**: Demonstrates basic prompt enhancement functionality.

**Workflow**:
```
Text Input → Qwen3.5 Prompt Enhancer → Show Text
                    ↓
            CLIP Text Encode → KSampler → Save Image
```

**Features**:
- Simple text input enhancement
- Multiple enhancement styles
- Direct integration with image generation

**How to use**:
1. Load the workflow in ComfyUI
2. Enter your simple prompt in the text field
3. Select enhancement style
4. Run the workflow
5. View the enhanced prompt and generated image

---

### 2. Translation + Enhancement (`workflow_translation_enhancement.json`)

**Purpose**: Shows how to combine translation and prompt enhancement for non-English inputs.

**Workflow**:
```
Chinese Text → Qwen Translator → Show Text (English)
                        ↓
            Qwen3.5 Prompt Enhancer → Show Text (Enhanced)
                        ↓
                CLIP Text Encode → KSampler → Save Image
```

**Features**:
- Automatic Chinese to English translation
- Enhanced prompt generation from translated text
- Complete pipeline from non-English to image

**How to use**:
1. Load the workflow
2. Enter Chinese text in the translator node
3. The workflow automatically translates and enhances
4. Generates image from the enhanced English prompt

---

### 3. Reverse Prompt (`workflow_reverse_prompt.json`)

**Purpose**: Demonstrates image-to-prompt functionality.

**Workflow**:
```
Load Image → Qwen3.5 Reverse Prompt → Show Text (Generated Prompt)
                        ↓
                CLIP Text Encode → KSampler → Save Image
```

**Features**:
- Analyze existing images
- Generate prompts that could recreate the image
- Compare original with recreated image

**How to use**:
1. Load the workflow
2. Select an image using the Load Image node
3. Choose analysis preset (Detailed Description, Prompt Reverse, or Style Focus)
4. Run to get the generated prompt
5. Use the prompt to generate a new image

---

## Loading Workflows

### Method 1: Drag and Drop
1. Open ComfyUI in your browser
2. Drag the JSON file onto the canvas
3. The workflow will load automatically

### Method 2: Load Button
1. Click "Load" in the ComfyUI interface
2. Select the JSON file
3. Click "Open"

### Method 3: Copy-Paste
1. Open the JSON file in a text editor
2. Copy all content
3. In ComfyUI, press Ctrl+V (or Cmd+V on Mac)

## Customizing Workflows

### Changing Models
- Double-click on any IAT node
- Select different model variant from dropdown
- Smaller models (0.8B, 3B) are faster but less capable
- Larger models (7B, 14B) provide better quality but require more VRAM

### Adjusting Parameters

**Temperature**:
- Lower (0.0-0.3): More deterministic, consistent output
- Higher (0.7-1.0): More creative, varied output

**Max Tokens**:
- Increase for longer prompts
- Decrease for shorter, more concise output

**Quantization**:
- None: Best quality, highest VRAM usage
- 8-bit: Good balance
- 4-bit: Lowest VRAM, slightly reduced quality

### Adding More Nodes

You can extend these workflows by:
- Adding multiple enhancement nodes with different styles
- Chaining translator → enhancer → another enhancer
- Using the output for ControlNet conditioning
- Saving prompts to file for later use

## Tips and Tricks

### Memory Management
- Use `keep_model_loaded = True` when processing multiple items
- Set to `False` to free VRAM between uses
- Use smaller models or quantization if you run out of memory

### Batch Processing
- Connect a text list to process multiple prompts
- Use the same settings for consistent results
- Save outputs with descriptive filenames

### Quality vs Speed
- For quick tests: Use 0.8B model, 4-bit quantization
- For best quality: Use 7B+ model, no quantization
- For balance: Use 3B model, 8-bit quantization

## Troubleshooting

### Node Not Found
- Ensure ComfyUI-IAT is properly installed
- Check that all dependencies are installed
- Restart ComfyUI after installation

### Model Download Issues
- First run requires internet connection
- Models are cached for subsequent uses
- Check config.yaml for download source settings

### Out of Memory
- Reduce model size (use 0.8B or 3B)
- Enable quantization (8-bit or 4-bit)
- Reduce max_tokens
- Close other GPU applications

## Creating Your Own Workflows

1. Start with one of these examples
2. Add or remove nodes as needed
3. Connect outputs to inputs
4. Adjust parameters for your use case
5. Save your custom workflow

For more information, see the [Usage Guide](../docs/USAGE.md).
