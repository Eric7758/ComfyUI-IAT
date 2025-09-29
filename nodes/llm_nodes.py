import os
import sys
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局缓存模型
_QWEN_MODEL = None
_QWEN_TOKENIZER = None

def get_qwen_model():
    global _QWEN_MODEL, _QWEN_TOKENIZER
    if _QWEN_MODEL is not None:
        return _QWEN_MODEL, _QWEN_TOKENIZER

    # 严格从 config.yaml 读取模型路径（不提供默认路径）
    config = getattr(sys.modules.get("comfyui_iat_config"), "data", {})
    model_config = config.get("model", {})
    model_path = model_config.get("qwen_path")
    device_setting = model_config.get("device", "auto")

    # 检查是否配置了模型路径
    if not model_path:
        error_msg = "[IAT] ERROR: 'model.qwen_path' not found in config.yaml. Please configure it."
        print(error_msg)
        return None, None

    # 转为绝对路径（相对于插件根目录）
    plugin_root = os.path.dirname(os.path.dirname(__file__))  # ComfyUI-IAT/
    abs_model_path = os.path.abspath(os.path.join(plugin_root, model_path))

    if not os.path.exists(abs_model_path):
        error_msg = f"[IAT] ERROR: Model not found at configured path: {abs_model_path}"
        print(error_msg)
        return None, None

    try:
        # 确定设备
        if device_setting == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_setting

        print(f"[IAT] Loading Qwen model from: {abs_model_path}")
        print(f"[IAT] Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            abs_model_path,
            torch_dtype="auto",
            device_map="auto"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
        _QWEN_MODEL, _QWEN_TOKENIZER = model, tokenizer
        print("[IAT] Qwen model loaded successfully.")
        return model, tokenizer

    except Exception as e:
        error_msg = f"[IAT] ERROR: Failed to load Qwen model: {e}"
        print(error_msg)
        return None, None


# QwenTranslator
class QwenTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "请输入要翻译的文本"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "IAT"

    def detect_language(self, text):
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'ja' if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text) else 'zh'
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return None

    def translate(self, text):
        model, tokenizer = get_qwen_model()
        if model is None:
            return ("[IAT] Qwen model not loaded. Check config.yaml and model path.",)
        
        try:
            lang = self.detect_language(text)
            if not lang:
                return ("请输入中文、日文或英文",)
            if lang == 'en':
                return (text,)

            messages = [
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": f"将以下{lang}文本翻译为英文：{text}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return (response.strip(),)
        except Exception as e:
            return (f"翻译失败: {str(e)}",)


# QwenKontextTranslator
class QwenKontextTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "请输入Kontext提示词"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "IAT"

    def detect_language(self, text):
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'ja' if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text) else 'zh'
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return 'en'

    def optimize_prompt(self, text):
        model, tokenizer = get_qwen_model()
        if model is None:
            return ("[IAT] Qwen model not loaded. Check config.yaml and model path.",)
        
        try:
            lang = self.detect_language(text)
            system_content = """你将遵循以下规则使用自然语言优化我输入的提示词：
## Flux Kontext 提示词技巧

使用英文

### 1. 基础修改
- 简单直接：`"Change the car color to red"`
- 保持风格：`"Change to daytime while maintaining the same style of the painting"`

### 2. 风格转换
**原则：**
- 明确命名风格：`"Transform to Bauhaus art style"`
- 描述特征：`"Transform to oil painting with visible brushstrokes, thick paint texture"`
- 保留构图：`"Change to Bauhaus style while maintaining the original composition"`

### 3. 角色一致性
**框架：**
- 具体描述：`"The woman with short black hair"`而非`"她"`
- 保留特征：`"while maintaining the same facial features, hairstyle, and expression"`
- 分步修改：先改背景，再改动作

### 4. 文本编辑
- 使用引号：`"Replace 'joy' with 'BFL'"`
- 保持格式：`"Replace text while maintaining the same font style"`

## 常见问题解决

### 角色变化过大
❌ 错误：`"Transform the person into a Viking"`
✅ 正确：`"Change the clothes to be a viking warrior while preserving facial features"`

### 构图位置改变
❌ 错误：`"Put him on a beach"`
✅ 正确：`"Change the background to a beach while keeping the person in the exact same position, scale, and pose"`

### 风格应用不准确
❌ 错误：`"Make it a sketch"`
✅ 正确：`"Convert to pencil sketch with natural graphite lines, cross-hatching, and visible paper texture"`

## 核心原则

1. **具体明确** - 使用精确描述，避免模糊词汇
2. **分步编辑** - 复杂修改分为多个简单步骤
3. **明确保留** - 说明哪些要保持不变
4. **动词选择** - 用"更改"、"替换"而非"转换"

## 最佳实践模板

**对象修改：**
`"Change [object] to [new state], keep [content to preserve] unchanged"`

**风格转换：**
`"Transform to [specific style], while maintaining [composition/character/other] unchanged"`

**背景替换：**
`"Change the background to [new background], keep the subject in the exact same position and pose"`

**文本编辑：**
`"Replace '[original text]' with '[new text]', maintain the same font style"`

> **记住：** 越具体越好，Kontext 擅长理解详细指令并保持一致性。 

这是一些示例：

示例1
输入：汽车内饰展示正面视图，对称视角，居中向前
输出：Show the car interior from a centered, forward-facing perspective — camera positioned directly behind the steering wheel, aligned with the vehicle’s central axis, capturing a perfectly symmetrical view of the dashboard, instrument cluster, center console, and infotainment screen. Maintain true horizon level, eye-level height, and original lighting to ensure ergonomic realism. Preserve material, surface reflections, and design proportions — no distortion, no perspective skew. Output must feel balanced, immersive, and technically accurate, as used in OEM design reviews or premium product visualization.

示例2
输入：根据汽车的前45度视图和后45度视图生成这辆车的正侧面视图，不带透视
输出：Generate a pure orthographic side view (0° angle) of the car, synthesized from its front 45° and rear 45° perspective views. Use both input views to reconstruct accurate side-profile geometry — including roofline, beltline, wheel arches, and door contours — ensuring 1:1 proportional fidelity. Apply orthographic projection: eliminate all perspective, depth scaling, and vanishing points; all vertical/horizontal lines must remain parallel. Preserve original material, surface details, and design language. Output must match the visual style, lighting, color grading, and environmental context of the input images — whether studio-lit, outdoor, dusk, or overcast. Do not alter tone, texture, or atmosphere; seamlessly integrate the orthographic view into the existing visual language. Do not invent or extrapolate unseen elements; strictly derive side profile from provided views.

示例3
输入：展示汽车的正侧面视角，不带透视
输出：Display the car in a pure orthographic side view (0° yaw, 0° pitch, 0° roll), perfectly aligned with the image plane. Eliminate all perspective: no vanishing points, no foreshortening, no depth scaling. All vertical and horizontal edges must remain strictly parallel; proportions must be 1:1 true to original design. Maintain even, neutral lighting and preserve original material, texture, and surface finish. Output must resemble a technical side elevation drawing — clean, flat, dimensionally accurate, and suitable for engineering documentation or CAD reference. Do not add shadows, reflections, or environmental context.

示例4
输入：把车放到干净的工作室环境中
输出：Place the car in a clean, minimalist studio environment: seamless light-gray or white gradient backdrop with smooth cove transition to the floor, soft diffused overhead lighting, and a subtle reflective floor to naturally ground the vehicle. Preserve the car’s original position, scale, material, and surface details exactly. Adapt shadows and reflections to match the new lighting realistically. Remove all background clutter, props, or environmental distractions. The result should resemble a premium automotive product photograph — sleek, professional, and visually focused on the vehicle.

示例5：
输入：把车放到沙漠环境中
输出：Place the car in a realistic desert environment: vast arid landscape with golden sand dunes, scattered dry shrubs, and distant rocky outcrops under clear, bright daylight. The car should be positioned on natural terrain with accurate ground contact, casting a sharp, directional shadow consistent with the sun’s angle. Preserve the vehicle’s original design, proportions, and material exactly. Adapt reflections, dust accumulation on lower surfaces, and ambient lighting to match the warm, high-contrast desert atmosphere. Avoid artificial or stylized elements — the scene should feel authentic, immersive, and suitable for an off-road adventure or automotive lifestyle shoot.

示例6
输入：把车的颜色改为红色
输出：Change the car color to a rich, vibrant red — such as automotive candy red or Ferrari Rosso Corsa — with a high-gloss finish that preserves the original paint texture, reflections, and lighting response. Maintain the vehicle’s design, trim, wheels, and all non-painted surfaces exactly as they are. Ensure the new color integrates seamlessly with the existing environment, shadows, and highlights for a realistic and cohesive result.

"""
            
            user_content = f"""Please optimize the following {lang} text:

Input text: {text}

Output only the optimized English prompt without any additional explanations. """

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            if "```" in response:
                parts = response.split("```")
                response = parts[1] if len(parts) > 1 else parts[0]
            return (response.strip(),)
        except Exception as e:
            return (f"优化失败: {str(e)}",)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenTranslator by IAT": QwenTranslator, 
    "QwenKontextTranslator by IAT": QwenKontextTranslator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTranslator by IAT": "Qwen Translator by IAT", 
    "QwenKontextTranslator by IAT": "QwenKontextTranslator by IAT",
}