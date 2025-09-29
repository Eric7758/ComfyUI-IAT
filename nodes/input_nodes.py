import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Base64ToImageNode
class Base64ToImageNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_str": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_base64"
    CATEGORY = "IAT/Input"

    def convert_base64(self, base64_str):
        try:
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            return (img_tensor,)
        except Exception as e:
            print(f"Base64转换失败: {str(e)}")
            blank_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_img,)

# FloatInputNode
class FloatInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "IAT/Input"

    def get_float(self, value):
        return (value,)

# IntInputNode
class IntInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "IAT/Input"

    def get_int(self, value):
        return (value,)

# TextInputNode
class TextInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "请输入文本"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_text"
    CATEGORY = "IAT/Input"

    def get_text(self, text):
        return (text,)

# SeedGeneratorNode
class SeedGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999999,
                    "step": 1
                }),
                "control_after_generate": ("BOOLEAN", {
                    "default": True
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_seed"
    CATEGORY = "IAT/Input"

    def generate_seed(self, seed, control_after_generate):
        return (seed,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "Base64ToImageNode by IAT": Base64ToImageNode,
    "FloatInputNode by IAT": FloatInputNode,
    "IntInputNode by IAT": IntInputNode,
    "TextInputNode by IAT": TextInputNode,
    "SeedGeneratorNode by IAT": SeedGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64ToImageNode by IAT": "Base64 to Image by IAT",
    "FloatInputNode by IAT": "Float Input by IAT",
    "IntInputNode by IAT": "Integer Input by IAT",
    "TextInputNode by IAT": "Text Input by IAT",
    "SeedGeneratorNode by IAT": "Seed Generator by IAT",
}