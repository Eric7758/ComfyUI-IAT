import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import datetime
import re

# SmartPathBuilderNode
def _format_datetime(fmt: str) -> str:
    """Format date/time string with ComfyUI-style tokens."""
    now = datetime.datetime.now()
    # Escape literal % (though rare in practice)
    fmt = fmt.replace("%%", "__ESCAPED_PERCENT__")
    fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y")
    fmt = fmt.replace("MM", "%m").replace("dd", "%d")
    fmt = fmt.replace("HH", "%H").replace("mm", "%M").replace("ss", "%S")
    fmt = fmt.replace("__ESCAPED_PERCENT__", "%%")
    return now.strftime(fmt)

class SmartPathBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level1": ("STRING", {"default": "", "multiline": False, "tooltip": "First path component (can be absolute path like C:\\Users\\xxx or %date%)"}),
                "level2": ("STRING", {"default": "", "multiline": False}),
                "level3": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build_path"
    CATEGORY = "utils"
    DESCRIPTION = "Builds a path from three string levels with %date:fmt% and %time:fmt% support. Handles absolute paths automatically."

    def _parse_placeholders(self, s: str) -> str:
        """Parse %date:...% and %time:...% placeholders."""
        def replace_date(match):
            fmt = match.group(1) or "yyyy-MM-dd"
            return _format_datetime(fmt)

        def replace_time(match):
            fmt = match.group(1) or "HH-mm-ss"
            return _format_datetime(fmt)

        s = re.sub(r"%date(?:\:([^%]*))?%", replace_date, s)
        s = re.sub(r"%time(?:\:([^%]*))?%", replace_time, s)
        return s

    def build_path(self, level1, level2, level3):
        # Collect non-empty levels
        raw_parts = [level1.strip(), level2.strip(), level3.strip()]
        parts = [p for p in raw_parts if p]

        if not parts:
            return ("",)

        # Parse placeholders in each part
        parsed_parts = [self._parse_placeholders(p) for p in parts]

        # Join all parts with os.path.join — this automatically handles:
        # - If first part is absolute (e.g., "C:\\" or "/xxx"), the rest are appended as subdirs
        # - If all are relative, it builds a relative path
        full_path = os.path.join(*parsed_parts)

        # Normalize path (resolve redundant separators, ., .. if any)
        full_path = os.path.normpath(full_path)

        return (full_path,)

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
    "SmartPathBuilderNode by IAT": SmartPathBuilderNode,
    "Base64ToImageNode by IAT": Base64ToImageNode,
    "FloatInputNode by IAT": FloatInputNode,
    "IntInputNode by IAT": IntInputNode,
    "TextInputNode by IAT": TextInputNode,
    "SeedGeneratorNode by IAT": SeedGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartPathBuilderNode by IAT": "Smart Path Builder by IAT",
    "Base64ToImageNode by IAT": "Base64 to Image by IAT",
    "FloatInputNode by IAT": "Float Input by IAT",
    "IntInputNode by IAT": "Integer Input by IAT",
    "TextInputNode by IAT": "Text Input by IAT",
    "SeedGeneratorNode by IAT": "Seed Generator by IAT",
}