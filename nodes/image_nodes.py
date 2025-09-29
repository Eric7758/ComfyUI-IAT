import torch
import numpy as np
from PIL import Image
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION  # 注意：这里需要从 comfy 导入 MAX_RESOLUTION

# ImageMatchSize
class ImageMatchSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "match_size"
    CATEGORY = "IAT"

    def match_size(self, reference_image, input_image):
        ref_img_np = 255. * reference_image[0].cpu().numpy()
        ref_img_pil = Image.fromarray(np.clip(ref_img_np, 0, 255).astype(np.uint8))
        ref_width, ref_height = ref_img_pil.size

        input_img_np = 255. * input_image[0].cpu().numpy()
        input_img_pil = Image.fromarray(np.clip(input_img_np, 0, 255).astype(np.uint8))
        resized_img_pil = input_img_pil.resize((ref_width, ref_height), Image.Resampling.LANCZOS)

        resized_img_np = np.array(resized_img_pil).astype(np.float32) / 255.0
        resized_img_tensor = torch.from_numpy(resized_img_np)[None,]
        return (resized_img_tensor,)

# ImageResizeLongestSideNode
class ImageResizeLongestSideNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_side": ("INT", {"default": 1536, "min": 64, "max": MAX_RESOLUTION}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_longest_side"
    CATEGORY = "IAT"

    def resize_longest_side(self, image, longest_side):
        img_np = 255. * image[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        width, height = img_pil.size

        if width > height:
            new_width = longest_side
            new_height = int(round(height * (longest_side / width)))
        else:
            new_height = longest_side
            new_width = int(round(width * (longest_side / height)))

        resized_img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img_np = np.array(resized_img_pil).astype(np.float32) / 255.0
        resized_img_tensor = torch.from_numpy(resized_img_np)[None,]
        return (resized_img_tensor,)

# ImageResizeToSDXL
class ImageResizeToSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("resized_16x", "resized_original_ratio")
    FUNCTION = "resize_image"
    CATEGORY = "IAT"

    def resize_image(self, image):
        target_pixels = 1152 * 1152
        img_np = 255. * image[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        orig_width, orig_height = img_pil.size
        orig_pixels = orig_width * orig_height

        if orig_pixels == 0:
            print("Warning: Input image has zero pixels.")
            blank_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_img, blank_img)

        scale = (target_pixels / orig_pixels) ** 0.5
        new_width_float = orig_width * scale
        new_height_float = orig_height * scale

        new_width_ratio = max(1, int(round(new_width_float)))
        new_height_ratio = max(1, int(round(new_height_float)))
        img_ratio_pil = img_pil.resize((new_width_ratio, new_height_ratio), Image.Resampling.LANCZOS)
        img_ratio_np = np.array(img_ratio_pil).astype(np.float32) / 255.0
        img_ratio_tensor = torch.from_numpy(img_ratio_np)[None,]

        new_width_4x_base = int(round(new_width_float))
        new_height_4x_base = int(round(new_height_float))
        new_width_4x = max(4, (new_width_4x_base // 4) * 4)
        new_height_4x = max(4, (new_height_4x_base // 4) * 4)

        if new_width_4x * new_height_4x > target_pixels:
            scale_int = (target_pixels / (new_width_4x * new_height_4x)) ** 0.5
            new_width_4x = max(4, int((new_width_4x * scale_int) // 4) * 4)
            new_height_4x = max(4, int((new_height_4x * scale_int) // 4) * 4)

        img_4x_pil = img_pil.resize((new_width_4x, new_height_4x), Image.Resampling.LANCZOS)
        img_4x_np = np.array(img_4x_pil).astype(np.float32) / 255.0
        img_4x_tensor = torch.from_numpy(img_4x_np)[None,]

        return (img_4x_tensor, img_ratio_tensor)

# ImageSizeNode
class ImageSizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "IAT"

    def get_size(self, image):
        img_np = 255. * image[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        width, height = img_pil.size
        return (width, height)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageMatchSize by IAT": ImageMatchSize,
    "ImageResizeLongestSide by IAT": ImageResizeLongestSideNode,
    "ImageResizeToSDXL by IAT": ImageResizeToSDXL,
    "ImageSize by IAT": ImageSizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMatchSize by IAT": "Image Match Size by IAT",
    "ImageResizeLongestSide by IAT": "Image Resize Longest Side by IAT",
    "ImageResizeToSDXL by IAT": "ImageResizeToSDXL by IAT",
    "ImageSize by IAT": "Image Size by IAT",
}