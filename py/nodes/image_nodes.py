import numpy as np
import torch
from PIL import Image

from nodes import MAX_RESOLUTION


def _tensor_to_pil_batch(image: torch.Tensor):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    pil_images = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr))
    return pil_images


def _pil_batch_to_tensor(images, device=None):
    np_batch = [np.asarray(img).astype(np.float32) / 255.0 for img in images]
    tensor = torch.from_numpy(np.stack(np_batch, axis=0))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class ImageMatchSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "match_size"
    CATEGORY = "IAT/Image"

    def match_size(self, reference_image, input_image):
        ref_pil = _tensor_to_pil_batch(reference_image)[0]
        target_w, target_h = ref_pil.size

        in_pil_batch = _tensor_to_pil_batch(input_image)
        out_pil_batch = [img.resize((target_w, target_h), Image.Resampling.LANCZOS) for img in in_pil_batch]
        return (_pil_batch_to_tensor(out_pil_batch, device=input_image.device),)


class ImageResizeLongestSideNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_side": ("INT", {"default": 1536, "min": 64, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_longest_side"
    CATEGORY = "IAT/Image"

    def resize_longest_side(self, image, longest_side):
        pil_batch = _tensor_to_pil_batch(image)
        out = []
        for img in pil_batch:
            width, height = img.size
            if width >= height:
                new_width = longest_side
                new_height = max(1, int(round(height * (longest_side / max(1, width)))))
            else:
                new_height = longest_side
                new_width = max(1, int(round(width * (longest_side / max(1, height)))))
            out.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
        return (_pil_batch_to_tensor(out, device=image.device),)


class ImageResizeToSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("resized_16x", "resized_original_ratio")
    FUNCTION = "resize_image"
    CATEGORY = "IAT/Image"

    def resize_image(self, image):
        target_pixels = 1152 * 1152
        pil_batch = _tensor_to_pil_batch(image)

        out_4x = []
        out_ratio = []

        for img in pil_batch:
            orig_width, orig_height = img.size
            orig_pixels = max(1, orig_width * orig_height)

            scale = (target_pixels / orig_pixels) ** 0.5
            new_width_float = orig_width * scale
            new_height_float = orig_height * scale

            new_width_ratio = max(1, int(round(new_width_float)))
            new_height_ratio = max(1, int(round(new_height_float)))
            out_ratio.append(img.resize((new_width_ratio, new_height_ratio), Image.Resampling.LANCZOS))

            new_width_4x = max(4, (int(round(new_width_float)) // 4) * 4)
            new_height_4x = max(4, (int(round(new_height_float)) // 4) * 4)
            if new_width_4x * new_height_4x > target_pixels:
                scale_int = (target_pixels / (new_width_4x * new_height_4x)) ** 0.5
                new_width_4x = max(4, int((new_width_4x * scale_int) // 4) * 4)
                new_height_4x = max(4, int((new_height_4x * scale_int) // 4) * 4)

            out_4x.append(img.resize((new_width_4x, new_height_4x), Image.Resampling.LANCZOS))

        return (
            _pil_batch_to_tensor(out_4x, device=image.device),
            _pil_batch_to_tensor(out_ratio, device=image.device),
        )


class ImageSizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "IAT/Image"

    def get_size(self, image):
        pil = _tensor_to_pil_batch(image)[0]
        width, height = pil.size
        return (width, height)


NODE_CLASS_MAPPINGS = {
    "ImageMatchSize by IAT": ImageMatchSize,
    "ImageResizeLongestSide by IAT": ImageResizeLongestSideNode,
    "ImageResizeToSDXL by IAT": ImageResizeToSDXL,
    "ImageSize by IAT": ImageSizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMatchSize by IAT": "Image Match Size by IAT",
    "ImageResizeLongestSide by IAT": "Image Resize Longest Side by IAT",
    "ImageResizeToSDXL by IAT": "Image Resize To SDXL by IAT",
    "ImageSize by IAT": "Image Size by IAT",
}
