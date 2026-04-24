import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Dict, List, Optional

from nodes import MAX_RESOLUTION

_SORT_ORDER_OPTIONS = ["ratio_desc", "ratio_asc", "lightness"]


def _tensor_to_pil_batch(image: torch.Tensor) -> List[Image.Image]:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    pil_images: List[Image.Image] = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(arr).convert("RGB"))
    return pil_images


def _pil_batch_to_tensor(images: List[Image.Image], device=None) -> torch.Tensor:
    np_batch = [np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 for img in images]
    tensor = torch.from_numpy(np.stack(np_batch, axis=0))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _extract_palette_data(
    image: Image.Image,
    num_colors: int,
    min_ratio: float,
    sort_order: str,
) -> List[Dict]:
    safe_image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = safe_image.size
    longest_side = max(1, max(width, height))
    scale = min(1.0, 400.0 / float(longest_side))
    if scale < 1.0:
        resized = safe_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        resized = safe_image

    quantized = resized.quantize(colors=num_colors)
    palette_raw = quantized.getpalette() or []
    pixels = np.asarray(quantized).reshape(-1)
    total = max(1, int(pixels.size))

    color_data = []
    for i in range(num_colors):
        base = i * 3
        if base + 2 >= len(palette_raw):
            continue
        ratio = float(np.sum(pixels == i)) / float(total)
        if ratio < min_ratio:
            continue
        color_data.append(
            {
                "rgb": (palette_raw[base], palette_raw[base + 1], palette_raw[base + 2]),
                "ratio": ratio,
            }
        )

    if not color_data:
        color_data = [{"rgb": (128, 128, 128), "ratio": 1.0}]

    total_ratio = sum(item["ratio"] for item in color_data)
    if total_ratio <= 0:
        total_ratio = 1.0
    for item in color_data:
        item["ratio"] = item["ratio"] / total_ratio

    if sort_order == "ratio_desc":
        color_data.sort(key=lambda x: x["ratio"], reverse=True)
    elif sort_order == "ratio_asc":
        color_data.sort(key=lambda x: x["ratio"])
    elif sort_order == "lightness":
        color_data.sort(
            key=lambda x: (
                0.2126 * (x["rgb"][0] / 255.0)
                + 0.7152 * (x["rgb"][1] / 255.0)
                + 0.0722 * (x["rgb"][2] / 255.0)
            )
        )

    return color_data


def _render_palette_image(
    color_data: List[Dict],
    output_width: int,
    output_height: int,
) -> Image.Image:
    result = Image.new("RGB", (output_width, output_height))
    offset = 0
    for idx, item in enumerate(color_data):
        if idx == len(color_data) - 1:
            bar_width = output_width - offset
        else:
            bar_width = int(output_width * item["ratio"])
        if bar_width <= 0:
            continue
        result.paste(Image.new("RGB", (bar_width, output_height), item["rgb"]), (offset, 0))
        offset += bar_width
    return result


def _build_color_info(color_data: List[Dict], image_index: Optional[int] = None) -> str:
    lines = []
    if image_index is not None:
        lines.append(f"Image {image_index}")
    lines.append("Color  |   HEX   |      RGB      |  Ratio")
    lines.append("-" * 44)
    for i, item in enumerate(color_data, start=1):
        r, g, b = item["rgb"]
        lines.append(f"Color {i:02d} | #{r:02X}{g:02X}{b:02X} | ({r:3d},{g:3d},{b:3d}) | {item['ratio'] * 100:5.1f}%")
    return "\n".join(lines)


class ImageColorPaletteExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": (
                    "INT",
                    {
                        "default": 6,
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "tooltip": "Number of dominant colors to extract.",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 64,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Output palette image width in pixels.",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 400,
                        "min": 64,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "Output palette image height in pixels.",
                    },
                ),
                "min_ratio": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Ignore colors whose pixel ratio is lower than this value.",
                    },
                ),
                "sort_order": (
                    _SORT_ORDER_OPTIONS,
                    {
                        "default": "ratio_desc",
                        "tooltip": "Sort bars by descending ratio, ascending ratio, or lightness.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("palette_image", "color_info")
    FUNCTION = "extract_palette"
    CATEGORY = "IAT/Image"
    DESCRIPTION = "Extract dominant colors from image and render a ratio-based palette bar chart."

    def extract_palette(
        self,
        image: torch.Tensor,
        num_colors: int,
        output_width: int,
        output_height: int,
        min_ratio: float,
        sort_order: str,
    ):
        pil_batch = _tensor_to_pil_batch(image)
        palette_images: List[Image.Image] = []
        info_blocks: List[str] = []

        is_batch = len(pil_batch) > 1
        for idx, pil_image in enumerate(pil_batch, start=1):
            color_data = _extract_palette_data(
                image=pil_image,
                num_colors=num_colors,
                min_ratio=min_ratio,
                sort_order=sort_order,
            )
            palette_images.append(
                _render_palette_image(
                    color_data=color_data,
                    output_width=output_width,
                    output_height=output_height,
                )
            )
            info_blocks.append(_build_color_info(color_data=color_data, image_index=idx if is_batch else None))

        info_text = "\n\n".join(info_blocks)
        print(f"[IAT] ImageColorPaletteExtractor: processed {len(palette_images)} image(s)")
        return (_pil_batch_to_tensor(palette_images, device=image.device), info_text)


NODE_CLASS_MAPPINGS = {
    "ImageColorPaletteExtractor by IAT": ImageColorPaletteExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColorPaletteExtractor by IAT": "Image Color Palette Extractor by IAT",
}
