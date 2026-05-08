import os
import subprocess
import tempfile
import wave

import numpy as np
import torch
from PIL import Image

import folder_paths


_MP3_QUALITY_OPTIONS = ["V0", "128k", "320k"]


def _tensor_to_cover_image(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dimensions, got {tuple(image.shape)}")

    arr = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _audio_to_waveform_and_sample_rate(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected ComfyUI AUDIO dict containing waveform and sample_rate")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected AUDIO waveform with 1, 2 or 3 dimensions, got {tuple(waveform.shape)}")

    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")

    return waveform, sample_rate


def _write_temp_wav(audio, wav_path: str) -> None:
    waveform, sample_rate = _audio_to_waveform_and_sample_rate(audio)
    pcm = waveform.clamp(-1.0, 1.0).transpose(0, 1).numpy()
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _get_ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_output_file(filename_prefix: str, extension: str):
    output_dir = folder_paths.get_output_directory()
    try:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
    except TypeError:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, 0, 0)

    os.makedirs(full_output_folder, exist_ok=True)
    while True:
        file = f"{filename}_{counter:05}_.{extension}"
        path = os.path.join(full_output_folder, file)
        if not os.path.exists(path):
            return path, file, subfolder
        counter += 1


def _quality_args(quality: str):
    if quality == "V0":
        return ["-q:a", "0"]
    return ["-b:a", quality]


def _save_mp3_with_cover(audio, image, filename_prefix: str, quality: str):
    if quality not in _MP3_QUALITY_OPTIONS:
        raise ValueError(f"Unsupported MP3 quality: {quality}")

    output_path, filename, subfolder = _get_output_file(filename_prefix, "mp3")
    cover_image = _tensor_to_cover_image(image)
    ffmpeg_exe = _get_ffmpeg_exe()

    with tempfile.TemporaryDirectory(prefix="iat_audio_cover_") as temp_dir:
        wav_path = os.path.join(temp_dir, "audio.wav")
        cover_path = os.path.join(temp_dir, "cover.jpg")
        _write_temp_wav(audio, wav_path)
        cover_image.save(cover_path, format="JPEG", quality=95)

        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            wav_path,
            "-i",
            cover_path,
            "-map",
            "0:a:0",
            "-map",
            "1:v:0",
            "-c:a",
            "libmp3lame",
            *_quality_args(quality),
            "-c:v",
            "mjpeg",
            "-disposition:v:0",
            "attached_pic",
            "-id3v2_version",
            "3",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"Failed to save MP3 with cover: {message}")

    return {"filename": filename, "subfolder": subfolder, "type": "output"}


class SaveAudioMP3WithCoverNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI", "multiline": False}),
                "quality": (_MP3_QUALITY_OPTIONS, {"default": "V0"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio_with_cover"
    OUTPUT_NODE = True
    CATEGORY = "IAT/Audio"
    DESCRIPTION = "Save a ComfyUI AUDIO input as MP3 and embed the first IMAGE input as the album cover."

    def save_audio_with_cover(self, audio, image, filename_prefix="audio/ComfyUI", quality="V0", prompt=None, extra_pnginfo=None):
        audio_info = _save_mp3_with_cover(audio, image, filename_prefix, quality)
        return {"ui": {"audio": [audio_info]}}


NODE_CLASS_MAPPINGS = {
    "SaveAudioMP3WithCover by IAT": SaveAudioMP3WithCoverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioMP3WithCover by IAT": "Save Audio MP3 With Cover by IAT",
}
