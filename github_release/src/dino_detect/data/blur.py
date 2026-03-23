from __future__ import annotations

import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .transforms import IMAGENET_MEAN, IMAGENET_STD


def _kernel_size_from_strength(strength: float) -> int:
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return max(3, min(kernel_size, 31))


def _unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def _normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean) / std


def _tensor_to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    unnormalized = _unnormalize_image(tensor.detach()).cpu().permute(1, 2, 0).numpy()
    return np.clip(unnormalized * 255.0, 0, 255).astype(np.uint8)


def _numpy_uint8_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    return _normalize_image(tensor)


def apply_motion_blur_tensor(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _kernel_size_from_strength(strength)
    kernel = torch.zeros((kernel_size, kernel_size), device=image.device, dtype=image.dtype)
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(image.shape[0], 1, -1, -1)
    blurred = F.conv2d(image.unsqueeze(0), kernel, padding=kernel_size // 2, groups=image.shape[0])
    return blurred.squeeze(0)


def apply_gaussian_blur_tensor(image: torch.Tensor, strength: float) -> torch.Tensor:
    sigma = max(0.3, strength * 8.0)
    kernel_size = _kernel_size_from_strength(strength)
    array = _tensor_to_numpy_uint8(image)
    blurred = cv2.GaussianBlur(array, (kernel_size, kernel_size), sigmaX=sigma)
    return _numpy_uint8_to_tensor(blurred, image.device)


def apply_box_blur_tensor(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _kernel_size_from_strength(strength)
    array = _tensor_to_numpy_uint8(image)
    blurred = cv2.blur(array, (kernel_size, kernel_size))
    return _numpy_uint8_to_tensor(blurred, image.device)


def apply_bokeh_blur_tensor(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _kernel_size_from_strength(strength)
    radius = kernel_size // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (x * x + y * y <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    array = _tensor_to_numpy_uint8(image)
    blurred = cv2.filter2D(array, -1, kernel)
    return _numpy_uint8_to_tensor(blurred, image.device)


def apply_radial_blur_tensor(image: torch.Tensor, strength: float) -> torch.Tensor:
    array = _tensor_to_numpy_uint8(image)
    height, width = array.shape[:2]
    center = (width / 2.0, height / 2.0)
    blend_count = max(4, int(4 + strength * 16))
    accumulator = np.zeros_like(array, dtype=np.float32)

    for blend_index in range(blend_count):
        alpha = blend_index / max(blend_count - 1, 1)
        scale = 1.0 + alpha * strength * 0.08
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
        warped = cv2.warpAffine(array, matrix, (width, height), flags=cv2.INTER_LINEAR)
        accumulator += warped.astype(np.float32)

    blurred = np.clip(accumulator / blend_count, 0, 255).astype(np.uint8)
    return _numpy_uint8_to_tensor(blurred, image.device)


def apply_jpeg_compression_tensor(image: torch.Tensor, quality: int) -> torch.Tensor:
    pil_image = Image.fromarray(_tensor_to_numpy_uint8(image))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    return _numpy_uint8_to_tensor(np.array(compressed), image.device)


@dataclass
class CCMBARecord:
    blurred_image: Path
    blur_mask: Path
    metadata: Path


class CCMBABlurBank:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.real_index: dict[str, CCMBARecord] = {}
        self.fake_index: dict[str, CCMBARecord] = {}
        self._index_records()

    def _index_records(self) -> None:
        if not self.root.exists():
            return

        for category_dir in self.root.iterdir():
            if not category_dir.is_dir():
                continue
            for split_name, target in (("nature", self.real_index), ("ai", self.fake_index)):
                split_dir = category_dir / split_name
                images_dir = split_dir / "blurred_images"
                masks_dir = split_dir / "blur_masks"
                metadata_dir = split_dir / "metadata"
                if not images_dir.exists():
                    continue

                for blurred_image in images_dir.glob("*.jpg"):
                    key = f"{category_dir.name}:{blurred_image.stem}"
                    target[key] = CCMBARecord(
                        blurred_image=blurred_image,
                        blur_mask=masks_dir / f"{blurred_image.stem}.png",
                        metadata=metadata_dir / f"{blurred_image.stem}.json",
                    )

    def get(self, image_name: str, category: str, is_real: bool) -> Optional[CCMBARecord]:
        key = f"{category}:{image_name}"
        index = self.real_index if is_real else self.fake_index
        return index.get(key)

    def load_image(self, image_name: str, category: str, is_real: bool) -> Optional[Image.Image]:
        record = self.get(image_name=image_name, category=category, is_real=is_real)
        if record is None or not record.blurred_image.exists():
            return None
        return Image.open(record.blurred_image).convert("RGB")

    def load_metadata(self, image_name: str, category: str, is_real: bool) -> dict[str, object]:
        record = self.get(image_name=image_name, category=category, is_real=is_real)
        if record is None or not record.metadata.exists():
            return {}
        with record.metadata.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class BlurAugmentor:
    def __init__(
        self,
        mode: str,
        probability: float,
        strength_range: tuple[float, float],
        mixed_ratio: float = 0.5,
        ccmba_root: Optional[str] = None,
        ccmba_transform=None,
    ) -> None:
        self.mode = mode
        self.probability = probability
        self.strength_range = strength_range
        self.mixed_ratio = mixed_ratio
        self.ccmba_transform = ccmba_transform
        self.ccmba_bank = CCMBABlurBank(ccmba_root) if ccmba_root else None

    def _sample_strength(self) -> float:
        return random.uniform(*self.strength_range)

    def _apply_named_blur(self, image: torch.Tensor, blur_mode: str, strength: float) -> torch.Tensor:
        if blur_mode == "motion":
            return apply_motion_blur_tensor(image, strength)
        if blur_mode == "gaussian":
            return apply_gaussian_blur_tensor(image, strength)
        if blur_mode == "box":
            return apply_box_blur_tensor(image, strength)
        if blur_mode == "radial":
            return apply_radial_blur_tensor(image, strength)
        if blur_mode == "bokeh":
            return apply_bokeh_blur_tensor(image, strength)
        if blur_mode == "jpeg":
            quality = max(20, min(95, int(100 - strength * 100)))
            return apply_jpeg_compression_tensor(image, quality)
        return image

    def apply(
        self,
        image: torch.Tensor,
        image_name: str,
        category: str,
        is_real: bool,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if self.mode in {"none", "no_blur"} or random.random() > self.probability:
            return image, {"mode": "none", "applied": False}

        if self.mode == "ccmba" and self.ccmba_bank is not None:
            loaded = self.ccmba_bank.load_image(image_name=image_name, category=category, is_real=is_real)
            if loaded is not None and self.ccmba_transform is not None:
                return (
                    self.ccmba_transform(loaded).to(image.device),
                    {"mode": "ccmba", "applied": True},
                )

        if self.mode == "mixed":
            if self.ccmba_bank is not None and random.random() < self.mixed_ratio:
                loaded = self.ccmba_bank.load_image(image_name=image_name, category=category, is_real=is_real)
                if loaded is not None and self.ccmba_transform is not None:
                    return (
                        self.ccmba_transform(loaded).to(image.device),
                        {"mode": "ccmba", "applied": True},
                    )
            strength = self._sample_strength()
            return self._apply_named_blur(image, "motion", strength), {
                "mode": "motion",
                "applied": True,
                "strength": strength,
            }

        strength = self._sample_strength()
        return self._apply_named_blur(image, self.mode, strength), {
            "mode": self.mode,
            "applied": True,
            "strength": strength,
        }
