from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class CorruptionSeverity:
    label: str
    value: float


CORRUPTION_PROFILES: Dict[str, Dict[str, Tuple[CorruptionSeverity, ...]]] = {
    "paper3": {
        "gaussian": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15)),
        "defocus": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15)),
        "box": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15)),
        "radial": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15)),
        "gaussian_noise": tuple(CorruptionSeverity(f"sigma_{value:.2f}", value) for value in (0.02, 0.05, 0.10)),
        "shot_noise": tuple(CorruptionSeverity(f"peak_{int(value)}", value) for value in (250.0, 100.0, 25.0)),
        "jpeg": tuple(CorruptionSeverity(f"quality_{int(value)}", value) for value in (95.0, 80.0, 60.0)),
    },
    "extended5": {
        "gaussian": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15, 0.20, 0.25)),
        "defocus": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15, 0.20, 0.25)),
        "box": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15, 0.20, 0.25)),
        "radial": tuple(CorruptionSeverity(f"strength_{value:.2f}", value) for value in (0.05, 0.10, 0.15, 0.20, 0.25)),
        "gaussian_noise": tuple(CorruptionSeverity(f"sigma_{value:.2f}", value) for value in (0.02, 0.04, 0.06, 0.08, 0.10)),
        "shot_noise": tuple(CorruptionSeverity(f"peak_{int(value)}", value) for value in (500.0, 250.0, 100.0, 50.0, 25.0)),
        "jpeg": tuple(CorruptionSeverity(f"quality_{int(value)}", value) for value in (95.0, 90.0, 80.0, 60.0, 40.0)),
    },
}


def _paper_kernel_size(strength: float, base: int, scale: float, minimum: int) -> int:
    kernel_size = int(base + (strength - 0.05) * scale)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return max(minimum, min(kernel_size, 31))


def _depthwise_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [channels, height, width]")
    channels = image.size(0)
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    kernel = kernel.view(1, 1, *kernel.shape).expand(channels, 1, -1, -1)
    padding = kernel.size(-1) // 2
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels).squeeze(0)


def gaussian_blur(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _paper_kernel_size(strength, base=5, scale=57.78, minimum=3)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1.0) + 0.8
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return _depthwise_filter(image, torch.outer(kernel_1d, kernel_1d)).clamp(0.0, 1.0)


def box_blur(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _paper_kernel_size(strength, base=5, scale=57.78, minimum=3)
    kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    kernel /= kernel.numel()
    return _depthwise_filter(image, kernel).clamp(0.0, 1.0)


def defocus_blur(image: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _paper_kernel_size(strength, base=7, scale=53.33, minimum=5)
    center = kernel_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(kernel_size, dtype=torch.float32),
        torch.arange(kernel_size, dtype=torch.float32),
        indexing="ij",
    )
    distance = torch.sqrt((xx - center).pow(2) + (yy - center).pow(2))
    radius = float(center)
    kernel = torch.clamp(radius - distance, min=0.0, max=1.0)
    kernel /= kernel.sum().clamp_min(1e-12)
    return _depthwise_filter(image, kernel).clamp(0.0, 1.0)


def radial_rotational_blur(image: torch.Tensor, strength: float) -> torch.Tensor:
    # This intentionally matches the rotational radial operator used for Table 9.
    batch = image.unsqueeze(0)
    _, _, height, width = batch.shape
    center_y = height // 2
    center_x = width // 2
    yy = torch.arange(height, device=image.device, dtype=image.dtype).view(-1, 1)
    xx = torch.arange(width, device=image.device, dtype=image.dtype).view(1, -1)
    sample_count = int(3 + strength * 20)
    accumulated = torch.zeros_like(batch)
    for sample_index in range(sample_count):
        angle_offset = (sample_index / sample_count - 0.5) * strength * 0.3
        cos_a = torch.cos(torch.tensor(angle_offset, device=image.device, dtype=image.dtype))
        sin_a = torch.sin(torch.tensor(angle_offset, device=image.device, dtype=image.dtype))
        grid_x = (xx - center_x) * cos_a - (yy - center_y) * sin_a + center_x
        grid_y = (xx - center_x) * sin_a + (yy - center_y) * cos_a + center_y
        grid_x = 2.0 * grid_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * grid_y / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        accumulated += F.grid_sample(
            batch,
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
    return (accumulated / sample_count).squeeze(0).clamp(0.0, 1.0)


def gaussian_noise(image: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    return (image + noise * sigma).clamp(0.0, 1.0)


def shot_noise(image: torch.Tensor, peak: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    array = image.detach().cpu().numpy().astype(np.float64)
    noisy = rng.poisson(array * peak).astype(np.float32) / peak
    return torch.from_numpy(noisy).to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)


def jpeg_compression(image: torch.Tensor, quality: int) -> torch.Tensor:
    buffer = BytesIO()
    TF.to_pil_image(image.cpu()).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return TF.to_tensor(decoded).to(device=image.device, dtype=image.dtype)


def apply_corruption(
    image: torch.Tensor,
    corruption: str,
    severity_value: float,
    seed: int,
) -> torch.Tensor:
    if corruption == "gaussian":
        return gaussian_blur(image, severity_value)
    if corruption == "defocus":
        return defocus_blur(image, severity_value)
    if corruption == "box":
        return box_blur(image, severity_value)
    if corruption == "radial":
        return radial_rotational_blur(image, severity_value)
    if corruption == "gaussian_noise":
        return gaussian_noise(image, severity_value, seed)
    if corruption == "shot_noise":
        return shot_noise(image, severity_value, seed)
    if corruption == "jpeg":
        return jpeg_compression(image, int(severity_value))
    raise ValueError(f"Unknown corruption: {corruption}")


def available_corruptions(profile: str) -> Iterable[str]:
    if profile not in CORRUPTION_PROFILES:
        raise ValueError(f"Unknown corruption profile: {profile}")
    return CORRUPTION_PROFILES[profile].keys()
