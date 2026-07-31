import json
import math
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class TransformConfig:
    resize_size: int = 224
    crop_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_image_files(folder_path: str | Path, recursive: bool = True) -> List[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image_safely(image_path: str | Path, max_retries: int = 3) -> Image.Image | None:
    image_path = Path(image_path)
    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert("RGB")
            _ = image.size
            return image
        except Exception:
            if attempt == max_retries - 1:
                return None
    return None


def _kernel_size_from_strength(strength: float) -> int:
    kernel_size = int(5 + max(strength - 0.05, 0.0) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return max(3, min(kernel_size, 31))


def _motion_kernel_2d(
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
    angle_degrees: float | None = None,
    trajectory_jitter: float = 0.12,
    phase: float | None = None,
) -> torch.Tensor:
    angle = math.radians(random.uniform(0.0, 180.0) if angle_degrees is None else angle_degrees)
    phase = random.uniform(0.0, 2.0 * math.pi) if phase is None else phase
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    along = xx * math.cos(angle) + yy * math.sin(angle)
    perpendicular = -xx * math.sin(angle) + yy * math.cos(angle)
    half_length = max((kernel_size - 2) / 2.0, 1.0)
    curvature = trajectory_jitter * kernel_size * torch.sin(
        math.pi * along / half_length + phase
    )
    distance = perpendicular - curvature
    support = (along.abs() <= half_length).to(dtype)
    kernel = torch.exp(-0.5 * (distance / 0.55).pow(2)) * support
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def apply_motion_blur_to_pil(
    image_pil: Image.Image,
    strength: float,
    angle_degrees: float | None = None,
    trajectory_jitter: float = 0.12,
    phase: float | None = None,
) -> Image.Image:
    tensor = TF.to_tensor(image_pil).unsqueeze(0)
    blurred = apply_motion_blur_to_tensor(
        tensor,
        strength,
        angle_degrees=angle_degrees,
        trajectory_jitter=trajectory_jitter,
        phase=phase,
    ).squeeze(0)
    return TF.to_pil_image(blurred.clamp(0.0, 1.0))


def apply_motion_blur_to_tensor(
    images: torch.Tensor,
    strength: float,
    angle_degrees: float | None = None,
    trajectory_jitter: float = 0.12,
    phase: float | None = None,
) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")

    kernel_size = _kernel_size_from_strength(strength)
    device = images.device
    channels = images.shape[1]

    kernel = _motion_kernel_2d(
        kernel_size,
        device=device,
        dtype=images.dtype,
        angle_degrees=angle_degrees,
        trajectory_jitter=trajectory_jitter,
        phase=phase,
    )
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, -1, -1)

    padding = kernel_size // 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def apply_gaussian_blur_to_tensor(images: torch.Tensor, strength: float) -> torch.Tensor:
    kernel_size = _kernel_size_from_strength(strength)
    sigma = max(0.1, strength * 10.0)
    return TF.gaussian_blur(images, kernel_size=kernel_size, sigma=sigma)


def apply_blur_to_tensor(
    images: torch.Tensor,
    blur_type: str,
    strength: float,
    **kwargs,
) -> torch.Tensor:
    if blur_type == "motion":
        return apply_motion_blur_to_tensor(images, strength, **kwargs)
    if blur_type == "gaussian":
        return apply_gaussian_blur_to_tensor(images, strength)
    raise ValueError(f"Unknown blur_type: {blur_type}")


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def build_blur_kernel(blur_type: str, strength: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    kernel_size = _kernel_size_from_strength(strength)
    if blur_type == "motion":
        kernel = torch.zeros(kernel_size, kernel_size, device=device, dtype=dtype)
        kernel[kernel_size // 2, :] = 1.0
        return kernel / kernel.sum()
    if blur_type == "gaussian":
        sigma = max(0.1, strength * 10.0)
        return _gaussian_kernel_2d(kernel_size, sigma, device, dtype)
    raise ValueError(f"Unknown blur_type: {blur_type}")


def _expand_stats(stats: Tuple[float, float, float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(stats, device=device, dtype=dtype).view(1, -1, 1, 1)


def unnormalize_tensor_images(images: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    return images * _expand_stats(std, images.device, images.dtype) + _expand_stats(mean, images.device, images.dtype)


def normalize_tensor_images(images: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    return (images - _expand_stats(mean, images.device, images.dtype)) / _expand_stats(std, images.device, images.dtype)


def apply_blur_to_normalized_tensor(
    images: torch.Tensor,
    blur_type: str,
    strength: float,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    **kwargs,
) -> torch.Tensor:
    input_was_3d = images.ndim == 3
    if input_was_3d:
        images = images.unsqueeze(0)
    raw_images = unnormalize_tensor_images(images.float(), mean, std).clamp(0.0, 1.0)
    blurred_raw = apply_blur_to_tensor(raw_images, blur_type, strength, **kwargs).clamp(0.0, 1.0)
    normalized = normalize_tensor_images(blurred_raw, mean, std)
    return normalized.squeeze(0) if input_was_3d else normalized


def apply_wiener_deblur_to_tensor(images: torch.Tensor, blur_type: str, strength: float, regularization: float = 0.01, mean: Tuple[float, float, float] | None = None, std: Tuple[float, float, float] | None = None) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")

    input_was_3d = images.ndim == 3
    if input_was_3d:
        images = images.unsqueeze(0)

    working = images.float()
    if mean is not None and std is not None:
        working = unnormalize_tensor_images(working, mean, std)
    working = working.clamp(0.0, 1.0)

    batch_size, channels, height, width = working.shape
    kernel_small = build_blur_kernel(blur_type, strength, device=working.device, dtype=working.dtype)
    psf = torch.zeros((height, width), device=working.device, dtype=working.dtype)
    kh, kw = kernel_small.shape
    top = (height - kh) // 2
    left = (width - kw) // 2
    psf[top : top + kh, left : left + kw] = kernel_small
    psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf))
    wiener_filter = torch.conj(psf_fft) / (torch.abs(psf_fft).pow(2) + regularization)

    restored_fft = torch.fft.fft2(working, dim=(-2, -1)) * wiener_filter.view(1, 1, height, width)
    restored = torch.fft.ifft2(restored_fft, dim=(-2, -1)).real.clamp(0.0, 1.0)

    if mean is not None and std is not None:
        restored = normalize_tensor_images(restored, mean, std)
    return restored.squeeze(0) if input_was_3d else restored


def apply_pil_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_cv2_jpeg(img: Image.Image, quality: int) -> Image.Image:
    img_cv2 = np.array(img)[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img_cv2, encode_param)
    decoded = cv2.imdecode(encoded, 1)
    return Image.fromarray(decoded[:, :, ::-1])


def apply_random_jpeg(img: Image.Image, quality: int) -> Image.Image:
    if random.random() < 0.5:
        return apply_pil_jpeg(img, quality)
    return apply_cv2_jpeg(img, quality)


def apply_light_jpeg(img: Image.Image, quality_range: Tuple[int, int] = (85, 95)) -> Image.Image:
    quality = random.randint(quality_range[0], quality_range[1])
    return apply_random_jpeg(img, quality)


def build_train_transform(config: TransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size)),
            transforms.RandomCrop((config.crop_size, config.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )


def build_strong_train_transform(config: TransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size)),
            transforms.RandomCrop((config.crop_size, config.crop_size)),
            transforms.Lambda(lambda img: apply_light_jpeg(img) if random.random() < 0.3 else img),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )


def build_paper_train_transform(
    config: TransformConfig,
    include_jpeg: bool = True,
) -> transforms.Compose:
    operations = [
        transforms.Resize((config.resize_size, config.resize_size)),
        transforms.RandomCrop((config.crop_size, config.crop_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=5),
    ]
    if include_jpeg:
        operations.append(
            transforms.Lambda(lambda img: apply_light_jpeg(img) if random.random() < 0.3 else img)
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    return transforms.Compose(operations)


def apply_paper_co_degradations_to_normalized_tensor(
    image: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    defocus_prob: float = 0.2,
    noise_prob: float = 0.1,
    jpeg_prob: float = 0.1,
    resize_prob: float = 0.1,
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [channels, height, width]")

    device = image.device
    dtype = image.dtype
    raw = unnormalize_tensor_images(image.unsqueeze(0).float(), mean, std).clamp(0.0, 1.0)

    if random.random() < defocus_prob:
        sigma = random.uniform(0.1, 2.5)
        kernel_size = min(31, max(3, 2 * math.ceil(3.0 * sigma) + 1))
        raw = TF.gaussian_blur(raw, kernel_size=kernel_size, sigma=sigma)

    if random.random() < noise_prob:
        noise_std = random.uniform(0.002, 0.02)
        raw = (raw + torch.randn_like(raw) * noise_std).clamp(0.0, 1.0)

    if random.random() < resize_prob:
        height, width = raw.shape[-2:]
        scale = random.uniform(0.5, 0.9)
        down_size = (max(8, int(height * scale)), max(8, int(width * scale)))
        raw = F.interpolate(raw, size=down_size, mode="bilinear", align_corners=False, antialias=True)
        raw = F.interpolate(raw, size=(height, width), mode="bilinear", align_corners=False, antialias=True)

    if random.random() < jpeg_prob:
        quality = random.randint(70, 95)
        pil_image = TF.to_pil_image(raw.squeeze(0).cpu())
        raw = TF.to_tensor(apply_random_jpeg(pil_image, quality)).unsqueeze(0).to(device=device)

    normalized = normalize_tensor_images(raw.to(device=device, dtype=dtype), mean, std)
    return normalized.squeeze(0)


def build_eval_transform(config: TransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size)),
            transforms.CenterCrop((config.crop_size, config.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )


class CCMBADataLoader:
    def __init__(self, ccmba_root_dir: str | Path, categories: Sequence[str] | None = None):
        self.ccmba_root_dir = Path(ccmba_root_dir)
        self.categories = list(categories) if categories else None
        self.real_mapping: Dict[str, Dict[str, Path]] = {}
        self.fake_mapping: Dict[str, Dict[str, Path]] = {}
        self.real_subdirs = ("nature", "0_real")
        self.fake_subdirs = ("ai", "1_fake")
        self._build_all_mappings()

    def _resolve_class_dir(self, category_dir: Path, candidates: Iterable[str]) -> Path | None:
        for name in candidates:
            candidate = category_dir / name
            if candidate.exists():
                return candidate
        return None

    def _build_all_mappings(self) -> None:
        if not self.ccmba_root_dir.exists():
            return

        if self.categories is None:
            categories = []
            for path in self.ccmba_root_dir.iterdir():
                if not path.is_dir():
                    continue
                if self._resolve_class_dir(path, self.real_subdirs) or self._resolve_class_dir(path, self.fake_subdirs):
                    categories.append(path.name)
            self.categories = sorted(categories)

        for category in self.categories:
            category_dir = self.ccmba_root_dir / category
            if not category_dir.exists():
                continue

            real_dir = self._resolve_class_dir(category_dir, self.real_subdirs)
            fake_dir = self._resolve_class_dir(category_dir, self.fake_subdirs)

            if real_dir:
                for key, value in self._build_mapping(real_dir).items():
                    self.real_mapping[f"{category}_{key}"] = value
            if fake_dir:
                for key, value in self._build_mapping(fake_dir).items():
                    self.fake_mapping[f"{category}_{key}"] = value

    def _build_mapping(self, class_dir: Path) -> Dict[str, Dict[str, Path]]:
        mapping: Dict[str, Dict[str, Path]] = {}
        blurred_images_dir = class_dir / "blurred_images"
        blur_masks_dir = class_dir / "blur_masks"
        metadata_dir = class_dir / "metadata"

        if not all(path.exists() for path in (blurred_images_dir, blur_masks_dir, metadata_dir)):
            return mapping

        for blur_img_path in blurred_images_dir.glob("*.jpg"):
            base_name = blur_img_path.stem
            blur_mask_path = blur_masks_dir / f"{base_name}.png"
            metadata_path = metadata_dir / f"{base_name}.json"
            if blur_mask_path.exists() and metadata_path.exists():
                mapping[base_name] = {
                    "blurred_image": blur_img_path,
                    "blur_mask": blur_mask_path,
                    "metadata": metadata_path,
                }
        return mapping

    def get_ccmba_data(self, image_name: str, category: str, is_real: bool) -> Dict[str, Path] | None:
        mapping = self.real_mapping if is_real else self.fake_mapping
        return mapping.get(f"{category}_{image_name}")

    def load_ccmba_blur_data(self, image_name: str, category: str, is_real: bool):
        ccmba_data = self.get_ccmba_data(image_name, category, is_real)
        if ccmba_data is None:
            return None, None, None

        blurred_image = load_image_safely(ccmba_data["blurred_image"])
        if blurred_image is None:
            return None, None, None

        blur_mask = None
        metadata = {}
        try:
            blur_mask = np.array(Image.open(ccmba_data["blur_mask"])) / 255.0
        except Exception:
            blur_mask = None

        try:
            with open(ccmba_data["metadata"], "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            metadata = {}

        return blurred_image, blur_mask, metadata


class BinaryFolderDataset(Dataset):
    def __init__(
        self,
        root_folder: str | Path | None = None,
        real_folder: str | Path | None = None,
        fake_folder: str | Path | None = None,
        transform: transforms.Compose | None = None,
        blur_prob: float = 0.0,
        blur_strength_range: Tuple[float, float] = (0.1, 0.3),
        blur_mode: str = "global",
        blur_type: str = "motion",
        mixed_mode_ratio: float = 0.5,
        ccmba_data_dir: str | Path | None = None,
        max_samples_per_class: int | None = None,
        enable_strong_aug: bool = False,
        strong_transform: transforms.Compose | None = None,
        strong_aug_prob: float = 0.4,
        normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.transform = transform
        self.blur_prob = blur_prob
        self.blur_strength_range = blur_strength_range
        self.blur_mode = blur_mode
        self.blur_type = blur_type
        self.mixed_mode_ratio = mixed_mode_ratio
        self.enable_strong_aug = enable_strong_aug
        self.strong_transform = strong_transform
        self.strong_aug_prob = strong_aug_prob
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.data: List[Tuple[Path, int, str]] = []
        self.ccmba_loader = CCMBADataLoader(ccmba_data_dir) if ccmba_data_dir and blur_mode in {"ccmba", "mixed"} else None

        if root_folder:
            self._load_category_root(Path(root_folder))
        elif real_folder and fake_folder:
            for img_path in get_image_files(real_folder):
                self.data.append((img_path, 0, "real"))
            for img_path in get_image_files(fake_folder):
                self.data.append((img_path, 1, "fake"))
        else:
            raise ValueError("Either root_folder or both real_folder/fake_folder must be provided.")

        if max_samples_per_class is not None:
            real_items = [item for item in self.data if item[1] == 0][:max_samples_per_class]
            fake_items = [item for item in self.data if item[1] == 1][:max_samples_per_class]
            self.data = real_items + fake_items

    def _load_category_root(self, root_folder: Path) -> None:
        for category_folder in sorted(root_folder.iterdir()):
            if not category_folder.is_dir():
                continue

            category_name = category_folder.name
            real_subfolder = None
            fake_subfolder = None
            for real_name in ("0_real", "nature"):
                candidate = category_folder / real_name
                if candidate.exists():
                    real_subfolder = candidate
                    break
            for fake_name in ("1_fake", "ai"):
                candidate = category_folder / fake_name
                if candidate.exists():
                    fake_subfolder = candidate
                    break

            if real_subfolder is not None:
                for img_file in get_image_files(real_subfolder):
                    self.data.append((img_file, 0, category_name))
            if fake_subfolder is not None:
                for img_file in get_image_files(fake_subfolder):
                    self.data.append((img_file, 1, category_name))

    def apply_blur_augmentation(self, tensor_image: torch.Tensor, image_name: str, category: str, is_real: bool):
        if self.blur_mode == "no_blur" or random.random() > self.blur_prob:
            return tensor_image, {"mode": "no_blur", "blur_applied": False}

        if self.blur_mode == "global":
            strength = random.uniform(*self.blur_strength_range)
            blurred = apply_blur_to_normalized_tensor(
                tensor_image,
                self.blur_type,
                strength,
                self.normalization_mean,
                self.normalization_std,
            )
            return blurred, {
                "mode": "global",
                "blur_type": self.blur_type,
                "blur_strength": strength,
                "blur_applied": True,
            }

        if self.blur_mode == "ccmba":
            if self.ccmba_loader is None:
                strength = random.uniform(*self.blur_strength_range)
                blurred = apply_blur_to_normalized_tensor(
                    tensor_image,
                    self.blur_type,
                    strength,
                    self.normalization_mean,
                    self.normalization_std,
                )
                return blurred, {
                    "mode": "ccmba_fallback_global",
                    "blur_type": self.blur_type,
                    "blur_strength": strength,
                    "blur_applied": True,
                }

            blurred_pil, blur_mask, metadata = self.ccmba_loader.load_ccmba_blur_data(image_name, category, is_real)
            if blurred_pil is None:
                strength = random.uniform(*self.blur_strength_range)
                blurred = apply_blur_to_normalized_tensor(
                    tensor_image,
                    self.blur_type,
                    strength,
                    self.normalization_mean,
                    self.normalization_std,
                )
                return blurred, {
                    "mode": "ccmba_fallback_global",
                    "blur_type": self.blur_type,
                    "blur_strength": strength,
                    "blur_applied": True,
                }

            blurred_tensor = self.transform(blurred_pil)
            return blurred_tensor, {
                "mode": "ccmba",
                "blur_applied": True,
                "blur_mask_ratio": float(np.mean(blur_mask)) if blur_mask is not None else 0.0,
                "metadata": metadata,
            }

        if self.blur_mode == "mixed":
            if self.ccmba_loader is not None and random.random() < self.mixed_mode_ratio:
                blurred_pil, blur_mask, metadata = self.ccmba_loader.load_ccmba_blur_data(image_name, category, is_real)
                if blurred_pil is not None:
                    blurred_tensor = self.transform(blurred_pil)
                    return blurred_tensor, {
                        "mode": "mixed_ccmba",
                        "blur_applied": True,
                        "blur_mask_ratio": float(np.mean(blur_mask)) if blur_mask is not None else 0.0,
                        "metadata": metadata,
                    }

            strength = random.uniform(*self.blur_strength_range)
            blurred = apply_blur_to_normalized_tensor(
                tensor_image,
                self.blur_type,
                strength,
                self.normalization_mean,
                self.normalization_std,
            )
            return blurred, {
                "mode": "mixed_global",
                "blur_type": self.blur_type,
                "blur_strength": strength,
                "blur_applied": True,
            }

        return tensor_image, {"mode": self.blur_mode, "blur_applied": False}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.data)
                img_path, label, category = self.data[current_idx]
                image = load_image_safely(img_path)
                if image is None:
                    raise RuntimeError(f"Failed to load image: {img_path}")

                if self.enable_strong_aug and self.strong_transform is not None and random.random() < self.strong_aug_prob:
                    tensor_image = self.strong_transform(image)
                else:
                    tensor_image = self.transform(image)
                return tensor_image, label, img_path.stem, category
            except Exception:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Too many consecutive corrupted images starting from index {idx}")
        raise RuntimeError("Unexpected error in __getitem__")


class MultiTestDataset(Dataset):
    def __init__(
        self,
        dataset_config: Dict[str, object],
        transform: transforms.Compose,
        blur_strength_range: Tuple[float, float] = (0.1, 0.3),
        blur_type: str = "motion",
        normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.dataset_config = dataset_config
        self.transform = transform
        self.blur_strength_range = blur_strength_range
        self.blur_type = blur_type
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.data: List[Tuple[Path, int, str]] = []

        if dataset_config["type"] == "simple":
            self._load_simple_dataset(dataset_config)
        elif dataset_config["type"] == "multi_class":
            self._load_multi_class_dataset(dataset_config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

    def _load_simple_dataset(self, config: Dict[str, object]) -> None:
        for img_path in get_image_files(config["real_folder"]):
            self.data.append((img_path, 0, "real"))
        for img_path in get_image_files(config["fake_folder"]):
            self.data.append((img_path, 1, "fake"))

    def _load_multi_class_dataset(self, config: Dict[str, object]) -> None:
        base_path = Path(config["base_path"])
        classes = config.get("classes")
        if classes is None:
            classes = sorted(
                item.name
                for item in base_path.iterdir()
                if item.is_dir() and ((item / "0_real").exists() or (item / "1_fake").exists())
            )
        for class_name in classes:
            class_path = base_path / class_name
            real_folder = class_path / "0_real"
            fake_folder = class_path / "1_fake"
            for img_path in get_image_files(real_folder):
                self.data.append((img_path, 0, f"{class_name}_real"))
            for img_path in get_image_files(fake_folder):
                self.data.append((img_path, 1, f"{class_name}_fake"))

    def apply_blur_augmentation(self, tensor_image: torch.Tensor):
        strength = random.uniform(*self.blur_strength_range)
        blurred = apply_blur_to_normalized_tensor(
            tensor_image,
            self.blur_type,
            strength,
            self.normalization_mean,
            self.normalization_std,
        )
        return blurred, {
            "mode": "global",
            "blur_type": self.blur_type,
            "blur_strength": strength,
            "blur_applied": True,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path, label, category = self.data[idx]
        image = load_image_safely(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        tensor_image = self.transform(image)
        return tensor_image, label, f"{category}_{img_path.stem}"


def validate_dataset(dataset_config: Dict[str, object]) -> bool:
    if dataset_config["type"] == "simple":
        return Path(dataset_config["real_folder"]).exists() and Path(dataset_config["fake_folder"]).exists()
    if dataset_config["type"] == "multi_class":
        return Path(dataset_config["base_path"]).exists()
    return False


def collect_image_paths(data_root: str | Path, max_images: int | None = None, recursive: bool = True) -> List[Path]:
    image_paths = get_image_files(data_root, recursive=recursive)
    if max_images is not None:
        return image_paths[:max_images]
    return image_paths
