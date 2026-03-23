from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    label: int
    category: str


@dataclass
class EvalDatasetConfigRecord:
    name: str
    type: str
    real_folder: Optional[str] = None
    fake_folder: Optional[str] = None
    base_path: Optional[str] = None
    classes: Optional[list[str]] = None


def _iter_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


class BinaryAIGCDataset(Dataset):
    def __init__(self, root: str | Path, transform, max_samples_per_class: Optional[int] = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._discover_samples(max_samples_per_class=max_samples_per_class)

    def _discover_samples(self, max_samples_per_class: Optional[int]) -> list[ImageRecord]:
        samples: list[ImageRecord] = []
        if not self.root.exists():
            raise FileNotFoundError(f"Training root does not exist: {self.root}")

        for category_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            category_name = category_dir.name
            real_images = list(_iter_images(category_dir / "0_real"))
            fake_images = list(_iter_images(category_dir / "1_fake"))

            if max_samples_per_class is not None:
                real_images = real_images[:max_samples_per_class]
                fake_images = fake_images[:max_samples_per_class]

            samples.extend(ImageRecord(path=path, label=0, category=category_name) for path in real_images)
            samples.extend(ImageRecord(path=path, label=1, category=category_name) for path in fake_images)

        if not samples:
            raise RuntimeError(f"No training images were found under: {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": sample.label,
            "image_name": sample.path.stem,
            "category": sample.category,
            "path": str(sample.path),
        }


class EvaluationDataset(Dataset):
    def __init__(self, config: EvalDatasetConfigRecord, transform) -> None:
        self.config = config
        self.transform = transform
        self.samples = self._discover_samples()

    def _discover_samples(self) -> list[ImageRecord]:
        if self.config.type == "simple":
            if not self.config.real_folder or not self.config.fake_folder:
                raise ValueError(f"Dataset '{self.config.name}' is missing real/fake folders.")
            samples = [ImageRecord(path=path, label=0, category="real") for path in _iter_images(Path(self.config.real_folder))]
            samples.extend(ImageRecord(path=path, label=1, category="fake") for path in _iter_images(Path(self.config.fake_folder)))
            return samples

        if self.config.type == "multi_class":
            if not self.config.base_path:
                raise ValueError(f"Dataset '{self.config.name}' is missing base_path.")
            base_path = Path(self.config.base_path)
            if not base_path.exists():
                raise FileNotFoundError(f"Dataset base path does not exist: {base_path}")

            class_names = self.config.classes
            if class_names is None:
                class_names = sorted(
                    item.name
                    for item in base_path.iterdir()
                    if item.is_dir() and ((item / "0_real").exists() or (item / "1_fake").exists())
                )

            samples: list[ImageRecord] = []
            for class_name in class_names:
                class_path = base_path / class_name
                samples.extend(
                    ImageRecord(path=path, label=0, category=f"{class_name}_real")
                    for path in _iter_images(class_path / "0_real")
                )
                samples.extend(
                    ImageRecord(path=path, label=1, category=f"{class_name}_fake")
                    for path in _iter_images(class_path / "1_fake")
                )
            return samples

        raise ValueError(f"Unsupported dataset type: {self.config.type}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": sample.label,
            "image_name": sample.path.stem,
            "category": sample.category,
            "path": str(sample.path),
        }
