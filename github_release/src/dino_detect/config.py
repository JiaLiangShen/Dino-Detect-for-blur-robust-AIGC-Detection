from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_known_fields(payload: dict[str, Any], cls: type) -> dict[str, Any]:
    known = {field.name for field in cls.__dataclass_fields__.values()}
    return {key: value for key, value in payload.items() if key in known}


@dataclass
class TrainConfig:
    experiment_name: str = "dino_detect_teacher_student"
    backbone_path: str = ""
    train_root: str = ""
    output_dir: str = "outputs/dino_detect"
    seed: int = 42
    num_workers: int = 4
    num_classes: int = 2
    image_size: int = 512
    crop_size: int = 448
    projection_dim: int = 512
    adapter_layers: int = 3
    dropout: float = 0.1
    teacher_epochs: int = 8
    student_epochs: int = 15
    teacher_batch_size: int = 64
    student_batch_size: int = 32
    teacher_lr: float = 1e-4
    student_lr: float = 5e-5
    weight_decay: float = 1e-4
    temperature: float = 0.07
    alpha_distill: float = 1.0
    alpha_cls: float = 1.0
    alpha_feature: float = 0.5
    alpha_contrastive: float = 0.3
    blur_mode: str = "motion"
    blur_probability: float = 0.2
    blur_strength_range: list[float] = field(default_factory=lambda: [0.1, 0.3])
    mixed_blur_ratio: float = 0.5
    ccmba_root: Optional[str] = None
    max_samples_per_class: Optional[int] = None
    amp: bool = True
    log_interval: int = 20
    save_every_epoch: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        payload = _load_json(path)
        return cls(**_filter_known_fields(payload, cls))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalDatasetConfig:
    name: str
    type: str
    real_folder: Optional[str] = None
    fake_folder: Optional[str] = None
    base_path: Optional[str] = None
    classes: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalDatasetConfig":
        return cls(**_filter_known_fields(payload, cls))


@dataclass
class EvalConfig:
    experiment_name: str = "dino_detect_eval"
    backbone_path: str = ""
    teacher_checkpoint: str = ""
    student_checkpoint: str = ""
    output_dir: str = "outputs/eval"
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42
    num_classes: int = 2
    image_size: int = 512
    crop_size: int = 448
    projection_dim: int = 512
    adapter_layers: int = 3
    dropout: float = 0.1
    model_mode: str = "both"
    blur_mode: str = "both"
    blur_strength_range: list[float] = field(default_factory=lambda: [0.1, 0.3])
    datasets: list[EvalDatasetConfig] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalConfig":
        payload = _load_json(path)
        datasets = [EvalDatasetConfig.from_dict(item) for item in payload.get("datasets", [])]
        filtered = _filter_known_fields(payload, cls)
        filtered["datasets"] = datasets
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [asdict(item) for item in self.datasets]
        return payload
