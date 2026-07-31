import argparse
import csv
import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blur_generalization_suite.common import compute_binary_metrics, load_checkpoint_state, save_json
from blur_generalization_suite.corruptions import CORRUPTION_PROFILES, CorruptionSeverity, apply_corruption
from blur_generalization_suite.data_utils import TransformConfig, get_image_files, load_image_safely
from blur_generalization_suite.model_zoo import (
    create_teacher_student_model_from_config,
    load_teacher_student_head_state_dict,
)


DEFAULT_DINOV3_7B = "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m"
DEFAULT_TRANSFORM = TransformConfig(
    resize_size=512,
    crop_size=448,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
TABLE_ROWS = (
    ("Gaussian blur", ("gaussian",)),
    ("Defocus blur", ("defocus",)),
    ("Box/radial blur", ("box", "radial")),
    ("Gaussian noise", ("gaussian_noise",)),
    ("Shot noise", ("shot_noise",)),
    ("JPEG compression", ("jpeg",)),
)
METRIC_KEYS = ("accuracy", "bacc", "real_accuracy", "fake_accuracy", "f1_score")


def stable_corruption_seed(
    base_seed: int,
    relative_path: str,
    corruption: str,
    severity_value: float,
) -> int:
    payload = f"{base_seed}|{relative_path}|{corruption}|{severity_value:.8g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


class WildRFCorruptionDataset(Dataset):
    def __init__(
        self,
        wildrf_root: str | Path,
        platform: str,
        transform_config: TransformConfig,
        corruption: str,
        severity: CorruptionSeverity,
        seed: int,
    ):
        self.wildrf_root = Path(wildrf_root)
        self.platform = platform
        self.transform_config = transform_config
        self.corruption = corruption
        self.severity = severity
        self.seed = seed
        self.samples = []
        platform_root = self.wildrf_root / platform
        real_files = get_image_files(platform_root / "0_real")
        fake_files = get_image_files(platform_root / "1_fake")
        self.class_counts = {0: len(real_files), 1: len(fake_files)}
        for path in real_files:
            self.samples.append((path, 0))
        for path in fake_files:
            self.samples.append((path, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = load_image_safely(path)
        if image is None:
            raise RuntimeError(f"Failed to decode image: {path}")
        image = TF.resize(
            image,
            [self.transform_config.resize_size, self.transform_config.resize_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = TF.center_crop(
            image,
            [self.transform_config.crop_size, self.transform_config.crop_size],
        )
        raw = TF.to_tensor(image)
        relative_path = path.relative_to(self.wildrf_root).as_posix()
        corruption_seed = stable_corruption_seed(
            self.seed,
            relative_path,
            self.corruption,
            self.severity.value,
        )
        corrupted = apply_corruption(
            raw,
            self.corruption,
            self.severity.value,
            corruption_seed,
        )
        normalized = TF.normalize(
            corrupted,
            mean=self.transform_config.mean,
            std=self.transform_config.std,
        )
        return normalized, label, relative_path


def transform_from_config(config: dict, fallback: TransformConfig) -> TransformConfig:
    raw = config.get("transform_config")
    if not raw:
        return fallback
    transform = TransformConfig(
        resize_size=int(raw["resize_size"]),
        crop_size=int(raw["crop_size"]),
        mean=tuple(float(value) for value in raw["mean"]),
        std=tuple(float(value) for value in raw["std"]),
    )
    if transform.resize_size <= 0 or transform.crop_size <= 0 or transform.crop_size > transform.resize_size:
        raise ValueError(f"Invalid checkpoint transform_config: {raw}")
    if len(transform.mean) != 3 or len(transform.std) != 3 or any(value <= 0 for value in transform.std):
        raise ValueError(f"Invalid checkpoint normalization statistics: {raw}")
    return transform


def normalized_checkpoint_config(
    checkpoint_path: str,
    args: argparse.Namespace,
) -> dict:
    checkpoint, _ = load_checkpoint_state(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    config.setdefault("dinov3_model_id", args.dinov3_model_id or DEFAULT_DINOV3_7B)
    config.setdefault("backbone_family", "dinov3")
    config.setdefault("backbone_preset", "dinov3_vit7b")
    config.setdefault("projection_dim", 512)
    config.setdefault("local_files_only", args.local_files_only)
    config.setdefault(
        "transform_config",
        {
            "resize_size": args.resize_size,
            "crop_size": args.crop_size,
            "mean": list(DEFAULT_TRANSFORM.mean),
            "std": list(DEFAULT_TRANSFORM.std),
        },
    )
    return config


def strict_provenance_violations(config: dict, expected_mode: str) -> List[str]:
    checks = {
        "strict_motion_only": config.get("strict_motion_only") is True,
        "training_profile": config.get("training_profile") == "strict_motion",
        "experiment_mode": config.get("experiment_mode") == expected_mode,
        "blur_mode": config.get("blur_mode") == "global",
        "blur_type": config.get("blur_type") == "motion",
        "ccmba_data_dir": not config.get("ccmba_data_dir"),
        "jpeg_augmentation": config.get("jpeg_augmentation") is False,
        "co_degradations": config.get("co_degradations") is False,
        "defocus_augmentation": config.get("defocus_augmentation") is False,
        "noise_augmentation": config.get("noise_augmentation") is False,
        "resize_degradation": config.get("resize_degradation") is False,
        "explicit_synthetic_augmentations": config.get("explicit_synthetic_augmentations")
        == ["motion_blur"],
    }
    return [name for name, passed in checks.items() if not passed]


def assert_comparable_configs(classification_config: dict, dino_detect_config: dict) -> TransformConfig:
    fields = (
        "dinov3_model_id",
        "backbone_family",
        "backbone_preset",
        "loader_backend",
        "architecture_name",
        "projection_dim",
        "data_preset",
        "train_root",
        "max_samples_per_class",
        "blur_mode",
        "blur_type",
        "blur_prob",
        "blur_strength_range",
        "trajectory_jitter",
        "classification_loss",
        "focal_gamma",
        "class_balanced_focal",
        "resolved_focal_alpha",
        "student_epochs",
        "student_learning_rate",
        "weight_decay",
        "max_grad_norm",
        "student_micro_batch",
        "student_accumulation_steps",
        "student_global_effective_batch",
        "world_size",
        "seed",
    )
    mismatches = [
        field
        for field in fields
        if classification_config.get(field) != dino_detect_config.get(field)
    ]
    classification_transform = transform_from_config(classification_config, DEFAULT_TRANSFORM)
    dino_transform = transform_from_config(dino_detect_config, DEFAULT_TRANSFORM)
    if classification_transform != dino_transform:
        mismatches.append("transform_config")
    if mismatches:
        raise RuntimeError(
            "The two checkpoints are not a one-variable control. Mismatched fields: "
            + ", ".join(mismatches)
        )
    return classification_transform


def load_student_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
    backbone_override: str | None,
):
    _, state_dict = load_checkpoint_state(checkpoint_path, map_location="cpu")
    required_prefixes = ("student_projection.", "student_classifier.")
    missing_prefixes = [
        prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in state_dict)
    ]
    if missing_prefixes:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain a student head: {missing_prefixes}"
        )
    model = create_teacher_student_model_from_config(
        config,
        device=device,
        backbone_path_override=backbone_override,
    )
    load_teacher_student_head_state_dict(model, state_dict, branch="student")
    model.eval()
    return model


def evaluate_loader(model, loader, device: torch.device):
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[float] = []
    paths: List[str] = []
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            _, logits = model.forward_student(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            labels.extend(int(value) for value in batch_labels.tolist())
            predictions.extend(int(value) for value in preds.cpu().tolist())
            probabilities.extend(float(value) for value in probs.cpu().tolist())
            paths.extend(batch_paths)
    return labels, predictions, probabilities, paths


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sample_manifest(wildrf_root: Path, platforms: Iterable[str]) -> List[dict]:
    rows = []
    for platform in platforms:
        for label, folder_name in ((0, "0_real"), (1, "1_fake")):
            files = get_image_files(wildrf_root / platform / folder_name)
            if not files:
                raise RuntimeError(
                    f"WildRF/{platform}/{folder_name} has no images; both classes are required."
                )
            for path in files:
                stat = path.stat()
                rows.append(
                    {
                        "platform": platform,
                        "label": label,
                        "path": path.relative_to(wildrf_root).as_posix(),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256_file(path),
                    }
                )
    return rows


def sample_manifest_sha256(rows: List[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aggregate_results(per_dataset_rows: List[dict], per_image_rows: List[dict]):
    severity_rows = []
    grouped_dataset = defaultdict(list)
    for row in per_dataset_rows:
        grouped_dataset[(row["model"], row["corruption"], row["severity_label"], row["severity_value"])].append(row)

    grouped_images = defaultdict(list)
    for row in per_image_rows:
        grouped_images[(row["model"], row["corruption"], row["severity_label"], row["severity_value"])].append(row)

    for key, dataset_rows in grouped_dataset.items():
        model_name, corruption, severity_label, severity_value = key
        image_rows = grouped_images[key]
        micro = compute_binary_metrics(
            [row["label"] for row in image_rows],
            [row["prediction"] for row in image_rows],
        )
        result = {
            "model": model_name,
            "corruption": corruption,
            "severity_label": severity_label,
            "severity_value": severity_value,
            "platform_count": len(dataset_rows),
            "total_samples": sum(row["total_samples"] for row in dataset_rows),
        }
        for metric in METRIC_KEYS:
            result[f"macro_{metric}"] = sum(row[metric] for row in dataset_rows) / len(dataset_rows)
            result[f"micro_{metric}"] = micro[metric]
        severity_rows.append(result)

    corruption_rows = []
    grouped_severity = defaultdict(list)
    for row in severity_rows:
        grouped_severity[(row["model"], row["corruption"])].append(row)
    grouped_corruption_images = defaultdict(list)
    for row in per_image_rows:
        grouped_corruption_images[(row["model"], row["corruption"])].append(row)

    for (model_name, corruption), rows in grouped_severity.items():
        image_rows = grouped_corruption_images[(model_name, corruption)]
        pooled = compute_binary_metrics(
            [row["label"] for row in image_rows],
            [row["prediction"] for row in image_rows],
        )
        result = {
            "model": model_name,
            "corruption": corruption,
            "severity_count": len(rows),
            "pooled_prediction_count": len(image_rows),
        }
        for metric in METRIC_KEYS:
            result[f"macro_{metric}"] = sum(row[f"macro_{metric}"] for row in rows) / len(rows)
            result[f"mean_severity_micro_{metric}"] = (
                sum(row[f"micro_{metric}"] for row in rows) / len(rows)
            )
            result[f"micro_{metric}"] = pooled[metric]
        corruption_rows.append(result)
    return severity_rows, corruption_rows


def build_paper_table(corruption_rows: List[dict]) -> List[dict]:
    lookup = {(row["model"], row["corruption"]): row for row in corruption_rows}
    table = []
    for display_name, corruption_names in TABLE_ROWS:
        row = {"test_corruption": display_name}
        for model_name in ("classification_only", "dino_detect"):
            component_rows = [
                lookup[(model_name, name)]
                for name in corruption_names
                if (model_name, name) in lookup
            ]
            if len(component_rows) != len(corruption_names):
                break
            value = sum(item["macro_accuracy"] for item in component_rows) / len(component_rows)
            row[model_name] = 100.0 * value
        if "classification_only" in row and "dino_detect" in row:
            table.append(row)
    return table


def write_markdown_table(path: Path, table_rows: List[dict], profile: str) -> None:
    lines = [
        f"Accuracy (%), platform-macro and severity-averaged with profile `{profile}`.",
        "",
        "| Test corruption | Classification-only DINOv3-7B Accuracy (%) | DINO-Detect Accuracy (%) |",
        "| --- | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['test_corruption']} | {row['classification_only']:.2f} | {row['dino_detect']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict motion-only zero-shot evaluation on unseen corruption families."
    )
    parser.add_argument("--classification-checkpoint", required=True)
    parser.add_argument("--dino-detect-checkpoint", required=True)
    parser.add_argument("--dinov3-model-id", default=None)
    parser.add_argument("--wildrf-root", default="/data/app.e0016372/WildRF/test")
    parser.add_argument("--platforms", nargs="+", default=["reddit", "facebook", "twitter"])
    parser.add_argument("--profile", choices=list(CORRUPTION_PROFILES), default="paper3")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=list(CORRUPTION_PROFILES["paper3"]),
        default=list(CORRUPTION_PROFILES["paper3"]),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default="blur_generalization_suite/outputs/reviewer_cross_corruption")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--allow-unverified-checkpoints", action="store_true")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--allow-remote-backbone", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")
    if args.resize_size <= 0 or args.crop_size <= 0 or args.crop_size > args.resize_size:
        raise ValueError("Fallback resize/crop sizes must be positive with crop <= resize")
    if len(set(args.platforms)) != len(args.platforms):
        raise ValueError("--platforms must not contain duplicates")
    if len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError("--corruptions must not contain duplicates")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = {
        "classification_only": args.classification_checkpoint,
        "dino_detect": args.dino_detect_checkpoint,
    }
    configs = {
        name: normalized_checkpoint_config(path, args)
        for name, path in checkpoint_paths.items()
    }
    violations = {
        name: strict_provenance_violations(configs[name], name)
        for name in checkpoint_paths
    }
    if any(violations.values()) and not args.allow_unverified_checkpoints:
        raise RuntimeError(
            "Strict motion-only provenance check failed. "
            + json.dumps(violations, ensure_ascii=False)
            + ". Use --allow-unverified-checkpoints only for diagnostics, not rebuttal numbers."
        )
    is_unverified = any(violations.values())
    if is_unverified:
        print(
            "WARNING: checkpoint provenance is unverified. Diagnostic artifacts will be "
            "written under an UNVERIFIED directory and cannot be used as formal table results."
        )
    transform_config = assert_comparable_configs(
        configs["classification_only"], configs["dino_detect"]
    )

    wildrf_root = Path(args.wildrf_root)
    missing_platforms = [name for name in args.platforms if not (wildrf_root / name).exists()]
    if missing_platforms:
        raise FileNotFoundError(
            f"WildRF platform directories not found under {wildrf_root}: {missing_platforms}"
        )
    sample_manifest = build_sample_manifest(wildrf_root, args.platforms)
    manifest_sha256 = sample_manifest_sha256(sample_manifest)
    checkpoint_sha256 = {
        name: sha256_file(path) for name, path in checkpoint_paths.items()
    }
    run_name = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{checkpoint_sha256['classification_only'][:8]}"
        + f"_{checkpoint_sha256['dino_detect'][:8]}"
    )
    if (
        not run_name
        or run_name in {".", ".."}
        or Path(run_name).name != run_name
        or any(character in run_name for character in ("/", "\\"))
    ):
        raise ValueError("--run-name must be a safe single directory name")
    output_base = Path(args.output_dir)
    if is_unverified:
        output_base = output_base / "UNVERIFIED"
    output_dir = output_base / run_name
    if output_dir.exists():
        raise FileExistsError(
            f"Output run directory already exists: {output_dir}. Choose a new --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    per_dataset_rows: List[dict] = []
    per_image_rows: List[dict] = []
    for model_name, checkpoint_path in checkpoint_paths.items():
        print(f"Loading {model_name}: {checkpoint_path}")
        model = load_student_model(
            checkpoint_path,
            configs[model_name],
            device,
            args.dinov3_model_id,
        )
        for corruption in args.corruptions:
            for severity in CORRUPTION_PROFILES[args.profile][corruption]:
                for platform in args.platforms:
                    dataset = WildRFCorruptionDataset(
                        wildrf_root,
                        platform,
                        transform_config,
                        corruption,
                        severity,
                        args.seed,
                    )
                    if not dataset.samples:
                        raise RuntimeError(f"No samples found for WildRF/{platform}")
                    if any(count == 0 for count in dataset.class_counts.values()):
                        raise RuntimeError(
                            f"WildRF/{platform} must contain both 0_real and 1_fake images: "
                            f"counts={dataset.class_counts}"
                        )
                    loader_kwargs = {
                        "dataset": dataset,
                        "batch_size": args.batch_size,
                        "shuffle": False,
                        "num_workers": args.num_workers,
                        "pin_memory": True,
                        "persistent_workers": args.num_workers > 0,
                    }
                    if args.num_workers > 0:
                        loader_kwargs["prefetch_factor"] = 2
                    loader = DataLoader(**loader_kwargs)
                    labels, predictions, probabilities, paths = evaluate_loader(model, loader, device)
                    metrics = compute_binary_metrics(labels, predictions)
                    result = {
                        "model": model_name,
                        "platform": platform,
                        "corruption": corruption,
                        "severity_label": severity.label,
                        "severity_value": severity.value,
                        **{key: metrics[key] for key in METRIC_KEYS},
                        "total_samples": metrics["total_samples"],
                    }
                    per_dataset_rows.append(result)
                    for path, label, prediction, probability in zip(
                        paths, labels, predictions, probabilities
                    ):
                        per_image_rows.append(
                            {
                                "model": model_name,
                                "platform": platform,
                                "corruption": corruption,
                                "severity_label": severity.label,
                                "severity_value": severity.value,
                                "path": path,
                                "label": label,
                                "prediction": prediction,
                                "prob_fake": probability,
                            }
                        )
                    print(
                        f"  {platform}/{corruption}/{severity.label}: "
                        f"acc={100.0 * metrics['accuracy']:.2f}, bacc={100.0 * metrics['bacc']:.2f}"
                    )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    severity_rows, corruption_rows = aggregate_results(per_dataset_rows, per_image_rows)
    table_rows = build_paper_table(corruption_rows)
    write_csv(
        output_dir / "sample_manifest.csv",
        sample_manifest,
        ["platform", "label", "path", "size_bytes", "mtime_ns", "sha256"],
    )
    write_csv(
        output_dir / "per_dataset_severity.csv",
        per_dataset_rows,
        list(per_dataset_rows[0]),
    )
    write_csv(output_dir / "per_image_predictions.csv", per_image_rows, list(per_image_rows[0]))
    write_csv(output_dir / "severity_macro.csv", severity_rows, list(severity_rows[0]))
    write_csv(output_dir / "corruption_summary.csv", corruption_rows, list(corruption_rows[0]))
    table_prefix = "UNVERIFIED_" if is_unverified else ""
    write_csv(
        output_dir / f"{table_prefix}paper_table.csv",
        table_rows,
        ["test_corruption", "classification_only", "dino_detect"],
    )
    write_markdown_table(
        output_dir / f"{table_prefix}paper_table.md", table_rows, args.profile
    )
    save_json(
        output_dir / "results.json",
        {
            "checkpoint_paths": checkpoint_paths,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_configs": configs,
            "provenance_violations": violations,
            "formal_results_verified": not is_unverified,
            "run_name": run_name,
            "sample_manifest_sha256": manifest_sha256,
            "sample_manifest_rows": len(sample_manifest),
            "evaluation": {
                "wildrf_root": str(wildrf_root),
                "platforms": args.platforms,
                "profile": args.profile,
                "corruptions": args.corruptions,
                "seed": args.seed,
                "aggregation": (
                    "Accuracy is computed per platform and severity; platforms are macro-averaged, "
                    "then severities are arithmetic-mean averaged. Box/radial is the arithmetic mean "
                    "of the independently evaluated box and radial summaries."
                ),
                "jpeg_scope_note": (
                    "Strict means unseen as an explicit synthetic augmentation. Source files may "
                    "already be JPEG encoded."
                ),
                "severity_scope_note": (
                    "Blur strengths follow the legacy Table 9 parameterization but are applied "
                    "after the checkpoint's resize/center-crop. Noise and JPEG severities are a "
                    "new rebuttal protocol and are explicitly listed in corruptions.py."
                ),
                "transform_config": {
                    "resize_size": transform_config.resize_size,
                    "crop_size": transform_config.crop_size,
                    "mean": list(transform_config.mean),
                    "std": list(transform_config.std),
                },
            },
            "per_dataset_severity": per_dataset_rows,
            "severity_macro": severity_rows,
            "corruption_summary": corruption_rows,
            "paper_table": table_rows,
        },
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
