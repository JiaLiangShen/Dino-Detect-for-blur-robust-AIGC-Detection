from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ..config import EvalConfig
from ..data import BlurAugmentor, EvalDatasetConfigRecord, EvaluationDataset, build_eval_transform
from ..models import StudentWrapper, TeacherStudentModel, TeacherWrapper
from ..utils.checkpoint import load_checkpoint
from ..utils.misc import ensure_dir, save_json, set_seed, timestamp


def _build_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_models(config: EvalConfig, device: torch.device):
    teacher_model = TeacherStudentModel(
        backbone_path=config.backbone_path,
        num_classes=config.num_classes,
        projection_dim=config.projection_dim,
        adapter_layers=config.adapter_layers,
        dropout=config.dropout,
    )
    student_model = TeacherStudentModel(
        backbone_path=config.backbone_path,
        num_classes=config.num_classes,
        projection_dim=config.projection_dim,
        adapter_layers=config.adapter_layers,
        dropout=config.dropout,
    )

    teacher_checkpoint = load_checkpoint(config.teacher_checkpoint, map_location="cpu")
    student_checkpoint = load_checkpoint(config.student_checkpoint, map_location="cpu")
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"], strict=False)
    student_model.load_state_dict(student_checkpoint["model_state_dict"], strict=False)

    teacher_wrapper = TeacherWrapper(teacher_model.to(device).eval())
    student_wrapper = StudentWrapper(student_model.to(device).eval())
    return teacher_wrapper, student_wrapper


def _compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": cm.tolist(),
    }


def _evaluate_single_mode(model, loader, blur_augmentor, device):
    predictions: list[int] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].tolist()
            image_names = batch["image_name"]
            categories = batch["category"]

            processed = []
            for image_tensor, label, image_name, category in zip(images, labels, image_names, categories):
                blurred, _ = blur_augmentor.apply(
                    image=image_tensor,
                    image_name=str(image_name),
                    category=str(category),
                    is_real=bool(label == 0),
                )
                processed.append(blurred)

            processed_batch = torch.stack(processed).to(device, non_blocking=True)
            _, logits = model(processed_batch)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels)

    return _compute_metrics(labels_all, predictions)


def run_evaluation(config: EvalConfig) -> None:
    set_seed(config.seed)
    device = _build_device()
    output_dir = ensure_dir(Path(config.output_dir) / config.experiment_name)
    results_dir = ensure_dir(output_dir / "results")
    transform = build_eval_transform(config.image_size, config.crop_size)

    teacher_model, student_model = _load_models(config=config, device=device)
    blur_modes = ["no_blur", "motion"] if config.blur_mode == "both" else [config.blur_mode]

    all_results: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "results_by_dataset": {},
    }

    for dataset_cfg in config.datasets:
        dataset_record = EvalDatasetConfigRecord(**asdict(dataset_cfg))
        dataset = EvaluationDataset(config=dataset_record, transform=transform)
        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dataset_results: dict[str, object] = {}

        for blur_mode in blur_modes:
            blur_augmentor = BlurAugmentor(
                mode=blur_mode,
                probability=0.0 if blur_mode in {"none", "no_blur"} else 1.0,
                strength_range=(float(config.blur_strength_range[0]), float(config.blur_strength_range[1])),
            )
            if config.model_mode in {"teacher", "both"}:
                teacher_metrics = _evaluate_single_mode(
                    model=teacher_model,
                    loader=loader,
                    blur_augmentor=blur_augmentor,
                    device=device,
                )
            else:
                teacher_metrics = None

            if config.model_mode in {"student", "both"}:
                student_metrics = _evaluate_single_mode(
                    model=student_model,
                    loader=loader,
                    blur_augmentor=blur_augmentor,
                    device=device,
                )
            else:
                student_metrics = None

            payload: dict[str, object] = {
                "num_samples": len(dataset),
                "blur_mode": blur_mode,
            }
            if teacher_metrics is not None:
                payload["teacher"] = teacher_metrics
            if student_metrics is not None:
                payload["student"] = student_metrics
            if teacher_metrics is not None and student_metrics is not None:
                diff = student_metrics["accuracy"] - teacher_metrics["accuracy"]
                payload["comparison"] = {
                    "accuracy_difference": float(diff),
                    "student_better": bool(diff > 0),
                }
            dataset_results[blur_mode] = payload

        all_results["results_by_dataset"][dataset_cfg.name] = dataset_results

    save_json(results_dir / f"evaluation_{timestamp()}.json", all_results)
