import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blur_generalization_suite.common import compute_binary_metrics, load_checkpoint_state, save_json
from blur_generalization_suite.data_utils import (
    MultiTestDataset,
    TransformConfig,
    apply_blur_to_normalized_tensor,
    apply_wiener_deblur_to_tensor,
    build_eval_transform,
    validate_dataset,
)
from blur_generalization_suite.dataset_configs import select_datasets
from blur_generalization_suite.model_zoo import TeacherStudentEvalWrapper, TeacherStudentNetwork, create_lora_model_from_config


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIPELINE_CHOICES = ["no_blur", "blur", "deblur", "blur_then_deblur"]


def resolve_dataset_config(args: argparse.Namespace) -> tuple[str, dict]:
    if args.real_folder and args.fake_folder:
        return "custom_benchmark", {
            "type": "simple",
            "real_folder": args.real_folder,
            "fake_folder": args.fake_folder,
        }
    datasets = select_datasets(args.dataset_name)
    return args.dataset_name, datasets[args.dataset_name]


def load_lora_model(model_path: str, device: torch.device):
    checkpoint, state_dict = load_checkpoint_state(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = create_lora_model_from_config(config, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    preprocess_config = TransformConfig(**config["preprocess_config"])
    metadata = {
        "model_name": config.get("model_family", "lora_model"),
        "model_type": "lora",
        "backbone_name": Path(config.get("backbone_path", "")).name,
        "report_checkpoint": config.get("report_checkpoint", "best"),
    }
    model.eval()
    return model, preprocess_config, metadata, missing, unexpected


def load_teacher_student_model(model_path: str, branch: str, device: torch.device):
    checkpoint, state_dict = load_checkpoint_state(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    network = TeacherStudentNetwork(
        dinov3_model_path=config["dinov3_model_id"],
        num_classes=2,
        projection_dim=int(config.get("projection_dim", 512)),
        local_files_only=bool(config.get("local_files_only", True)),
        device=device,
    )
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    model = TeacherStudentEvalWrapper(network, branch=branch)
    transform_dict = config.get("transform_config", {"resize_size": 512, "crop_size": 448, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    preprocess_config = TransformConfig(
        resize_size=int(transform_dict["resize_size"]),
        crop_size=int(transform_dict["crop_size"]),
        mean=tuple(transform_dict["mean"]),
        std=tuple(transform_dict["std"]),
    )
    metadata = {
        "model_name": f"dinov3_{branch}",
        "model_type": "teacher_student",
        "backbone_name": Path(config.get("dinov3_model_id", "")).name,
        "branch": branch,
    }
    model.eval()
    return model, preprocess_config, metadata, missing, unexpected


def build_model_entries(args: argparse.Namespace):
    entries = []
    if args.dinov3_student_path:
        entries.append(("dinov3_student", args.dinov3_student_path, "student"))
    if args.clip_lora_path:
        entries.append(("clip_lora", args.clip_lora_path, None))
    if args.eva_lora_path:
        entries.append(("eva_giant_lora", args.eva_lora_path, None))
    if not entries:
        raise ValueError("At least one model path must be provided.")
    return entries


def apply_pipeline(images: torch.Tensor, pipeline: str, blur_type: str, blur_strength: float, regularization: float, transform_config: TransformConfig) -> torch.Tensor:
    processed = images
    if pipeline in {"blur", "blur_then_deblur"}:
        processed = apply_blur_to_normalized_tensor(processed, blur_type, blur_strength, transform_config.mean, transform_config.std)
    if pipeline in {"deblur", "blur_then_deblur"}:
        processed = apply_wiener_deblur_to_tensor(
            processed,
            blur_type=blur_type,
            strength=blur_strength,
            regularization=regularization,
            mean=transform_config.mean,
            std=transform_config.std,
        )
    return processed


def evaluate_pipeline(model, loader, device, pipeline: str, blur_type: str, blur_strength: float, regularization: float, transform_config: TransformConfig):
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_idx, (images, batch_labels, _) in enumerate(loader):
            processed = apply_pipeline(images, pipeline, blur_type, blur_strength, regularization, transform_config)
            logits = model(processed.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
            if (batch_idx + 1) % 10 == 0:
                print(f"    [{pipeline}] Batch {batch_idx + 1}/{len(loader)}")
    return compute_binary_metrics(labels, predictions)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_name",
            "model_type",
            "backbone_name",
            "pipeline",
            "dataset",
            "accuracy",
            "bacc",
            "real_accuracy",
            "fake_accuracy",
            "balanced_accuracy_half_gap",
            "precision",
            "recall",
            "f1_score",
            "total_samples",
        ])
        for row in rows:
            writer.writerow([
                row["model_name"],
                row["model_type"],
                row["backbone_name"],
                row["pipeline"],
                row["dataset"],
                row["accuracy"],
                row["bacc"],
                row["real_accuracy"],
                row["fake_accuracy"],
                row["balanced_accuracy_half_gap"],
                row["precision"],
                row["recall"],
                row["f1_score"],
                row["total_samples"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deblur-then-detect comparison on own benchmark.")
    parser.add_argument("--dataset-name", type=str, default="own_benchmark")
    parser.add_argument("--real-folder", type=str, default=None)
    parser.add_argument("--fake-folder", type=str, default=None)
    parser.add_argument("--dinov3-student-path", type=str, default=None)
    parser.add_argument("--clip-lora-path", type=str, default=None)
    parser.add_argument("--eva-lora-path", type=str, default=None)
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINE_CHOICES, default=["no_blur", "blur", "blur_then_deblur"])
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-strength", type=float, default=0.3)
    parser.add_argument("--deblur-regularization", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/deblur_benchmark")
    args = parser.parse_args()

    dataset_name, dataset_config = resolve_dataset_config(args)
    if not validate_dataset(dataset_config):
        raise FileNotFoundError(f"Dataset path is not available for {dataset_name}: {dataset_config}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}
    summary_rows = []

    for entry_name, model_path, branch in build_model_entries(args):
        if branch is None:
            model, transform_config, metadata, missing, unexpected = load_lora_model(model_path, DEVICE)
        else:
            model, transform_config, metadata, missing, unexpected = load_teacher_student_model(model_path, branch, DEVICE)

        eval_transform = build_eval_transform(transform_config)
        dataset = MultiTestDataset(dataset_config, transform=eval_transform, blur_strength_range=(args.blur_strength, args.blur_strength), blur_type=args.blur_type)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("=" * 70)
        print(f"DEBLUR BENCHMARK EVAL | {metadata['model_name']}")
        print(f"Model path: {model_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        print("=" * 70)

        model_results = {}
        for pipeline in args.pipelines:
            start = time.time()
            metrics = evaluate_pipeline(
                model,
                loader,
                DEVICE,
                pipeline=pipeline,
                blur_type=args.blur_type,
                blur_strength=args.blur_strength,
                regularization=args.deblur_regularization,
                transform_config=transform_config,
            )
            metrics["time_seconds"] = time.time() - start
            model_results[pipeline] = metrics
            summary_rows.append({
                "model_name": metadata["model_name"],
                "model_type": metadata["model_type"],
                "backbone_name": metadata["backbone_name"],
                "pipeline": pipeline,
                "dataset": dataset_name,
                **metrics,
            })
            print(
                f"  {pipeline}: acc={metrics['accuracy']:.4f}, "
                f"bacc={metrics['bacc']:.4f}, f1={metrics['f1_score']:.4f}"
            )

        results[metadata["model_name"]] = {
            "metadata": metadata,
            "model_path": model_path,
            "results": model_results,
        }

    save_json(
        output_dir / f"deblur_benchmark_eval_{timestamp}.json",
        {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "eval_config": {
                "pipelines": args.pipelines,
                "blur_type": args.blur_type,
                "blur_strength": args.blur_strength,
                "deblur_regularization": args.deblur_regularization,
                "batch_size": args.batch_size,
            },
            "results": results,
        },
    )
    write_summary_csv(output_dir / f"deblur_benchmark_summary_{timestamp}.csv", summary_rows)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
