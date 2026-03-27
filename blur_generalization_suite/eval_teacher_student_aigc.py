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
from blur_generalization_suite.data_utils import MultiTestDataset, TransformConfig, build_eval_transform, validate_dataset
from blur_generalization_suite.dataset_configs import select_datasets
from blur_generalization_suite.model_zoo import TeacherStudentEvalWrapper, TeacherStudentNetwork


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    wrapper = TeacherStudentEvalWrapper(network, branch=branch)
    wrapper.eval()
    return wrapper, checkpoint, config, missing, unexpected


def evaluate_model(model, test_loader, device, blur_mode: str):
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_idx, (images, batch_labels, _) in enumerate(test_loader):
            if blur_mode == "no_blur":
                processed = images.to(device, non_blocking=True)
            else:
                blurred = []
                for image in images:
                    blurred_tensor, _ = test_loader.dataset.apply_blur_augmentation(image)
                    blurred.append(blurred_tensor)
                processed = torch.stack(blurred).to(device, non_blocking=True)

            logits = model(processed)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{blur_mode}] Batch {batch_idx + 1}/{len(test_loader)}")

    return compute_binary_metrics(labels, predictions)


def write_summary_csv(path: Path, results: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "blur_mode", "accuracy", "precision", "recall", "f1_score", "total_samples"])
        for dataset_name, dataset_results in results.items():
            for blur_mode, metrics in dataset_results.items():
                writer.writerow(
                    [
                        dataset_name,
                        blur_mode,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1_score"],
                        metrics["total_samples"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher/student DINOv3 checkpoints on AIGCBenchmark.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--branch", choices=["teacher", "student"], default="student")
    parser.add_argument("--dataset-group", type=str, default="aigc_benchmark")
    parser.add_argument("--blur-mode", choices=["no_blur", "global", "both"], default="both")
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-min", type=float, default=0.1)
    parser.add_argument("--blur-max", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    model, checkpoint, config, missing, unexpected = load_teacher_student_model(args.model_path, args.branch, DEVICE)
    transform_dict = config.get("transform_config", {"resize_size": 512, "crop_size": 448, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    transform_config = TransformConfig(
        resize_size=int(transform_dict["resize_size"]),
        crop_size=int(transform_dict["crop_size"]),
        mean=tuple(transform_dict["mean"]),
        std=tuple(transform_dict["std"]),
    )
    eval_transform = build_eval_transform(transform_config)

    print("=" * 70)
    print("TEACHER-STUDENT AIGCBENCHMARK EVALUATION")
    print(f"Model path: {args.model_path}")
    print(f"Branch: {args.branch}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print("=" * 70)

    datasets_to_test = select_datasets(args.dataset_group)
    test_modes = ["no_blur", "global"] if args.blur_mode == "both" else [args.blur_mode]
    all_results = {}

    for dataset_name, dataset_config in datasets_to_test.items():
        if not validate_dataset(dataset_config):
            print(f"Skip {dataset_name}: dataset path not available")
            continue
        print(f"\nTesting dataset: {dataset_name}")
        dataset = MultiTestDataset(
            dataset_config,
            transform=eval_transform,
            blur_strength_range=(args.blur_min, args.blur_max),
            blur_type=args.blur_type,
        )
        if len(dataset) == 0:
            print(f"Skip {dataset_name}: dataset is empty")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        dataset_results = {}
        for mode in test_modes:
            start = time.time()
            metrics = evaluate_model(model, loader, DEVICE, mode)
            metrics["dataset_name"] = dataset_name
            metrics["blur_mode"] = mode
            metrics["time_seconds"] = time.time() - start
            dataset_results[mode] = metrics
            print(f"  {mode}: acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

        all_results[dataset_name] = dataset_results

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).resolve().parent / f"aigc_eval_{args.branch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(
        output_dir / f"aigc_eval_{timestamp}.json",
        {
            "model_path": args.model_path,
            "branch": args.branch,
            "checkpoint_config": config,
            "eval_config": {
                "dataset_group": args.dataset_group,
                "blur_mode": args.blur_mode,
                "blur_type": args.blur_type,
                "blur_strength_range": [args.blur_min, args.blur_max],
                "batch_size": args.batch_size,
            },
            "results": all_results,
        },
    )
    write_summary_csv(output_dir / f"aigc_eval_summary_{timestamp}.csv", all_results)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
