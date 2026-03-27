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
from blur_generalization_suite.dataset_configs import TABLE_EXPORT_SPECS, select_datasets
from blur_generalization_suite.model_zoo import create_lora_model_from_config


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device):
    checkpoint, state_dict = load_checkpoint_state(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError("Checkpoint is missing config; cannot rebuild LoRA model.")
    model = create_lora_model_from_config(config, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint, config, missing, unexpected


def evaluate_model(model, test_loader, device, blur_mode: str):
    predictions = []
    labels = []

    with torch.no_grad():
        for batch_idx, (images, batch_labels, _) in enumerate(test_loader):
            if blur_mode == "no_blur":
                processed = images.to(device, non_blocking=True)
            else:
                blurred_images = []
                for image in images:
                    blurred_tensor, _ = test_loader.dataset.apply_blur_augmentation(image)
                    blurred_images.append(blurred_tensor)
                processed = torch.stack(blurred_images).to(device, non_blocking=True)

            logits = model(processed)
            preds = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
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


def write_table_exports(output_dir: Path, results: dict) -> None:
    for filename, blur_mode, metric_name in TABLE_EXPORT_SPECS:
        path = output_dir / f"{filename}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", metric_name])
            for dataset_name, dataset_results in results.items():
                if blur_mode in dataset_results:
                    writer.writerow([dataset_name, dataset_results[blur_mode][metric_name]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset clean/blur evaluation for LoRA baselines.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-group", type=str, default="all", help="all, aigc_benchmark, or a single dataset name")
    parser.add_argument("--blur-mode", choices=["no_blur", "global", "both"], default="both")
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-min", type=float, default=0.1)
    parser.add_argument("--blur-max", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    model, checkpoint, config, missing, unexpected = load_model(args.model_path, DEVICE)
    preprocess_dict = config.get("preprocess_config", {})
    transform_config = TransformConfig(**preprocess_dict)
    test_transform = build_eval_transform(transform_config)

    print("=" * 70)
    print("LORA CROSS-DATASET EVALUATION")
    print(f"Model path: {args.model_path}")
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
            transform=test_transform,
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

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).resolve().parent / "cross_dataset_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_json(
        output_dir / f"cross_dataset_eval_{timestamp}.json",
        {
            "model_path": args.model_path,
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
    write_summary_csv(output_dir / f"cross_dataset_summary_{timestamp}.csv", all_results)
    write_table_exports(output_dir, all_results)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
