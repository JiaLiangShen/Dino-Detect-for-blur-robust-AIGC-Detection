import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blur_generalization_suite.common import save_json, set_seed
from blur_generalization_suite.data_utils import collect_image_paths, build_eval_transform, TransformConfig, apply_blur_to_tensor, load_image_safely
from blur_generalization_suite.model_zoo import DEFAULT_DINOV3_MODELS, DEFAULT_PREPROCESS


DEFAULT_MODEL_PATH = DEFAULT_DINOV3_MODELS["dinov3_vit7b"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_dino_transform(model_path: str, local_files_only: bool):
    fallback = DEFAULT_PREPROCESS["dinov3"]
    try:
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
        size = getattr(processor, "size", {})
        crop_size = getattr(processor, "crop_size", {})
        resize_value = size.get("shortest_edge", 512) if isinstance(size, dict) else int(size)
        crop_value = crop_size.get("height", 448) if isinstance(crop_size, dict) else int(crop_size)
        mean = tuple(float(x) for x in getattr(processor, "image_mean", fallback.mean))
        std = tuple(float(x) for x in getattr(processor, "image_std", fallback.std))
        config = TransformConfig(resize_size=resize_value, crop_size=crop_value, mean=mean, std=std)
    except Exception:
        config = fallback
    return build_eval_transform(config), config


def load_dinov3(model_path: str, local_files_only: bool):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE).eval()
    return model


def extract_patch_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, return_dict=True)
        num_register = getattr(model.config, "num_register_tokens", 0)
        patch_features = outputs.last_hidden_state[:, 1 + num_register :, :]
        patch_features = F.normalize(patch_features, dim=-1)
        return patch_features.squeeze(0)


def cosine_matrix_correlation(reference_features: torch.Tensor, blurred_features: torch.Tensor) -> float:
    reference_matrix = torch.matmul(reference_features, reference_features.t()).detach().cpu().numpy().flatten()
    blurred_matrix = torch.matmul(blurred_features, blurred_features.t()).detach().cpu().numpy().flatten()
    correlation = np.corrcoef(reference_matrix, blurred_matrix)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def plot_curve(strengths, averages, output_path: Path, blur_type: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(strengths, averages, marker="o", linewidth=2.0, color="#1f77b4")
    plt.ylim(0.0, 1.05)
    plt.xlim(min(strengths), max(strengths))
    plt.grid(True, alpha=0.3)
    plt.xlabel(f"{blur_type.title()} Blur Strength")
    plt.ylabel("Consistency vs. Original")
    plt.title("DINOv3 Feature Consistency Across Blur Strengths")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 blur-strength feature consistency analysis.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--min-blur", type=float, default=0.0)
    parser.add_argument("--max-blur", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/dinov3_consistency")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    transform, transform_config = resolve_dino_transform(args.model_path, args.local_files_only)
    model = load_dinov3(args.model_path, args.local_files_only)
    image_paths = collect_image_paths(args.data_root, max_images=args.max_images)
    strengths = [round(x, 3) for x in np.arange(args.min_blur, args.max_blur + args.step, args.step)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_results = {}
    averages = {f"{strength:.3f}": [] for strength in strengths}

    print("=" * 70)
    print("DINOv3 BLUR CONSISTENCY ANALYSIS")
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Image count: {len(image_paths)}")
    print("=" * 70)

    for image_path in image_paths:
        image = load_image_safely(image_path)
        if image is None:
            continue
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        reference_features = extract_patch_features(model, image_tensor)

        image_scores = {}
        for strength in strengths:
            if strength == 0.0:
                consistency = 1.0
            else:
                blurred_tensor = apply_blur_to_tensor(image_tensor, args.blur_type, strength)
                blurred_features = extract_patch_features(model, blurred_tensor)
                consistency = cosine_matrix_correlation(reference_features, blurred_features)
            image_scores[f"{strength:.3f}"] = consistency
            averages[f"{strength:.3f}"].append(consistency)
        per_image_results[image_path.name] = image_scores

    average_results = {
        strength: float(np.mean(values)) if values else 0.0
        for strength, values in averages.items()
    }

    save_json(
        output_dir / "per_image_consistency.json",
        {
            "model_path": args.model_path,
            "data_root": args.data_root,
            "blur_type": args.blur_type,
            "transform_config": {
                "resize_size": transform_config.resize_size,
                "crop_size": transform_config.crop_size,
                "mean": list(transform_config.mean),
                "std": list(transform_config.std),
            },
            "results": per_image_results,
        },
    )
    save_json(
        output_dir / "average_consistency.json",
        {
            "model_path": args.model_path,
            "data_root": args.data_root,
            "blur_type": args.blur_type,
            "strengths": strengths,
            "average_consistency": average_results,
        },
    )
    plot_curve(
        strengths,
        [average_results[f"{strength:.3f}"] for strength in strengths],
        output_dir / "dinov3_consistency_curve.png",
        args.blur_type,
    )
    print(f"Consistency outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
