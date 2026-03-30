"""
Batch evaluation script for three model types:
  1. CLIP ViT-bigG + LoRA
  2. EVA02-Large + LoRA
  3. Teacher-Student (student branch)

Evaluates images from a benchmark directory containing 'real/' and 'fake/'
sub-directories. All images under 'real' are labeled 0, under 'fake' labeled 1.

Optional Wiener deblur preprocessing can be applied before inference.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blur_generalization_suite.common import compute_binary_metrics, load_checkpoint_state, save_json
from blur_generalization_suite.data_utils import TransformConfig, build_eval_transform
from blur_generalization_suite.model_zoo import (
    TeacherStudentEvalWrapper,
    TeacherStudentNetwork,
    create_lora_model_from_config,
)


# ─── Wiener deconvolution helpers ────────────────────────────────────────────

def _make_motion_kernel(length: int = 15, angle: float = 0.0) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= kernel.sum() + 1e-12
    return kernel


def wiener_deconvolve_channel(channel: np.ndarray, psf: np.ndarray, snr: float = 30.0) -> np.ndarray:
    h, w = channel.shape
    psf_pad = np.zeros_like(channel)
    ph, pw = psf.shape
    psf_pad[:ph, :pw] = psf
    psf_pad = np.roll(psf_pad, -ph // 2, axis=0)
    psf_pad = np.roll(psf_pad, -pw // 2, axis=1)
    G = np.fft.fft2(channel)
    H = np.fft.fft2(psf_pad)
    H_conj = np.conj(H)
    H_mag2 = np.abs(H) ** 2
    W = H_conj / (H_mag2 + 1.0 / snr)
    restored = np.fft.ifft2(W * G)
    restored = np.real(restored)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def wiener_deblur_pil(pil_img: Image.Image,
                      psf_length: int = 15,
                      psf_angle: float = 0.0,
                      snr: float = 30.0) -> Image.Image:
    try:
        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        psf = _make_motion_kernel(psf_length, psf_angle)
        channels = []
        for c in range(3):
            restored = wiener_deconvolve_channel(img_np[:, :, c], psf, snr=snr)
            channels.append(restored)
        restored_np = np.stack(channels, axis=2)
        restored_uint8 = (restored_np * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(restored_uint8, mode="RGB")
    except Exception as e:
        print(f"  [wiener] deblur failed ({e}), using original image.")
        return pil_img


# ─── Dataset ─────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def collect_images(root_dir: str, label: int) -> List[Tuple[str, int]]:
    samples = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                samples.append((os.path.join(dirpath, fname), label))
    return samples


class EvalDataset(Dataset):
    """
    Loads images, applies optional Wiener deblur, then the model's
    eval transform.  Returns [3, H, W] tensors.
    """
    def __init__(self, samples: list, transform: transforms.Compose,
                 use_deblur: bool = True,
                 psf_length: int = 15, psf_angle: float = 0.0,
                 snr: float = 30.0):
        self.samples = samples
        self.transform = transform
        self.use_deblur = use_deblur
        self.psf_length = psf_length
        self.psf_angle = psf_angle
        self.snr = snr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert('RGB')
            if self.use_deblur:
                pil_img = wiener_deblur_pil(pil_img,
                                            psf_length=self.psf_length,
                                            psf_angle=self.psf_angle,
                                            snr=self.snr)
            tensor = self.transform(pil_img)  # [3, H, W]
            return tensor, label, True
        except Exception as e:
            print(f"  [WARN] Error processing {img_path}: {e}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, label, False


# ─── Model loading ───────────────────────────────────────────────────────────

def load_lora_model(model_path: str, device: torch.device):
    """Load a LoRA model (CLIP or EVA02) from checkpoint."""
    checkpoint, state_dict = load_checkpoint_state(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint at {model_path} is missing 'config'; cannot rebuild LoRA model.")
    model = create_lora_model_from_config(config, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Build transform from checkpoint config
    preprocess_dict = config.get("preprocess_config", {})
    if preprocess_dict:
        transform_config = TransformConfig(**preprocess_dict)
    else:
        transform_config = TransformConfig()  # default 224x224
    eval_transform = build_eval_transform(transform_config)
    model_family = config.get("model_family", "unknown")
    print(f"  Loaded LoRA model: family={model_family}")
    print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return model, eval_transform, config


def load_teacher_student_model(model_path: str, device: torch.device):
    """Load a teacher-student model and return the student branch wrapper."""
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
    wrapper = TeacherStudentEvalWrapper(network, branch="student")
    wrapper.eval()
    # Build transform from checkpoint config
    transform_dict = config.get("transform_config", {
        "resize_size": 512, "crop_size": 448,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    })
    transform_config = TransformConfig(
        resize_size=int(transform_dict["resize_size"]),
        crop_size=int(transform_dict["crop_size"]),
        mean=tuple(transform_dict["mean"]),
        std=tuple(transform_dict["std"]),
    )
    eval_transform = build_eval_transform(transform_config)
    backbone_name = Path(config.get("dinov3_model_id", "")).name
    print(f"  Loaded Teacher-Student model (student branch): backbone={backbone_name}")
    print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return wrapper, eval_transform, config


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_single_model(model, loader, device):
    """Run inference and compute metrics."""
    all_preds = []
    all_labels = []
    errors = 0

    with torch.no_grad():
        for batch, labels_batch, ok_batch in tqdm(loader, desc="  Evaluating", dynamic_ncols=True):
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)  # [B, 2]
            preds = logits.argmax(dim=1).cpu()

            for pred, label, ok in zip(preds.tolist(), labels_batch.tolist(), ok_batch.tolist()):
                if not ok:
                    errors += 1
                    continue
                all_preds.append(pred)
                all_labels.append(label)

    if not all_labels:
        return {"error": "No valid samples processed", "errors": errors}

    metrics = compute_binary_metrics(all_labels, all_preds)
    metrics["errors"] = errors
    return metrics


def print_metrics(name: str, metrics: dict, data_root: str):
    print(f"\n{'=' * 70}")
    print(f"  [{name}] Results on: {data_root}")
    print(f"{'=' * 70}")
    print(f"  Accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy : {metrics['bacc'] * 100:.2f}%")
    print(f"  Real accuracy     : {metrics['real_accuracy'] * 100:.2f}%  ({metrics['real_total']} images)")
    print(f"  Fake accuracy     : {metrics['fake_accuracy'] * 100:.2f}%  ({metrics['fake_total']} images)")
    print(f"  Precision         : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall            : {metrics['recall'] * 100:.2f}%")
    print(f"  F1                : {metrics['f1_score'] * 100:.2f}%")
    if metrics.get("errors", 0) > 0:
        print(f"  Skipped (errors)  : {metrics['errors']}")
    print(f"{'=' * 70}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluation of CLIP-LoRA / EVA02-LoRA / Teacher-Student models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_root',
        default='/home/work/shenjialiang/rebuttal/own_benchmark/own_final',
        type=str,
        help='Root folder containing real/ and fake/ sub-directories.',
    )
    # Model paths (specify one or more)
    parser.add_argument(
        '--clip_lora_path',
        default=None,
        type=str,
        help='Path to the best CLIP ViT-bigG LoRA checkpoint (.pth).',
    )
    parser.add_argument(
        '--eva02_lora_path',
        default=None,
        type=str,
        help='Path to the best EVA02-Large LoRA checkpoint (.pth).',
    )
    parser.add_argument(
        '--student_model_path',
        default=None,
        type=str,
        help='Path to the best student model checkpoint (.pth).',
    )
    # Wiener deblur options
    parser.add_argument('--use_deblur', action='store_true', default=False,
                        help='Apply Wiener deblur preprocessing before inference.')
    parser.add_argument('--psf_length', default=15, type=int,
                        help='Motion-blur PSF length in pixels.')
    parser.add_argument('--psf_angle', default=0.0, type=float,
                        help='Motion-blur direction in degrees.')
    parser.add_argument('--snr', default=30.0, type=float,
                        help='Signal-to-noise ratio for Wiener deconvolution.')
    # Runtime options
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help='Device to run inference on.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of DataLoader worker processes.')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save evaluation results.')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Collect samples ──────────────────────────────────────────────────────
    benchmark_root = args.data_root
    real_dir = os.path.join(benchmark_root, '0_real')
    fake_dir = os.path.join(benchmark_root, '1_fake')
    assert os.path.isdir(real_dir), f"Expected '0_real' folder at {real_dir}"
    assert os.path.isdir(fake_dir), f"Expected '1_fake' folder at {fake_dir}"

    real_samples = collect_images(real_dir, label=0)
    fake_samples = collect_images(fake_dir, label=1)
    all_samples = real_samples + fake_samples
    print(f"Found {len(real_samples)} real images and {len(fake_samples)} fake images.")

    # ── Determine which models to evaluate ───────────────────────────────────
    model_specs = []
    if args.clip_lora_path:
        model_specs.append(("CLIP_ViT-bigG_LoRA", "lora", args.clip_lora_path))
    if args.eva02_lora_path:
        model_specs.append(("EVA02-Large_LoRA", "lora", args.eva02_lora_path))
    if args.student_model_path:
        model_specs.append(("Teacher-Student_Student", "teacher_student", args.student_model_path))

    if not model_specs:
        print("ERROR: No model specified. Provide at least one of:")
        print("  --clip_lora_path, --eva02_lora_path, --student_model_path")
        sys.exit(1)

    # ── Evaluate each model ──────────────────────────────────────────────────
    all_results = {}

    for model_name, model_type, model_path in model_specs:
        print(f"\n{'#' * 70}")
        print(f"# Loading model: {model_name}")
        print(f"# Checkpoint:    {model_path}")
        print(f"{'#' * 70}")

        # Load model and its transform
        if model_type == "lora":
            model, eval_transform, _ = load_lora_model(model_path, device)
        else:
            model, eval_transform, _ = load_teacher_student_model(model_path, device)

        # Build dataset with this model's transform
        dataset = EvalDataset(
            samples=all_samples,
            transform=eval_transform,
            use_deblur=args.use_deblur,
            psf_length=args.psf_length,
            psf_angle=args.psf_angle,
            snr=args.snr,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Evaluate
        start_time = time.time()
        metrics = evaluate_single_model(model, loader, device)
        elapsed = time.time() - start_time
        metrics["time_seconds"] = elapsed
        metrics["model_name"] = model_name
        metrics["model_path"] = model_path
        metrics["model_type"] = model_type

        print_metrics(model_name, metrics, benchmark_root)
        all_results[model_name] = metrics

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(benchmark_root) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    save_json(
        output_dir / f"own_batch_eval_{timestamp}.json",
        {
            "data_root": benchmark_root,
            "use_deblur": args.use_deblur,
            "psf_length": args.psf_length,
            "psf_angle": args.psf_angle,
            "snr": args.snr,
            "batch_size": args.batch_size,
            "results": all_results,
        },
    )

    # Save CSV summary
    csv_path = output_dir / f"own_batch_eval_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "model_type", "accuracy", "bacc",
            "real_accuracy", "fake_accuracy", "precision", "recall",
            "f1_score", "real_total", "fake_total", "errors", "time_seconds",
        ])
        for name, m in all_results.items():
            writer.writerow([
                name, m.get("model_type", ""),
                f"{m['accuracy']:.4f}", f"{m['bacc']:.4f}",
                f"{m['real_accuracy']:.4f}", f"{m['fake_accuracy']:.4f}",
                f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                m['real_total'], m['fake_total'],
                m.get('errors', 0), f"{m.get('time_seconds', 0):.1f}",
            ])

    print(f"\nResults saved to: {output_dir}")

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 90}")
    header = f"  {'Model':<30s} {'Acc':>8s} {'BAcc':>8s} {'Real':>8s} {'Fake':>8s} {'F1':>8s}"
    print(header)
    print(f"  {'-' * 78}")
    for name, m in all_results.items():
        print(f"  {name:<30s} {m['accuracy']*100:>7.2f}% {m['bacc']*100:>7.2f}% "
              f"{m['real_accuracy']*100:>7.2f}% {m['fake_accuracy']*100:>7.2f}% "
              f"{m['f1_score']*100:>7.2f}%")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
