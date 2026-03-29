import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blur_generalization_suite.common import (
    all_reduce_tensor,
    barrier,
    cleanup_distributed,
    count_trainable_parameters,
    ensure_dir,
    extract_trainable_state_dict,
    parse_distributed_env,
    save_json,
    set_seed,
    setup_distributed,
    setup_logging,
)
from blur_generalization_suite.data_utils import BinaryFolderDataset, build_train_transform
from blur_generalization_suite.model_zoo import (
    DEFAULT_LORA_BACKBONES,
    LORA_BACKBONE_SPECS,
    FocalLoss,
    create_lora_model_from_config,
    normalize_lora_model_family,
    resolve_preprocess_config,
    serialize_transform_config,
)


DEFAULT_TRAIN_ROOT = "/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4"
DEFAULT_CCMBA_DATA_DIR = "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train"


def normalize_model_family(model_family: str) -> str:
    return normalize_lora_model_family(model_family)


def experiment_name(args: argparse.Namespace) -> str:
    backbone_name = Path(args.backbone_path).name
    blur_prob_tag = str(args.blur_prob).replace(".", "")
    return f"{args.model_family}_{backbone_name}_blur{blur_prob_tag}"


def _safe_percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _build_epoch_metrics(total_loss: float, total_correct: float, total_samples: float, real_correct: float, real_total: float, fake_correct: float, fake_total: float, divisor: float) -> dict:
    avg_loss = total_loss / max(divisor, 1.0)
    accuracy = _safe_percent(total_correct, total_samples)
    real_accuracy = _safe_percent(real_correct, real_total)
    fake_accuracy = _safe_percent(fake_correct, fake_total)
    balanced_accuracy = 0.5 * (real_accuracy + fake_accuracy)
    balanced_accuracy_half_gap = abs(fake_accuracy - real_accuracy) / 2.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_half_gap": balanced_accuracy_half_gap,
        "total_samples": int(total_samples),
        "real_total": int(real_total),
        "fake_total": int(fake_total),
    }


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, world_size):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    real_correct = 0.0
    real_total = 0.0
    fake_correct = 0.0
    fake_total = 0.0
    total_steps = 0.0

    for batch_idx, (images, labels, image_names, categories) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        blurred_images = []
        for img_tensor, img_name, label, category in zip(images, image_names, labels, categories):
            is_real = label.item() == 0
            blurred_tensor, _ = train_loader.dataset.apply_blur_augmentation(img_tensor, img_name, category, is_real)
            blurred_images.append(blurred_tensor.to(device, non_blocking=True))
        blurred_images = torch.stack(blurred_images)

        with autocast(enabled=device.type == "cuda"):
            # Keep the forward pass on the DDP wrapper so reducer hooks stay in sync.
            logits = model(blurred_images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        real_mask = labels == 0
        fake_mask = labels == 1

        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
        real_total += real_mask.sum().item()
        fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
        fake_total += fake_mask.sum().item()
        total_steps += 1.0

        if rank == 0 and (batch_idx + 1) % 10 == 0:
            metrics = _build_epoch_metrics(
                total_loss=total_loss,
                total_correct=total_correct,
                total_samples=total_samples,
                real_correct=real_correct,
                real_total=real_total,
                fake_correct=fake_correct,
                fake_total=fake_total,
                divisor=batch_idx + 1,
            )
            print(
                f"[Train] Batch {batch_idx + 1:04d}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | BACC: {metrics['balanced_accuracy']:.2f}%"
            )

    if world_size > 1:
        metrics_tensor = torch.tensor(
            [
                total_loss,
                total_correct,
                total_samples,
                real_correct,
                real_total,
                fake_correct,
                fake_total,
                total_steps,
            ],
            device=device,
            dtype=torch.float32,
        )
        all_reduce_tensor(metrics_tensor, world_size)
        (
            total_loss,
            total_correct,
            total_samples,
            real_correct,
            real_total,
            fake_correct,
            fake_total,
            total_steps,
        ) = metrics_tensor.tolist()

    return _build_epoch_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_samples=total_samples,
        real_correct=real_correct,
        real_total=real_total,
        fake_correct=fake_correct,
        fake_total=fake_total,
        divisor=total_steps,
    )


def build_checkpoint(model, optimizer, scheduler, scaler, epoch: int, best_acc: float, best_epoch: int, history: dict, trainable_stats: dict, model_config: dict, args: argparse.Namespace, epoch_metrics: dict) -> dict:
    return {
        "epoch": epoch,
        "trainable_state_dict": extract_trainable_state_dict(model.module),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "last_acc": epoch_metrics["accuracy"],
        "last_bacc": epoch_metrics["balanced_accuracy"],
        "history": history,
        "trainable_parameter_stats": trainable_stats,
        "lora_modules": list(model.module.lora_modules),
        "config": {
            **model_config,
            "train_root": args.train_root,
            "ccmba_data_dir": args.ccmba_data_dir,
            "blur_prob": args.blur_prob,
            "blur_mode": args.blur_mode,
            "blur_type": args.blur_type,
            "blur_strength_range": [args.blur_min, args.blur_max],
            "mixed_mode_ratio": args.mixed_mode_ratio,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "report_checkpoint": args.report_checkpoint,
        },
    }


def main_distributed(rank: int, local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_distributed(rank, world_size, local_rank=local_rank)
    setup_logging(rank)
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    backbone_spec = LORA_BACKBONE_SPECS[args.model_family]
    preprocess_config = resolve_preprocess_config(
        model_family=args.model_family,
        backbone_path=args.backbone_path,
        local_files_only=args.local_files_only,
        resize_override=args.resize_size,
        crop_override=args.crop_size,
    )
    train_transform = build_train_transform(preprocess_config)

    train_dataset = BinaryFolderDataset(
        root_folder=args.train_root,
        transform=train_transform,
        blur_prob=args.blur_prob,
        blur_strength_range=(args.blur_min, args.blur_max),
        blur_mode=args.blur_mode,
        blur_type=args.blur_type,
        mixed_mode_ratio=args.mixed_mode_ratio,
        ccmba_data_dir=args.ccmba_data_dir,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model_config = {
        "model_family": args.model_family,
        "backbone_path": args.backbone_path,
        "backbone_repo_id": backbone_spec.repo_id,
        "backbone_local_dir": backbone_spec.local_dir,
        "backbone_loader_backend": backbone_spec.loader_backend,
        "num_classes": 2,
        "projection_dim": args.projection_dim,
        "classifier_dropout": args.classifier_dropout,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": args.lora_targets,
        "local_files_only": args.local_files_only,
        "preprocess_config": serialize_transform_config(preprocess_config),
    }
    model = create_lora_model_from_config(model_config, device=device)
    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    trainable_stats = count_trainable_parameters(model.module)
    if rank == 0:
        print("=" * 70)
        print("FIXED-BACKBONE LORA TRAINING")
        print(f"Model family: {args.model_family}")
        print(f"Backbone path: {args.backbone_path}")
        print(f"Backbone repo id: {backbone_spec.repo_id}")
        print(f"Train root: {args.train_root}")
        print(f"CCMBA data dir: {args.ccmba_data_dir}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Trainable params: {trainable_stats['trainable']} / {trainable_stats['total']}")
        print(f"LoRA modules: {len(model.module.lora_modules)}")
        print(f"Local files only: {args.local_files_only}")
        print(f"Report checkpoint: {args.report_checkpoint}")
        print("=" * 70)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    output_dir = ensure_dir(Path(args.output_dir) / experiment_name(args))
    best_acc = float("-inf")
    best_epoch = -1
    best_bacc = 0.0
    last_acc = 0.0
    last_bacc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_bacc": [],
        "train_real_acc": [],
        "train_fake_acc": [],
    }

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, world_size)
        scheduler.step()

        history["train_loss"].append(metrics["loss"])
        history["train_acc"].append(metrics["accuracy"])
        history["train_bacc"].append(metrics["balanced_accuracy"])
        history["train_real_acc"].append(metrics["real_accuracy"])
        history["train_fake_acc"].append(metrics["fake_accuracy"])

        improved_best = metrics["accuracy"] >= best_acc
        if improved_best:
            best_acc = metrics["accuracy"]
            best_bacc = metrics["balanced_accuracy"]
            best_epoch = epoch
        last_acc = metrics["accuracy"]
        last_bacc = metrics["balanced_accuracy"]

        if rank == 0:
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | BACC: {metrics['balanced_accuracy']:.2f}% | "
                f"Real: {metrics['real_accuracy']:.2f}% | Fake: {metrics['fake_accuracy']:.2f}% | Time: {elapsed:.1f}s"
            )

            checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_acc=best_acc,
                best_epoch=best_epoch,
                history=history,
                trainable_stats=trainable_stats,
                model_config=model_config,
                args=args,
                epoch_metrics=metrics,
            )
            torch.save(checkpoint, output_dir / "latest_lora_model.pth")
            if epoch == args.epochs - 1:
                torch.save(checkpoint, output_dir / "last_lora_model.pth")
            if improved_best:
                torch.save(checkpoint, output_dir / "best_lora_model.pth")
            if args.report_checkpoint == "last" or improved_best:
                torch.save(checkpoint, output_dir / "selected_lora_model.pth")

        barrier(world_size)

    if rank == 0:
        report_model_path = output_dir / ("best_lora_model.pth" if args.report_checkpoint == "best" else "last_lora_model.pth")
        save_json(
            output_dir / "training_history.json",
            {
                "history": history,
                "best_acc": best_acc,
                "best_bacc": best_bacc,
                "best_epoch": best_epoch,
                "last_acc": last_acc,
                "last_bacc": last_bacc,
                "last_epoch": args.epochs - 1,
                "report_checkpoint": args.report_checkpoint,
                "report_model_path": str(report_model_path),
                "selected_model_path": str(output_dir / "selected_lora_model.pth"),
                "trainable_parameter_stats": trainable_stats,
                "lora_modules": list(model.module.lora_modules),
                "config": {
                    **model_config,
                    "train_root": args.train_root,
                    "ccmba_data_dir": args.ccmba_data_dir,
                    "blur_prob": args.blur_prob,
                    "blur_mode": args.blur_mode,
                    "blur_type": args.blur_type,
                    "blur_strength_range": [args.blur_min, args.blur_max],
                    "mixed_mode_ratio": args.mixed_mode_ratio,
                    "report_checkpoint": args.report_checkpoint,
                },
            },
        )
        print(f"Training artifacts saved to: {output_dir}")
        print(f"Recommended report model: {report_model_path}")

    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-backbone LoRA training with blur augmentation.")
    parser.add_argument("--model-family", choices=["clip_lora", "eva_giant_lora", "eva02_large_lora", "vit_large_lora"], required=True)
    parser.add_argument("--backbone-path", type=str, default=None)
    parser.add_argument("--train-root", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--ccmba-data-dir", type=str, default=DEFAULT_CCMBA_DATA_DIR)
    parser.add_argument("--blur-mode", choices=["no_blur", "global", "ccmba", "mixed"], default="mixed")
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-prob", type=float, default=0.1)
    parser.add_argument("--blur-min", type=float, default=0.1)
    parser.add_argument("--blur-max", type=float, default=0.3)
    parser.add_argument("--mixed-mode-ratio", type=float, default=0.5)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="*", default=None)
    parser.add_argument("--focal-alpha", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--report-checkpoint", choices=["best", "last"], default="best", help="Choose whether this experiment should report best-accuracy or last-epoch results.")
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/lora_train")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true", help="Load CLIP/EVA checkpoints from local files only.")
    parser.add_argument("--allow-remote-backbone", dest="local_files_only", action="store_false", help="Allow remote Hugging Face resolution when a local snapshot is unavailable.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()
    args.model_family = normalize_model_family(args.model_family)
    if args.backbone_path is None:
        args.backbone_path = DEFAULT_LORA_BACKBONES[args.model_family]
    return args


if __name__ == "__main__":
    arguments = parse_args()
    rank, world_size, local_rank = parse_distributed_env()
    try:
        main_distributed(rank, local_rank, world_size, arguments)
    finally:
        cleanup_distributed()



