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
    parse_distributed_env,
    save_json,
    set_seed,
    setup_distributed,
    setup_logging,
)
from blur_generalization_suite.data_utils import BinaryFolderDataset, build_train_transform
from blur_generalization_suite.model_zoo import (
    DEFAULT_LORA_BACKBONES,
    FocalLoss,
    create_lora_model_from_config,
    resolve_preprocess_config,
    serialize_transform_config,
)


DEFAULT_TRAIN_ROOT = "/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4"
DEFAULT_CCMBA_DATA_DIR = "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train"


def experiment_name(args: argparse.Namespace) -> str:
    backbone_name = Path(args.backbone_path).name
    blur_prob_tag = str(args.blur_prob).replace(".", "")
    return f"{args.model_family}_{backbone_name}_blur{blur_prob_tag}"


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, world_size):
    actual_model = model.module if hasattr(model, "module") else model
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0

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
            logits = actual_model(blurred_images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        if rank == 0 and (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1.0)
            print(
                f"[Train] Batch {batch_idx + 1:04d}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%"
            )

    if world_size > 1:
        loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float32)
        correct_tensor = torch.tensor(total_correct, device=device, dtype=torch.float32)
        samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float32)
        all_reduce_tensor(loss_tensor, world_size)
        all_reduce_tensor(correct_tensor, world_size)
        all_reduce_tensor(samples_tensor, world_size)
        total_loss = loss_tensor.item()
        total_correct = correct_tensor.item()
        total_samples = samples_tensor.item()

    avg_loss = total_loss / max(len(train_loader) * world_size, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1.0)
    return avg_loss, avg_acc


def main_distributed(rank: int, local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_distributed(rank, world_size)
    setup_logging(rank)
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

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
        find_unused_parameters=True,
    )

    trainable_stats = count_trainable_parameters(model.module)
    if rank == 0:
        print("=" * 70)
        print("FIXED-BACKBONE LORA TRAINING")
        print(f"Model family: {args.model_family}")
        print(f"Backbone path: {args.backbone_path}")
        print(f"Train root: {args.train_root}")
        print(f"CCMBA data dir: {args.ccmba_data_dir}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Trainable params: {trainable_stats['trainable']} / {trainable_stats['total']}")
        print("=" * 70)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    output_dir = ensure_dir(Path(args.output_dir) / experiment_name(args))
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, world_size)
        scheduler.step()

        history["train_loss"].append(loss)
        history["train_acc"].append(acc)

        if rank == 0:
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s")

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_acc": max(best_acc, acc),
                "history": history,
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
                },
            }
            torch.save(checkpoint, output_dir / "latest_lora_model.pth")
            if acc >= best_acc:
                best_acc = acc
                torch.save(checkpoint, output_dir / "best_lora_model.pth")

        barrier(world_size)

    if rank == 0:
        save_json(
            output_dir / "training_history.json",
            {
                "history": history,
                "best_acc": best_acc,
                "config": {
                    **model_config,
                    "train_root": args.train_root,
                    "ccmba_data_dir": args.ccmba_data_dir,
                    "blur_prob": args.blur_prob,
                    "blur_mode": args.blur_mode,
                    "blur_type": args.blur_type,
                    "blur_strength_range": [args.blur_min, args.blur_max],
                    "mixed_mode_ratio": args.mixed_mode_ratio,
                },
            },
        )
        print(f"Training artifacts saved to: {output_dir}")

    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-backbone LoRA training with blur augmentation.")
    parser.add_argument("--model-family", choices=["clip_lora", "vit_large_lora"], required=True)
    parser.add_argument("--backbone-path", type=str, default=None)
    parser.add_argument("--train-root", type=str, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--ccmba-data-dir", type=str, default=DEFAULT_CCMBA_DATA_DIR)
    parser.add_argument("--blur-mode", choices=["no_blur", "global", "ccmba", "mixed"], default="mixed")
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-prob", type=float, default=0.2)
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
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/lora_train")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
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
