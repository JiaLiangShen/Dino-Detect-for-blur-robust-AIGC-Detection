import argparse
import sys
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
from blur_generalization_suite.data_utils import (
    BinaryFolderDataset,
    TransformConfig,
    apply_blur_to_tensor,
    build_strong_train_transform,
    build_train_transform,
)
from blur_generalization_suite.model_zoo import (
    DEFAULT_DINOV3_MODELS,
    DEFAULT_PREPROCESS,
    ImprovedTeacherStudentLoss,
    TeacherStudentNetwork,
)


DATA_PRESETS = {
    "original_motion": {
        "train_root": "/home/work/xueyunqi/11ar_datasets/extracted",
        "ccmba_data_dir": "/home/work/xueyunqi/11ar_datasets/progan_ccmba_train",
    },
    "sdv14": {
        "train_root": "/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4",
        "ccmba_data_dir": "/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44",
    },
}
TEMPERATURE = 0.07
ALPHA_DISTILL = 1.0
ALPHA_CLS = 1.0
ALPHA_FEATURE = 0.5


def resolve_data_paths(args: argparse.Namespace) -> argparse.Namespace:
    preset = DATA_PRESETS[args.data_preset]
    if args.train_root is None:
        args.train_root = preset["train_root"]
    if args.ccmba_data_dir is None:
        args.ccmba_data_dir = preset["ccmba_data_dir"]
    return args


def build_dino_transforms(args: argparse.Namespace):
    default_config = DEFAULT_PREPROCESS["dinov3"]
    transform_config = TransformConfig(
        resize_size=args.resize_size or default_config.resize_size,
        crop_size=args.crop_size or default_config.crop_size,
        mean=default_config.mean,
        std=default_config.std,
    )
    return build_train_transform(transform_config), build_strong_train_transform(transform_config), transform_config


def experiment_name(args: argparse.Namespace) -> str:
    backbone_name = Path(args.dinov3_model_id).name
    blur_prob_tag = str(args.blur_prob).replace(".", "")
    return f"{backbone_name}_{args.data_preset}_teacher_student_blur{blur_prob_tag}"


def build_common_config(args: argparse.Namespace, transform_config: TransformConfig) -> dict:
    return {
        "dinov3_model_id": args.dinov3_model_id,
        "projection_dim": args.projection_dim,
        "data_preset": args.data_preset,
        "train_root": args.train_root,
        "ccmba_data_dir": args.ccmba_data_dir,
        "blur_mode": args.blur_mode,
        "blur_type": args.blur_type,
        "blur_prob": args.blur_prob,
        "blur_strength_range": [args.blur_min, args.blur_max],
        "mixed_mode_ratio": args.mixed_mode_ratio,
        "enable_strong_aug": True,
        "transform_config": {
            "resize_size": transform_config.resize_size,
            "crop_size": transform_config.crop_size,
            "mean": list(transform_config.mean),
            "std": list(transform_config.std),
        },
        "default_eval_branch": "student",
        "local_files_only": args.local_files_only,
    }


def save_phase_checkpoint(path: Path, model, optimizer, scheduler, scaler, history: dict, config: dict, best_acc: float, epoch: int, phase: str) -> None:
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "best_acc": best_acc,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def train_teacher_phase(model, train_loader, optimizer, scheduler, scaler, device, rank, world_size, args, output_dir: Path, config: dict):
    criterion = ImprovedTeacherStudentLoss(
        temperature=TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        alpha_cls=ALPHA_CLS,
        alpha_feature=ALPHA_FEATURE,
        alpha_simclr=0.0,
    )
    actual_model = model.module if hasattr(model, "module") else model
    actual_model.freeze_teacher()
    actual_model.unfreeze_teacher_head()

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(args.teacher_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0

        for batch_idx, (images, labels, _, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                teacher_features, teacher_logits = actual_model.forward_teacher(images)
                losses = criterion(
                    student_features=teacher_features,
                    student_logits=teacher_logits,
                    labels=labels,
                    mode="teacher",
                )

            scaler.scale(losses["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses["total_loss"].item()
            total_correct += (teacher_logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            if rank == 0 and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = 100.0 * total_correct / max(total_samples, 1.0)
                print(
                    f"[Teacher] Epoch {epoch + 1}/{args.teacher_epochs} Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%"
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
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        scheduler.step()

        if rank == 0:
            print(f"Teacher epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%")
            latest_path = output_dir / "latest_teacher_model.pth"
            save_phase_checkpoint(latest_path, model, optimizer, scheduler, scaler, history, config, max(best_acc, avg_acc), epoch, "teacher")
            if avg_acc >= best_acc:
                best_acc = avg_acc
                save_phase_checkpoint(output_dir / "best_teacher_model.pth", model, optimizer, scheduler, scaler, history, config, best_acc, epoch, "teacher")

        barrier(world_size)

    return history, best_acc


def train_student_phase(model, train_loader, optimizer, scheduler, scaler, device, rank, world_size, args, output_dir: Path, config: dict):
    criterion = ImprovedTeacherStudentLoss(
        temperature=TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        alpha_cls=ALPHA_CLS,
        alpha_feature=ALPHA_FEATURE,
        alpha_simclr=args.alpha_simclr,
    )
    actual_model = model.module if hasattr(model, "module") else model
    actual_model.freeze_teacher()
    actual_model.unfreeze_student()

    history = {
        "train_total_loss": [],
        "train_cls_loss": [],
        "train_distill_loss": [],
        "train_feature_loss": [],
        "train_simclr_loss": [],
        "train_acc": [],
    }
    best_acc = 0.0

    for epoch in range(args.student_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        losses_accum = {"total": 0.0, "cls": 0.0, "distill": 0.0, "feature": 0.0, "simclr": 0.0}
        total_correct = 0.0
        total_samples = 0.0

        for batch_idx, (images, labels, image_names, categories) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast(enabled=device.type == "cuda"):
                    teacher_features, teacher_logits = actual_model.forward_teacher(images)

            student_inputs = []
            for img_tensor, img_name, label, category in zip(images, image_names, labels, categories):
                is_real = label.item() == 0
                blurred_tensor, _ = train_loader.dataset.apply_blur_augmentation(img_tensor, img_name, category, is_real)
                student_inputs.append(blurred_tensor.to(device, non_blocking=True))
            student_inputs = torch.stack(student_inputs)

            with autocast(enabled=device.type == "cuda"):
                student_features, student_logits = actual_model.forward_student(student_inputs)
                student_aug = []
                for img_tensor in student_inputs:
                    strength = torch.empty(1).uniform_(args.blur_min, args.blur_max).item()
                    aug_tensor = apply_blur_to_tensor(img_tensor.unsqueeze(0), args.blur_type, strength).squeeze(0)
                    student_aug.append(aug_tensor)
                student_aug = torch.stack(student_aug).to(device, non_blocking=True)
                student_features_aug, _ = actual_model.forward_student(student_aug)

                losses = criterion(
                    student_features=student_features,
                    student_logits=student_logits,
                    teacher_features=teacher_features,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mode="student",
                    student_features_aug=student_features_aug,
                )

            scaler.scale(losses["total_loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_accum["total"] += losses["total_loss"].item()
            losses_accum["cls"] += losses["cls_loss"].item()
            losses_accum["distill"] += losses["distill_loss"].item()
            losses_accum["feature"] += losses["feature_loss"].item()
            losses_accum["simclr"] += losses["simclr_loss"].item()
            total_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            if rank == 0 and (batch_idx + 1) % 10 == 0:
                avg_acc = 100.0 * total_correct / max(total_samples, 1.0)
                print(
                    f"[Student] Epoch {epoch + 1}/{args.student_epochs} Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Total: {losses['total_loss'].item():.4f} | Distill: {losses['distill_loss'].item():.4f} | Acc: {avg_acc:.2f}%"
                )

        if world_size > 1:
            for key, value in list(losses_accum.items()):
                tensor = torch.tensor(value, device=device, dtype=torch.float32)
                all_reduce_tensor(tensor, world_size)
                losses_accum[key] = tensor.item()
            correct_tensor = torch.tensor(total_correct, device=device, dtype=torch.float32)
            samples_tensor = torch.tensor(total_samples, device=device, dtype=torch.float32)
            all_reduce_tensor(correct_tensor, world_size)
            all_reduce_tensor(samples_tensor, world_size)
            total_correct = correct_tensor.item()
            total_samples = samples_tensor.item()

        divisor = max(len(train_loader) * world_size, 1)
        for key in losses_accum:
            losses_accum[key] /= divisor
        train_acc = 100.0 * total_correct / max(total_samples, 1.0)

        history["train_total_loss"].append(losses_accum["total"])
        history["train_cls_loss"].append(losses_accum["cls"])
        history["train_distill_loss"].append(losses_accum["distill"])
        history["train_feature_loss"].append(losses_accum["feature"])
        history["train_simclr_loss"].append(losses_accum["simclr"])
        history["train_acc"].append(train_acc)
        scheduler.step()

        if rank == 0:
            print(
                f"Student epoch {epoch + 1}: total={losses_accum['total']:.4f}, "
                f"distill={losses_accum['distill']:.4f}, acc={train_acc:.2f}%"
            )
            latest_path = output_dir / "latest_teacher_student_model.pth"
            save_phase_checkpoint(latest_path, model, optimizer, scheduler, scaler, history, config, max(best_acc, train_acc), epoch, "student")
            if train_acc >= best_acc:
                best_acc = train_acc
                save_phase_checkpoint(output_dir / "best_student_model.pth", model, optimizer, scheduler, scaler, history, config, best_acc, epoch, "student")

        barrier(world_size)

    return history, best_acc


def main_distributed(rank: int, local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_distributed(rank, world_size)
    setup_logging(rank)
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    train_transform, strong_train_transform, transform_config = build_dino_transforms(args)
    teacher_dataset = BinaryFolderDataset(
        root_folder=args.train_root,
        transform=train_transform,
        blur_prob=0.0,
        blur_strength_range=(args.blur_min, args.blur_max),
        blur_mode="no_blur",
        blur_type=args.blur_type,
        enable_strong_aug=True,
        strong_transform=strong_train_transform,
    )
    student_dataset = BinaryFolderDataset(
        root_folder=args.train_root,
        transform=train_transform,
        blur_prob=args.blur_prob,
        blur_strength_range=(args.blur_min, args.blur_max),
        blur_mode=args.blur_mode,
        blur_type=args.blur_type,
        mixed_mode_ratio=args.mixed_mode_ratio,
        ccmba_data_dir=args.ccmba_data_dir,
        enable_strong_aug=True,
        strong_transform=strong_train_transform,
    )

    teacher_sampler = DistributedSampler(teacher_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    student_sampler = DistributedSampler(student_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=args.teacher_batch_size,
        sampler=teacher_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    student_loader = DataLoader(
        student_dataset,
        batch_size=args.student_batch_size,
        sampler=student_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TeacherStudentNetwork(
        dinov3_model_path=args.dinov3_model_id,
        num_classes=2,
        projection_dim=args.projection_dim,
        local_files_only=args.local_files_only,
        device=device,
    )
    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=True,
    )

    if rank == 0:
        stats = count_trainable_parameters(model.module)
        print("=" * 70)
        print("TEACHER-STUDENT BACKBONE SWEEP")
        print(f"Backbone: {args.dinov3_model_id}")
        print(f"Data preset: {args.data_preset}")
        print(f"Train root: {args.train_root}")
        print(f"CCMBA data dir: {args.ccmba_data_dir}")
        print(f"Dataset size (teacher/student): {len(teacher_dataset)} / {len(student_dataset)}")
        print(f"Trainable params: {stats['trainable']} / {stats['total']}")
        print(f"Local files only: {args.local_files_only}")
        print("=" * 70)

    scaler = GradScaler(enabled=device.type == "cuda")
    teacher_optimizer = optim.AdamW(
        list(model.module.teacher.projection.parameters()) + list(model.module.teacher.classifier.parameters()),
        lr=args.teacher_learning_rate,
        weight_decay=args.weight_decay,
    )
    teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=args.teacher_epochs)

    student_optimizer = optim.AdamW(
        list(model.module.student_projection.parameters()) + list(model.module.student_classifier.parameters()),
        lr=args.student_learning_rate,
        weight_decay=args.weight_decay,
    )
    student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=args.student_epochs)

    output_dir = ensure_dir(Path(args.output_dir) / experiment_name(args))
    common_config = build_common_config(args, transform_config)

    teacher_history, teacher_best_acc = train_teacher_phase(
        model,
        teacher_loader,
        teacher_optimizer,
        teacher_scheduler,
        scaler,
        device,
        rank,
        world_size,
        args,
        output_dir,
        common_config,
    )

    student_history, student_best_acc = train_student_phase(
        model,
        student_loader,
        student_optimizer,
        student_scheduler,
        scaler,
        device,
        rank,
        world_size,
        args,
        output_dir,
        common_config,
    )

    if rank == 0:
        save_json(
            output_dir / "training_history.json",
            {
                "teacher_history": teacher_history,
                "student_history": student_history,
                "teacher_best_acc": teacher_best_acc,
                "student_best_acc": student_best_acc,
                "config": common_config,
            },
        )
        print(f"Training artifacts saved to: {output_dir}")

    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-student DINOv3 backbone sweep for SDV1.4.")
    parser.add_argument("--backbone-preset", choices=list(DEFAULT_DINOV3_MODELS.keys()), default="dinov3_vitl300m")
    parser.add_argument("--dinov3-model-id", type=str, default=None)
    parser.add_argument("--data-preset", choices=list(DATA_PRESETS.keys()), default="original_motion")
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--ccmba-data-dir", type=str, default=None)
    parser.add_argument("--blur-mode", choices=["global", "ccmba", "mixed"], default="mixed")
    parser.add_argument("--blur-type", choices=["motion", "gaussian"], default="motion")
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--blur-min", type=float, default=0.1)
    parser.add_argument("--blur-max", type=float, default=0.3)
    parser.add_argument("--mixed-mode-ratio", type=float, default=0.5)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--student-epochs", type=int, default=15)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--student-batch-size", type=int, default=32)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--student-learning-rate", type=float, default=5e-5)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--alpha-simclr", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/teacher_student")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true", help="Load DINOv3 checkpoints from local files only.")
    parser.add_argument("--allow-remote-backbone", dest="local_files_only", action="store_false", help="Allow Transformers to resolve remote model files.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()
    if args.dinov3_model_id is None:
        args.dinov3_model_id = DEFAULT_DINOV3_MODELS[args.backbone_preset]
    return resolve_data_paths(args)


if __name__ == "__main__":
    arguments = parse_args()
    rank, world_size, local_rank = parse_distributed_env()
    try:
        main_distributed(rank, local_rank, world_size, arguments)
    finally:
        cleanup_distributed()
