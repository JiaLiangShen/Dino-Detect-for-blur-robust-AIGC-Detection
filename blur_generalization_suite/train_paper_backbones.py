import argparse
import hashlib
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
from blur_generalization_suite.data_utils import (
    BinaryFolderDataset,
    TransformConfig,
    apply_blur_to_normalized_tensor,
    apply_paper_co_degradations_to_normalized_tensor,
    build_paper_train_transform,
    build_train_transform,
)
from blur_generalization_suite.model_zoo import (
    DEFAULT_DISTILLATION_BACKBONES,
    DEFAULT_PREPROCESS,
    DISTILLATION_BACKBONE_SPECS,
    FocalLoss,
    ImprovedTeacherStudentLoss,
    TeacherStudentNetwork,
    extract_teacher_student_head_state_dict,
)


DATA_PRESETS = {
    "sdv14": {
        "train_root": "/data/app.e0016372/imagenet_tmp/imagenet_ai_0419_sdv4",
        "ccmba_data_dir": "/data/app.e0016372/imagenet_tmp/ccmba_processed_sdv44",
    },
}


def actual_model(model):
    return model.module if hasattr(model, "module") else model


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    data_preset = DATA_PRESETS[args.data_preset]
    if args.train_root is None:
        args.train_root = data_preset["train_root"]
    if args.ccmba_data_dir is None:
        args.ccmba_data_dir = data_preset["ccmba_data_dir"]

    spec = DISTILLATION_BACKBONE_SPECS.get(args.backbone_preset)
    if spec is None and args.dinov3_model_id is None:
        raise ValueError("A custom backbone preset requires --dinov3-model-id")
    if spec is not None:
        if args.dinov3_model_id is None:
            args.dinov3_model_id = spec.local_dir
        if args.backbone_family is None:
            args.backbone_family = spec.backbone_family
        if args.loader_backend is None:
            args.loader_backend = spec.loader_backend
        if args.architecture_name is None:
            args.architecture_name = spec.architecture_name
    else:
        args.backbone_family = args.backbone_family or "dinov3"

    if args.training_profile == "strict_motion":
        args.blur_mode = "global"
        args.blur_type = "motion"
        args.ccmba_data_dir = None
        args.enable_co_degradations = False
        args.include_jpeg_augmentation = False
    if args.experiment_mode == "classification_only" and args.training_profile != "strict_motion":
        raise ValueError("classification_only is reserved for the strict_motion rebuttal control")
    if not 0.0 <= args.blur_prob <= 1.0:
        raise ValueError("--blur-prob must be in [0, 1]")
    if args.training_profile == "strict_motion" and args.blur_prob <= 0:
        raise ValueError("strict_motion requires a positive --blur-prob")
    if args.blur_min <= 0 or args.blur_max < args.blur_min:
        raise ValueError("Invalid blur strength range")
    positive_integer_fields = (
        "student_epochs",
        "teacher_batch_size",
        "student_batch_size",
        "teacher_accumulation_steps",
        "student_accumulation_steps",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.projection_dim <= 0:
        raise ValueError("--projection-dim must be positive")
    if args.teacher_learning_rate <= 0 or args.student_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0 or args.max_grad_norm < 0:
        raise ValueError("Weight decay and max gradient norm must be non-negative")
    if args.focal_gamma < 0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.experiment_mode == "dino_detect" and args.teacher_epochs <= 0:
        raise ValueError("--teacher-epochs must be positive for dino_detect")
    if args.experiment_mode == "dino_detect" and args.alpha_ordinal > 0 and args.blur_max <= args.blur_min:
        raise ValueError("Ordinal training requires --blur-max to be greater than --blur-min")
    return args


def resolve_transform_config(args: argparse.Namespace) -> TransformConfig:
    spec = DISTILLATION_BACKBONE_SPECS.get(args.backbone_preset)
    if spec is not None and spec.backbone_family == args.backbone_family:
        default = spec.preprocess
    else:
        default = DEFAULT_PREPROCESS.get(args.backbone_family, DEFAULT_PREPROCESS["dinov3"])
    return TransformConfig(
        resize_size=args.resize_size or default.resize_size,
        crop_size=args.crop_size or default.crop_size,
        mean=default.mean,
        std=default.std,
    )


def build_training_transform(args: argparse.Namespace, config: TransformConfig):
    if args.training_profile == "strict_motion":
        return build_train_transform(config)
    return build_paper_train_transform(config, include_jpeg=args.include_jpeg_augmentation)


def focal_alpha_for_dataset(dataset: BinaryFolderDataset, args: argparse.Namespace):
    if not args.class_balanced_focal:
        return args.focal_alpha
    labels = [label for _, label, _ in dataset.data]
    counts = [max(labels.count(index), 1) for index in (0, 1)]
    total = float(sum(counts))
    return [total / (2.0 * count) for count in counts]


def stable_rng(seed: int, epoch: int, category: str, image_name: str, stream: str) -> random.Random:
    payload = f"{seed}|{epoch}|{category}|{image_name}|{stream}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return random.Random(stable_seed)


def seed_student_forward(seed: int, rank: int, epoch: int, batch_index: int) -> None:
    """Pair student-head dropout across strict control runs, batch by batch."""
    forward_seed = seed + 1_000_003 * rank + 10_007 * epoch + batch_index
    torch.manual_seed(forward_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(forward_seed)


def strict_motion_view(
    image: torch.Tensor,
    transform_config: TransformConfig,
    args: argparse.Namespace,
    epoch: int,
    category: str,
    image_name: str,
    stream: str,
    strength_range: Tuple[float, float] | None = None,
    force_blur: bool = False,
):
    rng = stable_rng(args.seed, epoch, category, image_name, stream)
    if not force_blur and rng.random() > args.blur_prob:
        return image, 0.0
    low, high = strength_range or (args.blur_min, args.blur_max)
    strength = rng.uniform(low, high)
    angle = rng.uniform(0.0, 180.0)
    phase = rng.uniform(0.0, 2.0 * 3.141592653589793)
    blurred = apply_blur_to_normalized_tensor(
        image,
        "motion",
        strength,
        transform_config.mean,
        transform_config.std,
        angle_degrees=angle,
        trajectory_jitter=args.trajectory_jitter,
        phase=phase,
    )
    return blurred, strength


def build_primary_student_views(
    images: torch.Tensor,
    labels: torch.Tensor,
    image_names: Iterable[str],
    categories: Iterable[str],
    dataset: BinaryFolderDataset,
    transform_config: TransformConfig,
    args: argparse.Namespace,
    epoch: int,
) -> torch.Tensor:
    views = []
    for image, label, image_name, category in zip(images, labels, image_names, categories):
        if args.training_profile == "strict_motion":
            degraded, _ = strict_motion_view(
                image,
                transform_config,
                args,
                epoch,
                category,
                image_name,
                stream="primary",
            )
        else:
            degraded, _ = dataset.apply_blur_augmentation(
                image,
                image_name,
                category,
                is_real=label.item() == 0,
            )
            if args.enable_co_degradations:
                degraded = apply_paper_co_degradations_to_normalized_tensor(
                    degraded,
                    transform_config.mean,
                    transform_config.std,
                    defocus_prob=args.defocus_prob,
                    noise_prob=args.noise_prob,
                    jpeg_prob=args.co_jpeg_prob,
                    resize_prob=args.resize_degradation_prob,
                )
        views.append(degraded)
    return torch.stack(views)


def build_ordinal_views(
    images: torch.Tensor,
    image_names: Iterable[str],
    categories: Iterable[str],
    transform_config: TransformConfig,
    args: argparse.Namespace,
    epoch: int,
):
    midpoint = (args.blur_min + args.blur_max) / 2.0
    mild_views = []
    severe_views = []
    mild_levels = []
    severe_levels = []
    for image, image_name, category in zip(images, image_names, categories):
        mild, mild_strength = strict_motion_view(
            image,
            transform_config,
            args,
            epoch,
            category,
            image_name,
            stream="ordinal_mild",
            strength_range=(args.blur_min, midpoint),
            force_blur=True,
        )
        severe, severe_strength = strict_motion_view(
            image,
            transform_config,
            args,
            epoch,
            category,
            image_name,
            stream="ordinal_severe",
            strength_range=(midpoint, args.blur_max),
            force_blur=True,
        )
        mild_views.append(mild)
        severe_views.append(severe)
        mild_levels.append(mild_strength)
        severe_levels.append(severe_strength)
    return (
        torch.stack(mild_views),
        torch.stack(severe_views),
        torch.tensor(mild_levels, device=images.device, dtype=torch.float32),
        torch.tensor(severe_levels, device=images.device, dtype=torch.float32),
    )


def head_state_for_phase(network: TeacherStudentNetwork) -> Dict[str, torch.Tensor]:
    return extract_teacher_student_head_state_dict(network)


def save_phase_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    history: dict,
    config: dict,
    best_acc: float,
    epoch: int,
    phase: str,
) -> None:
    network = actual_model(model)
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "best_acc": best_acc,
            "head_state_dict": head_state_for_phase(network),
            "trainable_state_dict": extract_trainable_state_dict(network),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def restore_teacher_head(path: Path, model, rank: int, world_size: int) -> None:
    network = actual_model(model)
    if rank == 0:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("head_state_dict", checkpoint.get("trainable_state_dict", {}))
        teacher_state = {
            name: value
            for name, value in state_dict.items()
            if name.startswith(("teacher.projection.", "teacher.classifier."))
        }
        if not teacher_state:
            raise RuntimeError(f"No teacher-head parameters found in {path}")
        missing, unexpected = network.load_state_dict(teacher_state, strict=False)
        missing_teacher = [
            name
            for name in missing
            if name.startswith(("teacher.projection.", "teacher.classifier."))
        ]
        if missing_teacher or unexpected:
            raise RuntimeError(
                f"Teacher checkpoint mismatch: missing={missing_teacher}, unexpected={unexpected}"
            )

    if world_size > 1 and torch.distributed.is_initialized():
        for parameter in list(network.teacher.projection.parameters()) + list(
            network.teacher.classifier.parameters()
        ):
            torch.distributed.broadcast(parameter.data, src=0)
        for buffer in list(network.teacher.projection.buffers()) + list(
            network.teacher.classifier.buffers()
        ):
            torch.distributed.broadcast(buffer.data, src=0)


def reduce_epoch_metrics(values, device: torch.device, world_size: int):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    all_reduce_tensor(tensor, world_size)
    return tensor.tolist()


def accumulation_context(model, should_sync: bool):
    if should_sync or not isinstance(model, DDP):
        return nullcontext()
    return model.no_sync()


def train_teacher_phase(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    rank,
    world_size,
    args,
    output_dir,
    config,
    focal_alpha,
):
    criterion = ImprovedTeacherStudentLoss(
        temperature=args.temperature,
        alpha_distill=args.alpha_distill,
        alpha_cls=args.alpha_cls,
        alpha_feature=args.alpha_feature,
        alpha_ordinal=0.0,
        alpha_simclr=0.0,
        classification_loss="focal",
        focal_alpha=focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    network = actual_model(model)
    network.freeze_teacher()
    network.unfreeze_teacher_head()
    network.freeze_student()
    history = {"train_loss": [], "train_acc": []}
    best_acc = -1.0

    for epoch in range(args.teacher_epochs):
        loader.sampler.set_epoch(epoch)
        model.train()
        network.teacher.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_correct = total_samples = total_steps = 0.0

        for batch_index, (images, labels, _, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            window_start = (batch_index // args.teacher_accumulation_steps) * args.teacher_accumulation_steps
            window_size = min(args.teacher_accumulation_steps, len(loader) - window_start)
            should_step = (batch_index + 1) % args.teacher_accumulation_steps == 0 or batch_index + 1 == len(loader)

            with accumulation_context(model, should_step):
                with autocast(enabled=device.type == "cuda"):
                    features, logits = model(images, branch="teacher")
                    losses = criterion(
                        student_features=features,
                        student_logits=logits,
                        labels=labels,
                        mode="teacher",
                    )
                    backward_loss = losses["total_loss"] / window_size
                scaler.scale(backward_loss).backward()

            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(network.teacher.projection.parameters())
                        + list(network.teacher.classifier.parameters()),
                        args.max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += losses["total_loss"].item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
            total_steps += 1

        total_loss, total_correct, total_samples, total_steps = reduce_epoch_metrics(
            [total_loss, total_correct, total_samples, total_steps], device, world_size
        )
        epoch_loss = total_loss / max(total_steps, 1.0)
        epoch_acc = 100.0 * total_correct / max(total_samples, 1.0)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        scheduler.step()

        if rank == 0:
            print(f"Teacher epoch {epoch + 1}/{args.teacher_epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")
            save_phase_checkpoint(
                output_dir / "latest_teacher_model.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                config,
                max(best_acc, epoch_acc),
                epoch,
                "teacher",
            )
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                save_phase_checkpoint(
                    output_dir / "best_teacher_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    config,
                    best_acc,
                    epoch,
                    "teacher",
                )
        barrier(world_size)

    restore_teacher_head(output_dir / "best_teacher_model.pth", model, rank, world_size)
    barrier(world_size)
    return history, best_acc


def train_student_phase(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    rank,
    world_size,
    args,
    output_dir,
    config,
    transform_config,
    focal_alpha,
    classification_only: bool,
):
    criterion = ImprovedTeacherStudentLoss(
        temperature=args.temperature,
        alpha_distill=args.alpha_distill,
        alpha_cls=args.alpha_cls,
        alpha_feature=args.alpha_feature,
        alpha_simclr=0.0,
        alpha_ordinal=args.alpha_ordinal,
        classification_loss="focal",
        focal_alpha=focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    classification_criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    network = actual_model(model)
    network.freeze_teacher()
    network.unfreeze_student()
    history = {
        "train_total_loss": [],
        "train_cls_loss": [],
        "train_distill_loss": [],
        "train_feature_loss": [],
        "train_ordinal_loss": [],
        "train_acc": [],
    }
    best_acc = -1.0
    checkpoint_name = "classification_only" if classification_only else "student"

    for epoch in range(args.student_epochs):
        loader.sampler.set_epoch(epoch)
        model.train()
        network.teacher.eval()
        network.student_projection.train()
        network.student_classifier.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "cls": 0.0, "distill": 0.0, "feature": 0.0, "ordinal": 0.0}
        total_correct = total_samples = total_steps = 0.0

        for batch_index, (images, labels, image_names, categories) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            student_inputs = build_primary_student_views(
                images,
                labels,
                image_names,
                categories,
                loader.dataset,
                transform_config,
                args,
                epoch,
            ).to(device, non_blocking=True)

            window_start = (batch_index // args.student_accumulation_steps) * args.student_accumulation_steps
            window_size = min(args.student_accumulation_steps, len(loader) - window_start)
            should_step = (batch_index + 1) % args.student_accumulation_steps == 0 or batch_index + 1 == len(loader)

            with accumulation_context(model, should_step):
                with autocast(enabled=device.type == "cuda"):
                    if classification_only:
                        seed_student_forward(args.seed, rank, epoch, batch_index)
                        student_features, student_logits = model(student_inputs, branch="student")
                        cls_loss = classification_criterion(student_logits, labels)
                        zero = cls_loss.detach() * 0.0
                        losses = {
                            "total_loss": cls_loss,
                            "cls_loss": cls_loss,
                            "distill_loss": zero,
                            "feature_loss": zero,
                            "ordinal_loss": zero,
                        }
                    else:
                        with torch.no_grad():
                            teacher_features, teacher_logits = network.forward_teacher(images)
                        mild, severe, mild_levels, severe_levels = build_ordinal_views(
                            images,
                            image_names,
                            categories,
                            transform_config,
                            args,
                            epoch,
                        )
                        batch_size = images.size(0)
                        combined_inputs = torch.cat([student_inputs, mild, severe], dim=0)
                        seed_student_forward(args.seed, rank, epoch, batch_index)
                        combined_features, combined_logits = model(combined_inputs, branch="student")
                        student_features, mild_features, severe_features = combined_features.split(
                            batch_size, dim=0
                        )
                        student_logits = combined_logits[:batch_size]
                        ordinal_features = torch.stack(
                            [teacher_features.detach(), mild_features, severe_features], dim=1
                        )
                        ordinal_severities = torch.stack(
                            [torch.zeros_like(mild_levels), mild_levels, severe_levels], dim=1
                        )
                        losses = criterion(
                            student_features=student_features,
                            student_logits=student_logits,
                            teacher_features=teacher_features,
                            teacher_logits=teacher_logits,
                            labels=labels,
                            mode="student",
                            ordinal_features=ordinal_features,
                            ordinal_severities=ordinal_severities,
                        )
                    backward_loss = losses["total_loss"] / window_size
                scaler.scale(backward_loss).backward()

            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(network.student_projection.parameters())
                        + list(network.student_classifier.parameters()),
                        args.max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            totals["total"] += losses["total_loss"].item()
            totals["cls"] += losses["cls_loss"].item()
            totals["distill"] += losses["distill_loss"].item()
            totals["feature"] += losses["feature_loss"].item()
            totals["ordinal"] += losses["ordinal_loss"].item()
            total_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
            total_steps += 1

        reduced = reduce_epoch_metrics(
            [
                totals["total"],
                totals["cls"],
                totals["distill"],
                totals["feature"],
                totals["ordinal"],
                total_correct,
                total_samples,
                total_steps,
            ],
            device,
            world_size,
        )
        total_steps = max(reduced[7], 1.0)
        epoch_values = [value / total_steps for value in reduced[:5]]
        epoch_acc = 100.0 * reduced[5] / max(reduced[6], 1.0)
        for key, value in zip(
            ("train_total_loss", "train_cls_loss", "train_distill_loss", "train_feature_loss", "train_ordinal_loss"),
            epoch_values,
        ):
            history[key].append(value)
        history["train_acc"].append(epoch_acc)
        scheduler.step()

        if rank == 0:
            print(
                f"{checkpoint_name} epoch {epoch + 1}/{args.student_epochs}: "
                f"loss={epoch_values[0]:.4f}, acc={epoch_acc:.2f}%"
            )
            save_phase_checkpoint(
                output_dir / f"latest_{checkpoint_name}_model.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                config,
                max(best_acc, epoch_acc),
                epoch,
                checkpoint_name,
            )
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                save_phase_checkpoint(
                    output_dir / f"best_{checkpoint_name}_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    config,
                    best_acc,
                    epoch,
                    checkpoint_name,
                )
        barrier(world_size)
    return history, best_acc


def make_loader(dataset, batch_size, world_size, rank, workers, seed):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def experiment_name(args: argparse.Namespace) -> str:
    backbone_name = args.backbone_preset or Path(args.dinov3_model_id).name
    blur_tag = str(args.blur_prob).replace(".", "")
    return f"{backbone_name}_{args.data_preset}_{args.experiment_mode}_{args.training_profile}_blur{blur_tag}"


def build_config(args: argparse.Namespace, transform_config: TransformConfig) -> dict:
    strict = args.training_profile == "strict_motion"
    explicit_augmentations = ["motion_blur"] if args.blur_prob > 0 else []
    if not strict:
        if args.blur_mode in {"ccmba", "mixed"} and args.ccmba_data_dir:
            explicit_augmentations.append("ccmba")
        if args.enable_co_degradations and args.defocus_prob > 0:
            explicit_augmentations.append("defocus_blur")
        if args.enable_co_degradations and args.noise_prob > 0:
            explicit_augmentations.append("sensor_noise")
        if args.include_jpeg_augmentation or (
            args.enable_co_degradations and args.co_jpeg_prob > 0
        ):
            explicit_augmentations.append("jpeg")
        if args.enable_co_degradations and args.resize_degradation_prob > 0:
            explicit_augmentations.append("down_up_resize")
    return {
        "dinov3_model_id": args.dinov3_model_id,
        "backbone_family": args.backbone_family,
        "backbone_preset": args.backbone_preset,
        "loader_backend": args.loader_backend,
        "architecture_name": args.architecture_name,
        "num_classes": 2,
        "projection_dim": args.projection_dim,
        "data_preset": args.data_preset,
        "train_root": args.train_root,
        "max_samples_per_class": args.max_samples_per_class,
        "ccmba_data_dir": args.ccmba_data_dir,
        "training_profile": args.training_profile,
        "experiment_mode": args.experiment_mode,
        "strict_motion_only": strict,
        "blur_mode": args.blur_mode,
        "blur_type": args.blur_type,
        "blur_prob": args.blur_prob,
        "blur_strength_range": [args.blur_min, args.blur_max],
        "trajectory_jitter": args.trajectory_jitter,
        "mixed_mode_ratio": args.mixed_mode_ratio,
        "classification_loss": "focal",
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "alpha_cls": args.alpha_cls,
        "alpha_distill": args.alpha_distill,
        "alpha_feature": args.alpha_feature,
        "alpha_ordinal": 0.0 if args.experiment_mode == "classification_only" else args.alpha_ordinal,
        "temperature": args.temperature,
        "jpeg_augmentation": bool(args.include_jpeg_augmentation),
        "co_degradations": bool(args.enable_co_degradations),
        "defocus_augmentation": bool(args.enable_co_degradations and args.defocus_prob > 0),
        "noise_augmentation": bool(args.enable_co_degradations and args.noise_prob > 0),
        "resize_degradation": bool(args.enable_co_degradations and args.resize_degradation_prob > 0),
        "explicit_synthetic_augmentations": explicit_augmentations,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "teacher_learning_rate": args.teacher_learning_rate,
        "student_learning_rate": args.student_learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "teacher_micro_batch": args.teacher_batch_size,
        "student_micro_batch": args.student_batch_size,
        "teacher_accumulation_steps": args.teacher_accumulation_steps,
        "student_accumulation_steps": args.student_accumulation_steps,
        "teacher_effective_batch_per_gpu": args.teacher_batch_size * args.teacher_accumulation_steps,
        "student_effective_batch_per_gpu": args.student_batch_size * args.student_accumulation_steps,
        "teacher_global_effective_batch": args.teacher_batch_size * args.teacher_accumulation_steps,
        "student_global_effective_batch": args.student_batch_size * args.student_accumulation_steps,
        "class_balanced_focal": bool(args.class_balanced_focal),
        "transform_config": {
            "resize_size": transform_config.resize_size,
            "crop_size": transform_config.crop_size,
            "mean": list(transform_config.mean),
            "std": list(transform_config.std),
        },
        "local_files_only": args.local_files_only,
        "default_eval_branch": "student",
        "seed": args.seed,
    }


def main_distributed(rank: int, local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_distributed(rank, world_size, local_rank=local_rank)
    setup_logging(rank)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    transform_config = resolve_transform_config(args)
    train_transform = build_training_transform(args, transform_config)
    common_dataset_args = {
        "root_folder": args.train_root,
        "transform": train_transform,
        "blur_strength_range": (args.blur_min, args.blur_max),
        "blur_type": args.blur_type,
        "max_samples_per_class": args.max_samples_per_class,
        "normalization_mean": transform_config.mean,
        "normalization_std": transform_config.std,
    }
    teacher_dataset = BinaryFolderDataset(
        **common_dataset_args,
        blur_prob=0.0,
        blur_mode="no_blur",
    )
    student_dataset = BinaryFolderDataset(
        **common_dataset_args,
        blur_prob=args.blur_prob,
        blur_mode=args.blur_mode,
        mixed_mode_ratio=args.mixed_mode_ratio,
        ccmba_data_dir=args.ccmba_data_dir,
    )
    if not teacher_dataset.data or not student_dataset.data:
        raise RuntimeError(f"No training images found under {args.train_root}")

    model = TeacherStudentNetwork(
        dinov3_model_path=args.dinov3_model_id,
        num_classes=2,
        projection_dim=args.projection_dim,
        local_files_only=args.local_files_only,
        device=device,
        backbone_family=args.backbone_family,
        loader_backend=args.loader_backend,
        architecture_name=args.architecture_name,
    )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    output_dir = ensure_dir(Path(args.output_dir) / experiment_name(args))
    config = build_config(args, transform_config)
    focal_alpha = focal_alpha_for_dataset(student_dataset, args)
    config["resolved_focal_alpha"] = focal_alpha
    config["world_size"] = world_size
    config["teacher_global_effective_batch"] *= world_size
    config["student_global_effective_batch"] *= world_size
    network = actual_model(model)

    if rank == 0:
        stats = count_trainable_parameters(network)
        print("=" * 78)
        print("PAPER-FAITHFUL DINO-DETECT BACKBONE TRAINING")
        print(f"Backbone: {args.backbone_preset} ({args.backbone_family})")
        print(f"Path: {args.dinov3_model_id}")
        print(f"Profile/mode: {args.training_profile} / {args.experiment_mode}")
        print(f"Input: resize {transform_config.resize_size}, crop {transform_config.crop_size}")
        print(
            f"Effective batch per GPU: teacher={config['teacher_effective_batch_per_gpu']}, "
            f"student={config['student_effective_batch_per_gpu']}"
        )
        print(f"Trainable parameters before phase selection: {stats['trainable']} / {stats['total']}")
        print(f"Artifacts: {output_dir}")
        print("=" * 78)

    teacher_history = None
    teacher_best_acc = None
    if args.experiment_mode == "dino_detect":
        teacher_loader = make_loader(
            teacher_dataset,
            args.teacher_batch_size,
            world_size,
            rank,
            args.num_workers,
            args.seed,
        )
        if len(teacher_loader) == 0:
            raise RuntimeError(
                "Teacher DataLoader is empty after drop_last=True; reduce --teacher-batch-size "
                "or provide more training samples."
            )
        teacher_scaler = GradScaler(enabled=device.type == "cuda")
        teacher_optimizer = optim.AdamW(
            list(network.teacher.projection.parameters()) + list(network.teacher.classifier.parameters()),
            lr=args.teacher_learning_rate,
            weight_decay=args.weight_decay,
        )
        teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer, T_max=args.teacher_epochs
        )
        teacher_history, teacher_best_acc = train_teacher_phase(
            model,
            teacher_loader,
            teacher_optimizer,
            teacher_scheduler,
            teacher_scaler,
            device,
            rank,
            world_size,
            args,
            output_dir,
            config,
            focal_alpha,
        )
        del teacher_loader

    # Reset phase-specific RNG so the strict control and DINO-Detect see identical
    # sample order, random crop stream, and primary motion parameters.
    set_seed(args.seed + 10000 + rank)
    student_loader = make_loader(
        student_dataset,
        args.student_batch_size,
        world_size,
        rank,
        args.num_workers,
        args.seed + 10000,
    )
    if len(student_loader) == 0:
        raise RuntimeError(
            "Student DataLoader is empty after drop_last=True; reduce --student-batch-size "
            "or provide more training samples."
        )
    # The strict classification control and DINO-Detect student start with the
    # same fresh AMP state; teacher training must not influence loss scaling.
    student_scaler = GradScaler(enabled=device.type == "cuda")
    student_optimizer = optim.AdamW(
        list(network.student_projection.parameters()) + list(network.student_classifier.parameters()),
        lr=args.student_learning_rate,
        weight_decay=args.weight_decay,
    )
    student_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        student_optimizer, T_max=args.student_epochs
    )
    student_history, student_best_acc = train_student_phase(
        model,
        student_loader,
        student_optimizer,
        student_scheduler,
        student_scaler,
        device,
        rank,
        world_size,
        args,
        output_dir,
        config,
        transform_config,
        focal_alpha,
        classification_only=args.experiment_mode == "classification_only",
    )

    if rank == 0:
        save_json(
            output_dir / "training_history.json",
            {
                "teacher_history": teacher_history,
                "student_history": student_history,
                "teacher_best_acc": teacher_best_acc,
                "student_best_acc": student_best_acc,
                "config": config,
            },
        )
        print(f"Training artifacts saved to: {output_dir}")
    barrier(world_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful frozen-backbone DINO-Detect training for DINOv3, "
            "CLIP ViT-bigG, EVA-GIANT, and AIMv2-Huge."
        )
    )
    parser.add_argument("--backbone-preset", choices=list(DEFAULT_DISTILLATION_BACKBONES), default="dinov3_vit7b")
    parser.add_argument("--dinov3-model-id", type=str, default=None)
    parser.add_argument("--backbone-family", choices=("dinov3", "siglip2", "aimv2", "clip", "eva"), default=None)
    parser.add_argument("--loader-backend", choices=("transformers_auto", "transformers_clip", "timm"), default=None)
    parser.add_argument("--architecture-name", type=str, default=None)
    parser.add_argument("--training-profile", choices=("paper", "strict_motion"), default="paper")
    parser.add_argument("--experiment-mode", choices=("dino_detect", "classification_only"), default="dino_detect")
    parser.add_argument("--data-preset", choices=list(DATA_PRESETS), default="sdv14")
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--ccmba-data-dir", type=str, default=None)
    parser.add_argument("--blur-mode", choices=("global", "ccmba", "mixed"), default="mixed")
    parser.add_argument("--blur-type", choices=("motion", "gaussian"), default="motion")
    parser.add_argument("--blur-prob", type=float, default=0.1)
    parser.add_argument("--blur-min", type=float, default=0.1)
    parser.add_argument("--blur-max", type=float, default=0.3)
    parser.add_argument("--trajectory-jitter", type=float, default=0.12)
    parser.add_argument("--mixed-mode-ratio", type=float, default=0.5)
    parser.add_argument("--include-jpeg-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-co-degradations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--defocus-prob", type=float, default=0.2)
    parser.add_argument("--noise-prob", type=float, default=0.1)
    parser.add_argument("--co-jpeg-prob", type=float, default=0.1)
    parser.add_argument("--resize-degradation-prob", type=float, default=0.1)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=4)
    parser.add_argument("--student-epochs", type=int, default=15)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--student-batch-size", type=int, default=1)
    parser.add_argument("--teacher-accumulation-steps", type=int, default=32)
    parser.add_argument("--student-accumulation-steps", type=int, default=128)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--student-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha-cls", type=float, default=1.0)
    parser.add_argument("--alpha-distill", type=float, default=1.0)
    parser.add_argument("--alpha-feature", type=float, default=0.5)
    parser.add_argument("--alpha-ordinal", type=float, default=0.3)
    parser.add_argument("--focal-alpha", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-balanced-focal", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="blur_generalization_suite/outputs/paper_parity")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--allow-remote-backbone", dest="local_files_only", action="store_false")
    parser.add_argument("--seed", type=int, default=3407)
    parser.set_defaults(local_files_only=True)
    return parser


def main() -> None:
    args = resolve_args(build_parser().parse_args())
    rank, world_size, local_rank = parse_distributed_env()
    try:
        main_distributed(rank, local_rank, world_size, args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
