from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..config import TrainConfig
from ..data import BinaryAIGCDataset, BlurAugmentor, build_train_transform
from ..models import TeacherStudentModel
from ..utils.checkpoint import save_checkpoint
from ..utils.distributed import (
    autocast_context,
    barrier,
    cleanup_distributed,
    is_main_process,
    reduce_counts,
    reduce_scalar,
    setup_distributed,
)
from ..utils.misc import ensure_dir, save_json, set_seed, timestamp
from .losses import TeacherStudentLoss


def _build_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _make_loader(dataset, batch_size: int, num_workers: int, distributed: bool, shuffle: bool):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, sampler


def _teacher_epoch(model, loader, sampler, optimizer, scheduler, scaler, criterion, config, device, rank):
    if sampler is not None:
        sampler.set_epoch(_teacher_epoch.epoch_index)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device=device, enabled=config.amp):
            _, logits = _unwrap_model(model).forward_teacher(images)
            losses = criterion.forward_teacher(logits=logits, labels=labels)

        scaler.scale(losses["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(losses["total_loss"].item())
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

        if is_main_process(rank) and step % config.log_interval == 0:
            print(
                f"[Teacher] epoch={_teacher_epoch.epoch_index + 1} "
                f"step={step}/{len(loader)} loss={losses['total_loss'].item():.4f}"
            )

    scheduler.step()
    avg_loss = reduce_scalar(total_loss / max(len(loader), 1), device=device)
    total_correct, total_samples = reduce_counts(total_correct, total_samples, device=device)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


_teacher_epoch.epoch_index = 0


def _student_epoch(model, loader, sampler, optimizer, scheduler, scaler, criterion, blur_augmentor, config, device, rank):
    if sampler is not None:
        sampler.set_epoch(_student_epoch.epoch_index)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        clean_images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_names = batch["image_name"]
        categories = batch["category"]

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with autocast_context(device=device, enabled=config.amp):
                teacher_features, teacher_logits = _unwrap_model(model).forward_teacher(clean_images)

        blurred_images = []
        for image_tensor, label, image_name, category in zip(clean_images, labels, image_names, categories):
            blurred_image, _ = blur_augmentor.apply(
                image=image_tensor,
                image_name=str(image_name),
                category=str(category),
                is_real=bool(label.item() == 0),
            )
            blurred_images.append(blurred_image)

        blurred_batch = torch.stack(blurred_images).to(device, non_blocking=True)

        with autocast_context(device=device, enabled=config.amp):
            student_features, student_logits = _unwrap_model(model).forward_student(blurred_batch)
            losses = criterion.forward_student(
                student_features=student_features,
                student_logits=student_logits,
                teacher_features=teacher_features,
                teacher_logits=teacher_logits,
                labels=labels,
            )

        scaler.scale(losses["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(losses["total_loss"].item())
        predictions = student_logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

        if is_main_process(rank) and step % config.log_interval == 0:
            print(
                f"[Student] epoch={_student_epoch.epoch_index + 1} "
                f"step={step}/{len(loader)} loss={losses['total_loss'].item():.4f}"
            )

    scheduler.step()
    avg_loss = reduce_scalar(total_loss / max(len(loader), 1), device=device)
    total_correct, total_samples = reduce_counts(total_correct, total_samples, device=device)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


_student_epoch.epoch_index = 0


def run_training(config: TrainConfig) -> None:
    context = {"rank": 0, "local_rank": 0, "world_size": 1, "distributed": False}
    device = _build_device(local_rank=0)
    try:
        context = setup_distributed(device=device)
        device = _build_device(local_rank=int(context["local_rank"]))
        set_seed(config.seed + int(context["rank"]))

        torch.set_float32_matmul_precision("high")
        output_dir = ensure_dir(Path(config.output_dir) / config.experiment_name)
        checkpoint_dir = ensure_dir(output_dir / "checkpoints")
        log_dir = ensure_dir(output_dir / "logs")

        transform = build_train_transform(config.image_size, config.crop_size)
        dataset = BinaryAIGCDataset(
            root=config.train_root,
            transform=transform,
            max_samples_per_class=config.max_samples_per_class,
        )

        teacher_loader, teacher_sampler = _make_loader(
            dataset=dataset,
            batch_size=config.teacher_batch_size,
            num_workers=config.num_workers,
            distributed=bool(context["distributed"]),
            shuffle=True,
        )
        student_loader, student_sampler = _make_loader(
            dataset=dataset,
            batch_size=config.student_batch_size,
            num_workers=config.num_workers,
            distributed=bool(context["distributed"]),
            shuffle=True,
        )

        model = TeacherStudentModel(
            backbone_path=config.backbone_path,
            num_classes=config.num_classes,
            projection_dim=config.projection_dim,
            adapter_layers=config.adapter_layers,
            dropout=config.dropout,
        ).to(device)

        if context["distributed"]:
            ddp_kwargs = {"device_ids": [int(context["local_rank"])]} if device.type == "cuda" else {}
            model = DDP(model, **ddp_kwargs)

        criterion = TeacherStudentLoss(
            temperature=config.temperature,
            alpha_distill=config.alpha_distill,
            alpha_cls=config.alpha_cls,
            alpha_feature=config.alpha_feature,
            alpha_contrastive=config.alpha_contrastive,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")

        teacher_optimizer = torch.optim.AdamW(
            _unwrap_model(model).teacher_parameters(),
            lr=config.teacher_lr,
            weight_decay=config.weight_decay,
        )
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=config.teacher_epochs)

        student_optimizer = torch.optim.AdamW(
            _unwrap_model(model).student_parameters(),
            lr=config.student_lr,
            weight_decay=config.weight_decay,
        )
        student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=config.student_epochs)

        blur_augmentor = BlurAugmentor(
            mode=config.blur_mode,
            probability=config.blur_probability,
            strength_range=(float(config.blur_strength_range[0]), float(config.blur_strength_range[1])),
            mixed_ratio=config.mixed_blur_ratio,
            ccmba_root=config.ccmba_root,
            ccmba_transform=transform,
        )

        if is_main_process(int(context["rank"])):
            save_json(log_dir / "train_config.json", asdict(config))
            print(f"Training samples: {len(dataset)}")
            print(f"Outputs will be saved to: {output_dir}")

        best_teacher_accuracy = 0.0
        best_student_accuracy = 0.0
        history = {"teacher": [], "student": []}

        for epoch in range(config.teacher_epochs):
            _teacher_epoch.epoch_index = epoch
            metrics = _teacher_epoch(
                model=model,
                loader=teacher_loader,
                sampler=teacher_sampler,
                optimizer=teacher_optimizer,
                scheduler=teacher_scheduler,
                scaler=scaler,
                criterion=criterion,
                config=config,
                device=device,
                rank=int(context["rank"]),
            )
            history["teacher"].append({"epoch": epoch + 1, **metrics})

            if is_main_process(int(context["rank"])):
                print(f"[Teacher] epoch={epoch + 1} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
                if metrics["accuracy"] >= best_teacher_accuracy:
                    best_teacher_accuracy = metrics["accuracy"]
                    save_checkpoint(
                        checkpoint_dir / "best_teacher.pth",
                        model_state=_unwrap_model(model).state_dict(),
                        optimizer_state=teacher_optimizer.state_dict(),
                        scheduler_state=teacher_scheduler.state_dict(),
                        scaler_state=scaler.state_dict(),
                        metadata={"stage": "teacher", "epoch": epoch + 1, "accuracy": metrics["accuracy"]},
                    )
            barrier()

        for epoch in range(config.student_epochs):
            _student_epoch.epoch_index = epoch
            metrics = _student_epoch(
                model=model,
                loader=student_loader,
                sampler=student_sampler,
                optimizer=student_optimizer,
                scheduler=student_scheduler,
                scaler=scaler,
                criterion=criterion,
                blur_augmentor=blur_augmentor,
                config=config,
                device=device,
                rank=int(context["rank"]),
            )
            history["student"].append({"epoch": epoch + 1, **metrics})

            if is_main_process(int(context["rank"])):
                print(f"[Student] epoch={epoch + 1} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
                if metrics["accuracy"] >= best_student_accuracy:
                    best_student_accuracy = metrics["accuracy"]
                    save_checkpoint(
                        checkpoint_dir / "best_student.pth",
                        model_state=_unwrap_model(model).state_dict(),
                        optimizer_state=student_optimizer.state_dict(),
                        scheduler_state=student_scheduler.state_dict(),
                        scaler_state=scaler.state_dict(),
                        metadata={"stage": "student", "epoch": epoch + 1, "accuracy": metrics["accuracy"]},
                    )
            barrier()

        if is_main_process(int(context["rank"])):
            save_json(
                log_dir / f"training_history_{timestamp()}.json",
                {
                    "config": asdict(config),
                    "best_teacher_accuracy": best_teacher_accuracy,
                    "best_student_accuracy": best_student_accuracy,
                    "history": history,
                },
            )
    finally:
        cleanup_distributed()
