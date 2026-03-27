import math
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel, ViTModel

from .data_utils import TransformConfig


DEFAULT_DINOV3_MODELS = {
    "dinov3_vit7b": "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m",
    "dinov3_vitl300m": "/nas_train/app.e0016372/models/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith840m": "/nas_train/app.e0016372/models/dinov3-vith16plus-pretrain-lvd1689m",
}

DEFAULT_LORA_BACKBONES = {
    "clip_lora": "/nas_train/app.e0016372/models/clip-vit-large-patch14",
    "vit_large_lora": "/nas_train/app.e0016372/models/vit-large-patch16-224-in21k",
}

DEFAULT_LORA_TARGETS = {
    "clip_lora": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
    ),
    "vit_large_lora": (
        "attention.attention.query",
        "attention.attention.key",
        "attention.attention.value",
        "attention.output.dense",
    ),
}

DEFAULT_PREPROCESS = {
    "clip_lora": TransformConfig(
        resize_size=224,
        crop_size=224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    "vit_large_lora": TransformConfig(
        resize_size=256,
        crop_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "dinov3": TransformConfig(
        resize_size=512,
        crop_size=448,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


def _resolve_size(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, dict):
        for key in ("shortest_edge", "height", "width", "size"):
            if key in value:
                return int(value[key])
        return default
    if isinstance(value, (tuple, list)):
        return int(value[0])
    return int(value)


def resolve_preprocess_config(
    model_family: str,
    backbone_path: str,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    resize_override: int | None = None,
    crop_override: int | None = None,
) -> TransformConfig:
    fallback = DEFAULT_PREPROCESS[model_family if model_family in DEFAULT_PREPROCESS else "dinov3"]
    try:
        processor = AutoImageProcessor.from_pretrained(
            backbone_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        resize_size = _resolve_size(getattr(processor, "size", None), fallback.resize_size)
        crop_size = _resolve_size(getattr(processor, "crop_size", None), fallback.crop_size)
        mean = tuple(float(x) for x in getattr(processor, "image_mean", fallback.mean))
        std = tuple(float(x) for x in getattr(processor, "image_std", fallback.std))
        return TransformConfig(
            resize_size=resize_override or resize_size,
            crop_size=crop_override or crop_size,
            mean=mean,
            std=std,
        )
    except Exception:
        return TransformConfig(
            resize_size=resize_override or fallback.resize_size,
            crop_size=crop_override or fallback.crop_size,
            mean=fallback.mean,
            std=fallback.std,
        )


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, features.t()) / self.temperature
        labels = torch.arange(features.size(0), device=features.device)
        loss = F.cross_entropy(similarity, labels)
        return loss


def _initialize_linear_stack(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


def _set_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def apply_lora_to_linear_layers(
    model: nn.Module,
    target_suffixes: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    replaced: List[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent_name, child_name = module_name.rsplit(".", 1)
        parent_module = model.get_submodule(parent_name)
        _set_module(parent_module, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    return replaced


class LoraVisionBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_family: str,
        backbone_path: str,
        num_classes: int = 2,
        projection_dim: int = 512,
        classifier_dropout: float = 0.1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        local_files_only: bool = False,
        device: str | torch.device = "cuda",
        lora_targets: Sequence[str] | None = None,
    ):
        super().__init__()
        self.model_family = model_family
        self.backbone_path = backbone_path
        self.projection_dim = projection_dim
        self.local_files_only = local_files_only

        self.backbone = self._load_backbone(model_family, backbone_path, local_files_only)
        self.hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError(f"Cannot infer hidden_size for {model_family}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        targets = tuple(lora_targets) if lora_targets else DEFAULT_LORA_TARGETS[model_family]
        self.lora_modules = apply_lora_to_linear_layers(
            self.backbone,
            targets,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        _initialize_linear_stack(self.projection)
        _initialize_linear_stack(self.classifier)
        self.to(device)

    def _load_backbone(self, model_family: str, backbone_path: str, local_files_only: bool) -> nn.Module:
        if model_family == "clip_lora":
            return CLIPVisionModel.from_pretrained(backbone_path, local_files_only=local_files_only)
        if model_family == "vit_large_lora":
            return ViTModel.from_pretrained(backbone_path, local_files_only=local_files_only)
        raise ValueError(f"Unsupported model_family: {model_family}")

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.float()
        return outputs.last_hidden_state[:, 0].float()

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        features = self.extract_features(pixel_values)
        projected = self.projection(features)
        logits = self.classifier(projected)
        if return_features:
            return {
                "features": features,
                "projected_features": projected,
                "logits": logits,
            }
        return logits


class ImprovedDinoV3Adapter(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        projection_dim: int = 512,
        adapter_layers: int = 3,
        dropout_rate: float = 0.1,
        local_files_only: bool = True,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)
        for param in self.backbone.parameters():
            param.requires_grad = False

        projection_layers: List[nn.Module] = []
        current_dim = self.hidden_size
        for layer_index in range(adapter_layers):
            if layer_index == adapter_layers - 1:
                projection_layers.extend(
                    [
                        nn.Linear(current_dim, projection_dim),
                        nn.LayerNorm(projection_dim),
                    ]
                )
            else:
                next_dim = max(projection_dim, current_dim // 2)
                projection_layers.extend(
                    [
                        nn.Linear(current_dim, next_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(next_dim),
                    ]
                )
                current_dim = next_dim

        self.projection = nn.Sequential(*projection_layers)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim // 2, num_classes),
        )
        _initialize_linear_stack(self.projection)
        _initialize_linear_stack(self.classifier)
        self.to(device)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
            return features.float()

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        raw_features = self.extract_features(pixel_values)
        projected = self.projection(raw_features)
        logits = self.classifier(projected)
        if return_features:
            return {
                "logits": logits,
                "projected_features": projected,
                "raw_features": raw_features,
            }
        return logits

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_projection(self) -> None:
        for param in self.projection.parameters():
            param.requires_grad = True

    def unfreeze_classifier(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = True


class TeacherStudentNetwork(nn.Module):
    def __init__(
        self,
        dinov3_model_path: str,
        num_classes: int = 2,
        projection_dim: int = 512,
        student_dropout: float = 0.2,
        teacher_dropout: float = 0.1,
        local_files_only: bool = True,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.teacher = ImprovedDinoV3Adapter(
            model_path=dinov3_model_path,
            num_classes=num_classes,
            projection_dim=projection_dim,
            adapter_layers=3,
            dropout_rate=teacher_dropout,
            local_files_only=local_files_only,
            device=device,
        )
        self.student_projection = nn.Sequential(
            nn.Linear(self.teacher.hidden_size, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(student_dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.student_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(teacher_dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        _initialize_linear_stack(self.student_projection)
        _initialize_linear_stack(self.student_classifier)
        self.to(device)

    def forward_teacher(self, pixel_values: torch.Tensor):
        outputs = self.teacher(pixel_values, return_features=True)
        return outputs["projected_features"], outputs["logits"]

    def forward_student(self, pixel_values: torch.Tensor):
        raw_features = self.teacher.extract_features(pixel_values)
        projected = self.student_projection(raw_features)
        logits = self.student_classifier(projected)
        return projected, logits

    def freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad = False

    def unfreeze_teacher_head(self) -> None:
        self.teacher.unfreeze_projection()
        self.teacher.unfreeze_classifier()

    def unfreeze_student(self) -> None:
        for param in self.student_projection.parameters():
            param.requires_grad = True
        for param in self.student_classifier.parameters():
            param.requires_grad = True


class ImprovedTeacherStudentLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        alpha_distill: float = 1.0,
        alpha_cls: float = 1.0,
        alpha_feature: float = 0.5,
        alpha_simclr: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_cls = alpha_cls
        self.alpha_feature = alpha_feature
        self.alpha_simclr = alpha_simclr
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.simclr_loss = SimCLRLoss(temperature=temperature)

    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kl_loss = self.kl_loss(student_log_probs, teacher_probs)
        return kl_loss * (self.temperature ** 2)

    def feature_alignment_loss(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        cosine_sim = (student_norm * teacher_norm).sum(dim=1)
        return 1.0 - cosine_sim.mean()

    def forward(
        self,
        student_features: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_features: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mode: str = "student",
        student_features_aug: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if labels is None:
            raise ValueError("labels must be provided")

        if mode == "teacher":
            cls_loss = self.ce_loss(student_logits, labels)
            zero = torch.tensor(0.0, device=cls_loss.device)
            return {
                "total_loss": cls_loss,
                "cls_loss": cls_loss,
                "distill_loss": zero,
                "feature_loss": zero,
                "simclr_loss": zero,
            }

        if teacher_features is None or teacher_logits is None:
            raise ValueError("teacher_features and teacher_logits are required in student mode")

        cls_loss = self.ce_loss(student_logits, labels)
        distill_loss = self.distillation_loss(student_logits, teacher_logits)
        feature_loss = self.feature_alignment_loss(student_features, teacher_features)
        simclr_loss = torch.tensor(0.0, device=cls_loss.device)
        if student_features_aug is not None:
            simclr_loss = self.simclr_loss(torch.cat([student_features, student_features_aug], dim=0))

        total_loss = (
            self.alpha_cls * cls_loss
            + self.alpha_distill * distill_loss
            + self.alpha_feature * feature_loss
            + self.alpha_simclr * simclr_loss
        )
        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "distill_loss": distill_loss,
            "feature_loss": feature_loss,
            "simclr_loss": simclr_loss,
        }


class TeacherStudentEvalWrapper(nn.Module):
    def __init__(self, network: TeacherStudentNetwork, branch: str = "student"):
        super().__init__()
        if branch not in {"teacher", "student"}:
            raise ValueError("branch must be 'teacher' or 'student'")
        self.network = network
        self.branch = branch

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.branch == "teacher":
            _, logits = self.network.forward_teacher(pixel_values)
            return logits
        _, logits = self.network.forward_student(pixel_values)
        return logits


def create_lora_model_from_config(config: Dict[str, object], device: str | torch.device) -> LoraVisionBinaryClassifier:
    return LoraVisionBinaryClassifier(
        model_family=config["model_family"],
        backbone_path=config["backbone_path"],
        num_classes=int(config.get("num_classes", 2)),
        projection_dim=int(config.get("projection_dim", 512)),
        classifier_dropout=float(config.get("classifier_dropout", 0.1)),
        lora_rank=int(config.get("lora_rank", 8)),
        lora_alpha=float(config.get("lora_alpha", 16.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        local_files_only=bool(config.get("local_files_only", False)),
        device=device,
        lora_targets=config.get("lora_targets"),
    )


def serialize_transform_config(config: TransformConfig) -> Dict[str, object]:
    return asdict(config)
