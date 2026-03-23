from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOAdapter(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        num_classes: int = 2,
        projection_dim: int = 512,
        adapter_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not backbone_path:
            raise ValueError("A valid backbone_path is required.")

        self.backbone = AutoModel.from_pretrained(
            backbone_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        projection_layers: list[nn.Module] = []
        current_dim = self.hidden_size
        for layer_index in range(adapter_layers):
            is_last = layer_index == adapter_layers - 1
            if is_last:
                projection_layers.extend([nn.Linear(current_dim, projection_dim), nn.LayerNorm(projection_dim)])
            else:
                next_dim = max(projection_dim, current_dim // 2)
                projection_layers.extend(
                    [
                        nn.Linear(current_dim, next_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.LayerNorm(next_dim),
                    ]
                )
                current_dim = next_dim

        self.projection = nn.Sequential(*projection_layers)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        self._initialize_trainable_layers()

    def _initialize_trainable_layers(self) -> None:
        def init_layer(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.projection.apply(init_layer)
        self.classifier.apply(init_layer)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output.float()
            return outputs.last_hidden_state[:, 0].float()

    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        raw_features = self.extract_features(pixel_values)
        projected_features = self.projection(raw_features)
        logits = self.classifier(projected_features)
        if return_features:
            return {
                "raw_features": raw_features,
                "projected_features": projected_features,
                "logits": logits,
            }
        return logits


class TeacherStudentModel(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        num_classes: int = 2,
        projection_dim: int = 512,
        adapter_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.teacher = DINOAdapter(
            backbone_path=backbone_path,
            num_classes=num_classes,
            projection_dim=projection_dim,
            adapter_layers=adapter_layers,
            dropout=dropout,
        )
        self.student_projection = nn.Sequential(
            nn.Linear(self.teacher.hidden_size, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.student_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, num_classes),
        )
        self._initialize_student_layers()

    def _initialize_student_layers(self) -> None:
        def init_layer(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.student_projection.apply(init_layer)
        self.student_classifier.apply(init_layer)

    def forward_teacher(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.teacher(pixel_values, return_features=True)
        return outputs["projected_features"], outputs["logits"]

    def forward_student(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_features = self.teacher.extract_features(pixel_values)
        projected_features = self.student_projection(raw_features)
        logits = self.student_classifier(projected_features)
        return projected_features, logits

    def teacher_parameters(self):
        return list(self.teacher.projection.parameters()) + list(self.teacher.classifier.parameters())

    def student_parameters(self):
        return list(self.student_projection.parameters()) + list(self.student_classifier.parameters())


class TeacherWrapper(nn.Module):
    def __init__(self, model: TeacherStudentModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_teacher(pixel_values)


class StudentWrapper(nn.Module):
    def __init__(self, model: TeacherStudentModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_student(pixel_values)
