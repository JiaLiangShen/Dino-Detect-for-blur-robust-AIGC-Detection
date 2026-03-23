from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        logits = torch.matmul(anchors, positives.T) / self.temperature
        labels = torch.arange(anchors.size(0), device=anchors.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class TeacherStudentLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        alpha_distill: float = 1.0,
        alpha_cls: float = 1.0,
        alpha_feature: float = 0.5,
        alpha_contrastive: float = 0.3,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_cls = alpha_cls
        self.alpha_feature = alpha_feature
        self.alpha_contrastive = alpha_contrastive
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.pair_contrastive = PairContrastiveLoss(temperature=temperature)

    def forward_teacher(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_loss = self.cross_entropy(logits, labels)
        zero = torch.zeros((), device=logits.device)
        return {
            "total_loss": cls_loss,
            "cls_loss": cls_loss,
            "distill_loss": zero,
            "feature_loss": zero,
            "contrastive_loss": zero,
        }

    def forward_student(
        self,
        student_features: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_loss = self.cross_entropy(student_logits, labels)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature**2)

        student_norm = F.normalize(student_features, dim=1)
        teacher_norm = F.normalize(teacher_features, dim=1)
        feature_loss = 1.0 - (student_norm * teacher_norm).sum(dim=1).mean()
        contrastive_loss = self.pair_contrastive(student_features, teacher_features)

        total_loss = (
            self.alpha_cls * cls_loss
            + self.alpha_distill * distill_loss
            + self.alpha_feature * feature_loss
            + self.alpha_contrastive * contrastive_loss
        )
        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "distill_loss": distill_loss,
            "feature_loss": feature_loss,
            "contrastive_loss": contrastive_loss,
        }
