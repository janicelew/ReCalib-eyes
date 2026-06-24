from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


DR_GRADE_CLASS_NAMES = [
    "grade_0_normal",
    "grade_1_mild",
    "grade_2_moderate",
    "grade_3_severe",
    "grade_4_proliferative",
]
DR_REFERABLE_CLASS_NAMES = ["non_referable", "referable"]
NON_REFERABLE_GRADE_INDICES = [0, 1]
REFERABLE_GRADE_INDICES = [2, 3, 4]


def build_text_prototypes(
    model,
    clip_module,
    prompts_by_class: Mapping[str, Sequence[str]],
    device,
) -> torch.Tensor:
    prototypes = []
    model.eval()
    with torch.no_grad():
        for class_name, prompts in prompts_by_class.items():
            if not prompts:
                raise ValueError(f"No prompts configured for class {class_name!r}")
            tokens = clip_module.tokenize(list(prompts)).to(device)
            features = model.encode_text(tokens)
            features = F.normalize(features.float(), dim=-1)
            prototype = F.normalize(features.mean(dim=0), dim=-1)
            prototypes.append(prototype)
    return torch.stack(prototypes).to(device)


def dr_referable_probs_from_grade_logits(grade_logits: torch.Tensor) -> torch.Tensor:
    grade_probs = torch.softmax(grade_logits, dim=-1)
    return grade_probs[:, REFERABLE_GRADE_INDICES].sum(dim=-1)


def dr_referable_probs_from_binary_logits(binary_logits: torch.Tensor) -> torch.Tensor:
    binary_probs = torch.softmax(binary_logits, dim=-1)
    return binary_probs[:, 1]


def image_text_logits(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: float = 100.0) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    return logit_scale * image_features @ text_features.T
