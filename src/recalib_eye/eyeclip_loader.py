from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_EYECLIP_REPO = Path("/Users/lewjanice/Documents/EyeCLIP")
DEFAULT_CHECKPOINT = DEFAULT_EYECLIP_REPO / "eyeclip_visual.pt"


@dataclass
class EyeCLIPBundle:
    model: Any
    preprocess: Any
    clip: Any
    device: torch.device
    repo_path: Path
    checkpoint_path: Path
    tensor_count: int


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def add_eyeclip_repo_to_path(repo_path: str | Path | None = None) -> Path:
    path = Path(repo_path or os.environ.get("EYECLIP_REPO", DEFAULT_EYECLIP_REPO)).expanduser().resolve()
    if not (path / "eyeclip").is_dir():
        raise FileNotFoundError(
            f"EyeCLIP package not found at {path}. Expected an 'eyeclip' folder there."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def count_tensors(obj: Any) -> int:
    if not isinstance(obj, dict):
        return 0
    return sum(isinstance(value, torch.Tensor) for value in obj.values())


def extract_state_dict(raw_checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(raw_checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(raw_checkpoint)!r}")

    candidates: list[dict[str, Any]] = []
    for key in ("model_state_dict", "model", "state_dict", "module", "net"):
        value = raw_checkpoint.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    candidates.append(raw_checkpoint)
    candidates.extend(value for value in raw_checkpoint.values() if isinstance(value, dict))
    candidates = [candidate for candidate in candidates if count_tensors(candidate) > 0]
    if not candidates:
        raise ValueError("No tensor state_dict found in checkpoint.")

    state_dict = max(candidates, key=count_tensors)
    return {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def load_eyeclip(
    repo_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> EyeCLIPBundle:
    repo = add_eyeclip_repo_to_path(repo_path)
    ckpt = Path(checkpoint_path or os.environ.get("EYECLIP_CKPT", DEFAULT_CHECKPOINT)).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"EyeCLIP checkpoint not found: {ckpt}")

    target_device = resolve_device(device)

    from eyeclip import clip
    from eyeclip.model import build_model

    raw = torch.load(ckpt, map_location="cpu")
    state_dict = extract_state_dict(raw)
    model = build_model(dict(state_dict)).to(target_device)
    if target_device.type == "cpu":
        model.float()
    model.eval()

    preprocess = clip._transform(model.visual.input_resolution)
    return EyeCLIPBundle(
        model=model,
        preprocess=preprocess,
        clip=clip,
        device=target_device,
        repo_path=repo,
        checkpoint_path=ckpt,
        tensor_count=len(state_dict),
    )
