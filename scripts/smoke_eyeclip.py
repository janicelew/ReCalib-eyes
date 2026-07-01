from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recalib_eye.eyeclip_loader import load_eyeclip
from recalib_eye.prototypes import build_text_prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the local EyeCLIP checkpoint and run a small smoke test.")
    parser.add_argument("--eyeclip-repo", default="/Users/lewjanice/Documents/EyeCLIP")
    parser.add_argument("--checkpoint", default="/Users/lewjanice/Documents/EyeCLIP/eyeclip_visual.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--image", default=None, help="Optional image path for one-image inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_eyeclip(
        repo_path=args.eyeclip_repo,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print(f"EyeCLIP repo: {bundle.repo_path}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Device: {bundle.device}")
    print(f"Checkpoint tensors: {bundle.tensor_count}")
    print(f"Input resolution: {bundle.model.visual.input_resolution}")

    prompts = {
        "normal": ["color fundus, normal retina"],
        "dr": ["color fundus, diabetic retinopathy"],
    }
    text_features = build_text_prototypes(bundle.model, bundle.clip, prompts, bundle.device)
    print(f"Text prototype matrix: {tuple(text_features.shape)}")

    if args.image is None:
        print("Smoke test complete. Add --image data/sample_retina.jpg to test image inference.")
        return

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = bundle.preprocess(image).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        image_features = bundle.model.encode_image(image_tensor)
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = 100.0 * image_features @ text_features.T
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    print(f"Image: {image_path}")
    print(f"normal probability: {probs[0]:.4f}")
    print(f"dr probability: {probs[1]:.4f}")


if __name__ == "__main__":
    main()
