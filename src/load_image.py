import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Check that a retinal image can be opened.")
    parser.add_argument("--image", default="data/sample_retina.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Next step: create data/ and add one retinal image, or pass --image /path/to/file.jpg")
        return

    image = Image.open(image_path).convert("RGB")
    print("Image loaded successfully")
    print("Path:", image_path)
    print("Size:", image.size)
    print("Mode:", image.mode)


if __name__ == "__main__":
    main()
