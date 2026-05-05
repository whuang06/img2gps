from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


TARGET_SIZE = 224
DEFAULT_VARIANTS = 5


def random_square_then_resize(
    img: Image.Image, rng: random.Random
) -> tuple[Image.Image, int, int, int]:
    width, height = img.size

    shortest_side = min(width, height)
    min_square = max(1, int(shortest_side * 0.5))
    square_size = rng.randint(min_square, shortest_side)
    max_left = width - square_size
    max_top = height - square_size

    left = rng.randint(0, max_left) if max_left > 0 else 0
    top = rng.randint(0, max_top) if max_top > 0 else 0

    square = img.crop((left, top, left + square_size, top + square_size))
    resized = square.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    return resized, square_size, left, top


def preprocess_images(
    input_root: Path,
    metadata_path: Path,
    output_root: Path,
    variants_per_image: int,
    seed: int | None,
) -> None:
    df = pd.read_csv(metadata_path)

    output_images_dir = output_root / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    output_rows = []

    for _, row in df.iterrows():
        relative_image_path = Path(str(row["image_file"]))
        source_image_path = input_root / relative_image_path

        if not source_image_path.exists():
            continue

        with Image.open(source_image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            orig_width, orig_height = img.size
            stem = source_image_path.stem

            for variant_idx in range(variants_per_image):
                cropped, crop_size, crop_left, crop_top = random_square_then_resize(img, rng)
                out_file_name = f"{stem}_{variant_idx}.jpg"
                out_relative = Path("images") / out_file_name
                out_path = output_root / out_relative
                cropped.save(out_path, format="JPEG", quality=95)

                output_rows.append(
                    {
                        "image_file": out_relative.as_posix(),
                        "source_image_file": relative_image_path.as_posix(),
                        "variant_index": variant_idx,
                        "Latitude": row["Latitude"],
                        "Longitude": row["Longitude"],
                        "orig_width": orig_width,
                        "orig_height": orig_height,
                        "crop_size": crop_size,
                        "crop_left": crop_left,
                        "crop_top": crop_top,
                        "resize_width": TARGET_SIZE,
                        "resize_height": TARGET_SIZE,
                    }
                )

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_root / "metadata.csv", index=False)

    print(f"Saved {len(output_rows)} preprocessed images to {output_images_dir}")
    print(f"Saved metadata to {output_root / 'metadata.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random square crops with side length between 50% and 100% of "
            "the shortest image side, resize to 224x224, then write new metadata."
        )
    )
    parser.add_argument(
        "--input-root",
        default="sample_data",
        help="Folder containing images and metadata.csv from previous step.",
    )
    parser.add_argument(
        "--output-root",
        default="preprocessed",
        help="Folder where preprocessed/images and metadata.csv are written.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=DEFAULT_VARIANTS,
        help="Number of random crops to generate per input image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible crops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    metadata_path = input_root / "metadata.csv"
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    preprocess_images(
        input_root=input_root,
        metadata_path=metadata_path,
        output_root=output_root,
        variants_per_image=args.variants,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
