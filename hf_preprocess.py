import random
from PIL import Image, ImageOps
from datasets import load_dataset
import os

TARGET_SIZE = 224
DEFAULT_VARIANTS = 3
SEED = 42

def random_square_then_resize(img, rng):
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

def transform_batch(examples):
    rng = random.Random(SEED)
    new_examples = {
        "image": [],
        "latitude": [],
        "longitude": [],
        "crop_size": [],
        "crop_left": [],
        "crop_top": []
    }

    for i in range(len(examples["image"])):
        img = examples["image"][i]
        img = ImageOps.exif_transpose(img).convert("RGB")
        
        for v_idx in range(DEFAULT_VARIANTS):
            cropped, c_size, c_left, c_top = random_square_then_resize(img, rng)
            
            new_examples["image"].append(cropped)
            new_examples["latitude"].append(examples["latitude"][i])
            new_examples["longitude"].append(examples["longitude"][i])
            new_examples["crop_size"].append(c_size)
            new_examples["crop_left"].append(c_left)
            new_examples["crop_top"].append(c_top)

    return new_examples

def main():
    ds = load_dataset("Willh96/image2gps_dataset")

    print("Starting preprocessing...")
    
    augmented_ds = ds.map(
        transform_batch,
        batched=True,
        batch_size=25,
        remove_columns=ds["train"].column_names,
        num_proc=16
    )

    output_path = "/home/ec2-user/image2gps/preprocessed_augmented_dataset"
    augmented_ds.save_to_disk(output_path)
    
    print(f"Success! Augmented dataset saved to {output_path}")
    print(f"New dataset size: {len(augmented_ds['train'])} examples")

if __name__ == "__main__":
    main()