from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = "hf://datasets/gydou/released_img/data/train-00000-of-00001.parquet"


def export_dataset(source: str, output_dir: Path) -> None:
	df = pd.read_parquet(source)

	images_dir = output_dir / "images"
	images_dir.mkdir(parents=True, exist_ok=True)

	records = []
	for idx, row in df.iterrows():
		image_obj = row["image"]
		image_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None

		if not image_bytes:
			continue

		file_name = f"img_{idx:05d}.jpg"
		image_path = images_dir / file_name
		image_path.write_bytes(image_bytes)

		records.append(
			{
				"image_file": f"images/{file_name}",
				"Latitude": row["Latitude"],
				"Longitude": row["Longitude"],
			}
		)

	metadata_df = pd.DataFrame(records)
	metadata_df.to_csv(output_dir / "metadata.csv", index=False)

	print(f"Exported {len(records)} samples to {output_dir}")
	print(f"Images: {images_dir}")
	print(f"Metadata: {output_dir / 'metadata.csv'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download Hugging Face parquet samples and save into a local folder."
	)
	parser.add_argument(
		"--source",
		default=DEFAULT_SOURCE,
		help="Parquet path (supports hf://datasets/...).",
	)
	parser.add_argument(
		"--output-dir",
		default="sample_data",
		help="Directory where images and metadata.csv are written.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	export_dataset(args.source, output_dir)


if __name__ == "__main__":
	main()
