"""Load and preprocess the CelebA face dataset (crop, resize, normalize-ready).

Reads the CelebA parquet shard(s) in data/raw, center-crops each aligned face,
resizes it to IMAGE_SIZE x IMAGE_SIZE and stores everything as one uint8 array
(data/faces_64.npy) so training can load the whole dataset into memory at once.
Normalization to [-1, 1] happens in train.py when batches are fed to the GAN.

Download the data first (done automatically if missing):
    huggingface: nielsr/CelebA-faces -> data/raw/data/train-0000X-of-00003.parquet
"""
import argparse
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

RAW_DIR = os.path.join("data", "raw")
IMAGE_SIZE = 64
CROP_SIZE = 140  # CelebA aligned images are 178x218; a 140px center crop keeps the face


def download(num_shards):
    from huggingface_hub import hf_hub_download

    for i in range(num_shards):
        hf_hub_download(
            "nielsr/CelebA-faces",
            f"data/train-{i:05d}-of-00003.parquet",
            repo_type="dataset",
            local_dir=RAW_DIR,
        )


def preprocess(img_bytes, image_size):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    # Center crop slightly above the middle, where the face sits in aligned CelebA
    left = (w - CROP_SIZE) // 2
    top = (h - CROP_SIZE) // 2 - 10
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    img = img.resize((image_size, image_size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1, help="parquet shards to use (1-3, ~67k faces each)")
    parser.add_argument("--max-images", type=int, default=0, help="limit number of images (0 = all)")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(RAW_DIR, "data", "*.parquet")))[: args.shards]
    if len(files) < args.shards:
        download(args.shards)
        files = sorted(glob.glob(os.path.join(RAW_DIR, "data", "*.parquet")))[: args.shards]

    faces = []
    for path in files:
        table = pq.read_table(path, columns=["image"])
        for row in tqdm(table.column("image").to_pylist(), desc=os.path.basename(path)):
            faces.append(preprocess(row["bytes"], args.image_size))
            if args.max_images and len(faces) >= args.max_images:
                break
        if args.max_images and len(faces) >= args.max_images:
            break

    faces = np.stack(faces)
    out = os.path.join("data", f"faces_{args.image_size}.npy")
    np.save(out, faces)
    print(f"Saved {faces.shape} -> {out}")

    # Preview grid of real faces for the report
    os.makedirs("outputs", exist_ok=True)
    grid = faces[:64].reshape(8, 8, args.image_size, args.image_size, 3)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(8 * args.image_size, 8 * args.image_size, 3)
    Image.fromarray(grid).save(os.path.join("outputs", "real_faces.png"))


if __name__ == "__main__":
    main()
