from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

DATA_DIR = "data/images"

def download_test_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading Oxford-IIIT Pet test split")

    dataset = load_dataset("timm/oxford-iiit-pet", split="test")

    for i, sample in enumerate(tqdm(dataset, desc="Saving images", ncols=80)):
        try:
            img = sample["image"].convert("RGB")   # convert RGBA → RGB
            label = sample["label"]
            filename = f"{i}_{label}.jpg"
            img.save(os.path.join(DATA_DIR, filename))
        except Exception as e:
            print(f"⚠️ Skipping image {i} ({e})")

    print(f"Saved {len(os.listdir(DATA_DIR))} images to {DATA_DIR}")

if __name__ == "__main__":
    download_test_data()