import os
import shutil
import kagglehub
from configs import config



RAW_DIR = os.path.join(config.DATA_ROOT, "0_raw")
os.makedirs(RAW_DIR, exist_ok=True)

ds_path = kagglehub.dataset_download(config.DATASET_URL)

# Copy dataset to Drive
if not os.path.exists(os.path.join(RAW_DIR, "png_images")):
    shutil.copytree(
        os.path.join(ds_path, "png_images"),
        os.path.join(RAW_DIR, "png_images")
    )

if not os.path.exists(os.path.join(RAW_DIR, "png_masks")):
    shutil.copytree(
        os.path.join(ds_path, "png_masks"),
        os.path.join(RAW_DIR, "png_masks")
    )

if not os.path.exists(os.path.join(RAW_DIR, "labels.csv")):
    shutil.copy(
        os.path.join(ds_path, "labels.csv"),
        os.path.join(RAW_DIR, "labels.csv")
    )

# Quick check
images = sorted(os.listdir(os.path.join(RAW_DIR, "png_images", "IMAGES")))
masks = sorted(os.listdir(os.path.join(RAW_DIR, "png_masks", "MASKS")))

print(f"Images: {len(images)}, Masks: {len(masks)}")
assert len(images) == len(masks), "Image/mask count mismatch!"