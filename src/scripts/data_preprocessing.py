"""
data_preprocessing.py

- read images > remap masks > split > resize > write 

Expects the raw folder to look like:
    Data/0_raw/
        png_images/IMAGES/img_0001.png, img_0002.png, ...
        png_masks/MASKS/seg_0001.png, seg_0002.png, ...
        labels.csv

Produces:
    Data/1_processed/
        train/images/*.png   train/masks/*.png   (masks remapped to 9 superclasses)
        val/images/*.png     val/masks/*.png
        test/images/*.png    test/masks/*.png
        labels.csv           (9-superclass names, replaces the original 59-class file)

Usage:
    %run "/content/drive/MyDrive/Colab Notebooks/clothes-segmentation/scripts/data_preprocessing.py"
"""

import os
import re
import csv
import sys
import shutil
from configs import config
import cv2
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from class_mapping import remap_mask, load_superclass_names, NUM_SUPERCLASSES


RAW_DIR = os.path.join(config.PROJECT_ROOT, "Data", "0_raw")
PROCESSED_DIR = os.path.join(config.PROJECT_ROOT, "Data", "1_processed")

IMAGES_DIR = os.path.join(RAW_DIR, "png_images", "IMAGES")
MASKS_DIR = os.path.join(RAW_DIR, "png_masks", "MASKS")

IMG_SIZE = config.IMG_SIZE
VAL_SIZE = config.VAL_SIZE
TEST_SIZE = config.TEST_SIZE
RANDOM_SEED = config.RANDOM_SEED


def get_id(filename):
    digits = re.findall(r"\d+", filename)
    return digits[-1] if digits else filename


def find_image_mask_pairs(images_dir, masks_dir):
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    image_by_id = {get_id(f): f for f in image_files}
    mask_by_id = {get_id(f): f for f in mask_files}

    common_ids = sorted(set(image_by_id) & set(mask_by_id))
    missing_images = set(mask_by_id) - set(image_by_id)
    missing_masks = set(image_by_id) - set(mask_by_id)

    if missing_images:
        print(f"Warning: {len(missing_images)} masks have no matching image, skipping.")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images have no matching mask, skipping.")

    pairs = [(image_by_id[i], mask_by_id[i]) for i in common_ids]
    return pairs


def split_pairs(pairs, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=RANDOM_SEED):
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=(val_size + test_size), random_state=seed
    )
    relative_test_size = test_size / (val_size + test_size)
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=relative_test_size, random_state=seed
    )
    return train_pairs, val_pairs, test_pairs


def resize_image(path, img_size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return image


def resize_and_remap_mask(path, img_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = remap_mask(mask)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask


def process_and_save_split(pairs, split_name, images_dir, masks_dir, out_dir, img_size):
    split_images_dir = os.path.join(out_dir, split_name, "images")
    split_masks_dir = os.path.join(out_dir, split_name, "masks")
    os.makedirs(split_images_dir, exist_ok=True)
    os.makedirs(split_masks_dir, exist_ok=True)

    for image_fname, mask_fname in pairs:
        image = resize_image(os.path.join(images_dir, image_fname), img_size)
        mask = resize_and_remap_mask(os.path.join(masks_dir, mask_fname), img_size)

        cv2.imwrite(os.path.join(split_images_dir, image_fname), image)
        cv2.imwrite(os.path.join(split_masks_dir, mask_fname), mask)

    print(f"{split_name}: saved {len(pairs)} pairs to {os.path.join(out_dir, split_name)}")


def write_superclass_labels_file(out_dir):
    names = load_superclass_names()
    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label"])
        for i, name in enumerate(names):
            writer.writerow([i, name])
    print(f"Wrote {len(names)}-superclass labels.csv to {labels_path}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pairs = find_image_mask_pairs(IMAGES_DIR, MASKS_DIR)
    print(f"Found {len(pairs)} matched image/mask pairs.")
    print(f"Remapping masks from 59 original classes to {NUM_SUPERCLASSES} superclasses.")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"Split -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    process_and_save_split(train_pairs, "train", IMAGES_DIR, MASKS_DIR, PROCESSED_DIR, IMG_SIZE)
    process_and_save_split(val_pairs, "val", IMAGES_DIR, MASKS_DIR, PROCESSED_DIR, IMG_SIZE)
    process_and_save_split(test_pairs, "test", IMAGES_DIR, MASKS_DIR, PROCESSED_DIR, IMG_SIZE)

    write_superclass_labels_file(PROCESSED_DIR)

    print(f"\nDone. Processed data (9-superclass masks) is at: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()