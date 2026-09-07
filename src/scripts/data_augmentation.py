"""
data_augmentation.py

Produces:
    Data/1_augmented/
        train/images/*.png   train/masks/*.png   (originals + augmented copies)
        val/images/*.png     val/masks/*.png     (copied as-is)
        test/images/*.png    test/masks/*.png    (copied as-is)

Usage:
    %run "/content/drive/MyDrive/Colab Notebooks/clothes-segmentation/scripts/data_augmentation.py"
"""

import os
import shutil

import cv2
import numpy as np
from configs import config

PROJECT_PATH = config.PROJECT_ROOT
PROCESSED_DIR = os.path.join(PROJECT_PATH, "Data", "1_processed")
AUGMENTED_DIR = os.path.join(PROJECT_PATH, "Data", "2_augmented")


NUM_AUGMENTED_COPIES = config.NUM_AUGMENTED_COPIES
RANDOM_SEED = config.RANDOM_SEED
np.random.seed(RANDOM_SEED)


def flip_horizontal(image, mask):
    return cv2.flip(image, 1), cv2.flip(mask, 1)


def random_crop_resize(image, mask, crop_fraction=0.85):
    h, w = image.shape[:2]
    crop_h, crop_w = int(h * crop_fraction), int(w * crop_fraction)

    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)

    image_cropped = image[top:top + crop_h, left:left + crop_w]
    mask_cropped = mask[top:top + crop_h, left:left + crop_w]

    image_resized = cv2.resize(image_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return image_resized, mask_resized


def adjust_brightness(image, mask, delta_range=(-25, 25)):
    delta = np.random.uniform(*delta_range)
    image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return image, mask


def adjust_contrast(image, mask, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(*factor_range)
    mean = image.mean()
    image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return image, mask


def adjust_saturation(image, mask, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image, mask


def apply_random_augmentation(image, mask):
    if np.random.rand() < 0.5:
        image, mask = flip_horizontal(image, mask)

    if np.random.rand() < 0.4:
        image, mask = random_crop_resize(image, mask)

    if np.random.rand() < 0.5:
        image, mask = adjust_brightness(image, mask)

    if np.random.rand() < 0.5:
        image, mask = adjust_contrast(image, mask)

    if np.random.rand() < 0.3:
        image, mask = adjust_saturation(image, mask)

    return image, mask


def list_pairs(split_dir):
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    assert len(image_files) == len(mask_files), \
        f"Mismatched counts in {split_dir}: {len(image_files)} images vs {len(mask_files)} masks"
    return list(zip(image_files, mask_files)), images_dir, masks_dir


def copy_split_unchanged(split_name, processed_dir, augmented_dir):
    src_dir = os.path.join(processed_dir, split_name)
    dst_dir = os.path.join(augmented_dir, split_name)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    n = len(os.listdir(os.path.join(dst_dir, "images")))
    print(f"{split_name}: copied {n} pairs unchanged to {dst_dir}")


def augment_split(split_name, processed_dir, augmented_dir, num_copies):
    src_dir = os.path.join(processed_dir, split_name)
    pairs, images_dir, masks_dir = list_pairs(src_dir)

    out_images_dir = os.path.join(augmented_dir, split_name, "images")
    out_masks_dir = os.path.join(augmented_dir, split_name, "masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    for image_fname, mask_fname in pairs:
        image = cv2.imread(os.path.join(images_dir, image_fname), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(masks_dir, mask_fname), cv2.IMREAD_GRAYSCALE)

        # Save the original, unaltered pair too.
        cv2.imwrite(os.path.join(out_images_dir, image_fname), image)
        cv2.imwrite(os.path.join(out_masks_dir, mask_fname), mask)

        base_image, ext_image = os.path.splitext(image_fname)
        base_mask, ext_mask = os.path.splitext(mask_fname)

        for i in range(num_copies):
            aug_image, aug_mask = apply_random_augmentation(image.copy(), mask.copy())
            aug_image_fname = f"{base_image}_aug{i}{ext_image}"
            aug_mask_fname = f"{base_mask}_aug{i}{ext_mask}"
            cv2.imwrite(os.path.join(out_images_dir, aug_image_fname), aug_image)
            cv2.imwrite(os.path.join(out_masks_dir, aug_mask_fname), aug_mask)

    total = len(pairs) * (1 + num_copies)
    print(f"{split_name}: saved {total} pairs ({len(pairs)} original + "
          f"{len(pairs) * num_copies} augmented) to {os.path.join(augmented_dir, split_name)}")


def main():
    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    augment_split("train", PROCESSED_DIR, AUGMENTED_DIR, NUM_AUGMENTED_COPIES)
    copy_split_unchanged("val", PROCESSED_DIR, AUGMENTED_DIR)
    copy_split_unchanged("test", PROCESSED_DIR, AUGMENTED_DIR)

    labels_src = os.path.join(PROCESSED_DIR, "labels.csv")
    if os.path.exists(labels_src):
        shutil.copy(labels_src, os.path.join(AUGMENTED_DIR, "labels.csv"))

    print(f"\nDone. Augmented data is at: {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()