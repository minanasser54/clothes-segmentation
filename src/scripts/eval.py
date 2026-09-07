"""
eval.py

Usage:
    python eval.py --run-name unet_v1
    python eval.py --run-name unet_v1 --checkpoint final_model.weights.h5
    python eval.py --run-name unet_v1 --data-dir 1_processed --num-samples 8
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import config
from model import build_unet, build_unet_plus_plus



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net on the test split.")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Subfolder name under MODEL_ROOT containing the checkpoint to evaluate.")
    parser.add_argument("--checkpoint", type=str, default="best_model.weights.h5",
                        help="Checkpoint filename inside the run folder.")
    parser.add_argument("--data-dir", type=str, default="1_processed",
                        help="Folder name under DATA_ROOT to evaluate on "
                            "(test split is never read from '1_augmented' by default, "
                            "since augmentation only applies to train).")
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Number of prediction visualizations to save.")
    return parser.parse_args()



def list_pairs(split_dir):
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    assert len(image_files) == len(mask_files), \
        f"Mismatched counts in {split_dir}: {len(image_files)} images vs {len(mask_files)} masks"
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]
    return image_paths, mask_paths


def _decode_image(path):
    raw = tf.io.read_file(path)
    image = tf.image.decode_png(raw, channels=3, dtype=tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [config.IMG_SIZE, config.IMG_SIZE, 3])
    return image


def _decode_mask(path):
    raw = tf.io.read_file(path)
    mask = tf.image.decode_png(raw, channels=1, dtype=tf.uint8)
    mask = tf.ensure_shape(mask, [config.IMG_SIZE, config.IMG_SIZE, 1])
    return mask


def build_dataset(image_paths, mask_paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(
        lambda img_p, mask_p: (_decode_image(img_p), _decode_mask(mask_p)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_class_names(data_root):
    labels_path = os.path.join(data_root, "labels.csv")
    if not os.path.exists(labels_path):
        return None
    import csv
    names = []
    with open(labels_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                names.append(row[-1])
    return names



def evaluate_test_set(model, test_ds, num_classes, background_class_id=0):

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    dice_numerator = np.zeros(num_classes, dtype=np.int64)   # 2 * intersection
    dice_denominator = np.zeros(num_classes, dtype=np.int64)  # |pred| + |true|
    correct_pixels = 0
    total_pixels = 0

    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=-1)
        masks = masks.numpy().squeeze(axis=-1)

        correct_pixels += (preds == masks).sum()
        total_pixels += masks.size

        for c in range(num_classes):
            pred_c = preds == c
            mask_c = masks == c

            inter = np.logical_and(pred_c, mask_c).sum()
            uni = np.logical_or(pred_c, mask_c).sum()

            intersection[c] += inter
            union[c] += uni
            dice_numerator[c] += 2 * inter
            dice_denominator[c] += pred_c.sum() + mask_c.sum()

    iou_per_class = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0
    )
    dice_per_class = np.divide(
        dice_numerator, dice_denominator,
        out=np.zeros_like(dice_numerator, dtype=float), where=dice_denominator != 0
    )

    present = union > 0 
    overall_miou = iou_per_class[present].mean() if present.any() else 0.0
    mean_dice = dice_per_class[present].mean() if present.any() else 0.0

    foreground_mask = present.copy()
    if 0 <= background_class_id < num_classes:
        foreground_mask[background_class_id] = False
    foreground_miou = iou_per_class[foreground_mask].mean() if foreground_mask.any() else 0.0

    pixel_accuracy = correct_pixels / total_pixels

    metrics_summary = {
        "overall_miou": float(overall_miou),
        "foreground_miou": float(foreground_miou),
        "mean_dice": float(mean_dice),
        "pixel_accuracy": float(pixel_accuracy),
    }

    return metrics_summary, iou_per_class, dice_per_class, present


def save_prediction_visuals(model, test_ds, out_path, num_samples):
    images, masks = next(iter(test_ds))
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=-1)

    n_show = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))
    if n_show == 1:
        axes = axes.reshape(3, 1)

    for i in range(n_show):
        axes[0, i].imshow(images[i].numpy())
        axes[0, i].set_title("image")
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i].numpy().squeeze(-1), cmap="tab20")
        axes[1, i].set_title("ground truth")
        axes[1, i].axis("off")

        axes[2, i].imshow(preds[i], cmap="tab20")
        axes[2, i].set_title("prediction")
        axes[2, i].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()

    run_dir = os.path.join(config.MODEL_ROOT, args.run_name)
    checkpoint_path = os.path.join(run_dir, args.checkpoint)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    data_root = os.path.join(config.DATA_ROOT, args.data_dir)
    test_dir = os.path.join(data_root, "test")

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Evaluating on: {test_dir}")

    test_images, test_masks = list_pairs(test_dir)
    print(f"Test pairs: {len(test_images)}")
    test_ds = build_dataset(test_images, test_masks, args.batch_size)

    class_names = load_class_names(data_root)

    run_config_path = os.path.join(run_dir, "run_config.json")
    architecture = "unet_plus_plus"
    if os.path.exists(run_config_path):
        with open(run_config_path) as f:
            saved_run_config = json.load(f)
        architecture = saved_run_config.get("architecture", architecture)
    else:
        print(f"Warning: {run_config_path} not found — assuming architecture="
              f"'{architecture}'. If this run used a different architecture, "
              f"weight loading will fail.")

    print(f"Building architecture: {architecture}")
    if architecture == "unet_plus_plus":
        model = build_unet_plus_plus(
            num_classes=config.NUM_CLASSES,
            img_size=config.IMG_SIZE,
            features=tuple(getattr(config, "UNET_PP_FEATURES", (32, 64, 128, 256, 512))),
        )
    else:
        model = build_unet(
            num_classes=config.NUM_CLASSES,
            img_size=config.IMG_SIZE,
            features=tuple(config.UNET_FEATURES),
        )
    model.load_weights(checkpoint_path)

    metrics_summary, iou_per_class, dice_per_class, present = evaluate_test_set(
        model, test_ds, config.NUM_CLASSES, background_class_id=0
    )

    print(f"Overall mIoU:    {metrics_summary['overall_miou']:.4f}")
    print(f"Foreground mIoU: {metrics_summary['foreground_miou']:.4f}")
    print(f"Mean Dice:       {metrics_summary['mean_dice']:.4f}")
    print(f"Pixel accuracy:  {metrics_summary['pixel_accuracy']:.4f}")

    per_class_metrics = {}
    for c in range(config.NUM_CLASSES):
        label = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class_metrics[label] = {
            "iou": float(iou_per_class[c]),
            "dice": float(dice_per_class[c]),
            "present_in_test_set": bool(present[c]),
        }

    metrics = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "num_test_pairs": len(test_images),
        "num_classes": config.NUM_CLASSES,
        "overall_miou": metrics_summary["overall_miou"],
        "foreground_miou": metrics_summary["foreground_miou"],
        "mean_dice": metrics_summary["mean_dice"],
        "pixel_accuracy": metrics_summary["pixel_accuracy"],
        "per_class_metrics": per_class_metrics,
    }
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    visuals_path = os.path.join(eval_dir, "sample_predictions.png")
    save_prediction_visuals(model, test_ds, visuals_path, args.num_samples)
    print(f"Saved prediction visuals to: {visuals_path}")


if __name__ == "__main__":
    main()