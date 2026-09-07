"""
train.py
Usage:
    python train.py --epochs 50 --batch-size 16 --lr 1e-3 --run-name unet_v1
    python train.py --data-dir "1_processed" --run-name unet_no_aug
"""

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import config
from model import build_unet, build_unet_plus_plus, dice_ce_loss, MeanIoUMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for clothes segmentation.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--run-name", type=str, default="unet_run",
                         help="Subfolder name under MODEL_ROOT for this run's outputs.")
    parser.add_argument("--data-dir", type=str, default="2_augmented",
                         help="Folder name under DATA_ROOT to train from "
                              "(e.g. '2_augmented' or '1_processed').")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Optional path to a .weights.h5 checkpoint to resume training from.")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--cache-to-disk", action="store_true",
                         help="Cache decoded images/masks to local Colab disk instead of RAM. "
                              "Use this if training crashes with an OOM error when caching "
                              "in memory (larger datasets / higher IMG_SIZE).")
    parser.add_argument("--architecture", type=str, default="unet_plus_plus",
                         choices=["unet", "unet_plus_plus"],
                         help="Which architecture to train. Defaults to U-Net++ "
                              "(nested skip pathways), chosen as the primary "
                              "model for this task's fine boundary/small-region "
                              "clothing segmentation. Pass 'unet' to train the "
                              "plain baseline for comparison in the report.")
    parser.add_argument("--log-every-n-steps", type=int, default=10,
                         help="Print an in-epoch progress line every N training steps.")
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


def build_dataset(image_paths, mask_paths, batch_size, shuffle, cache=True, cache_path=None):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(
        lambda img_p, mask_p: (_decode_image(img_p), _decode_mask(mask_p)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,  # allow out-of-order completion for parallel decode throughput
    )
    if cache:
        ds = ds.cache(cache_path) if cache_path else ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(image_paths)), seed=config.RANDOM_SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(mask_paths, num_classes, sample_size=300):
    sample = mask_paths[:sample_size]
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for path in sample:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vals, counts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, counts):
            if v < num_classes:
                pixel_counts[v] += c

    pixel_counts = np.maximum(pixel_counts, 1)  # avoid div-by-zero for unseen classes
    weights = 1.0 / np.log(1.02 + pixel_counts / pixel_counts.sum())
    weights = weights / weights.sum() * num_classes  # normalize to mean 1.0
    return weights.tolist()


class InEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._epoch_start_time = None
        self._epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch + 1
        self._epoch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        if step % self.log_every_n_steps != 0:
            return
        logs = logs or {}
        elapsed = time.time() - self._epoch_start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0.0
        metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(f"  [epoch {self._epoch}] step {step} "
              f"({steps_per_sec:.2f} steps/s, {elapsed:.1f}s elapsed) - {metrics_str}")


def main():
    args = parse_args()

    run_dir = os.path.join(config.MODEL_ROOT, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    data_root = os.path.join(config.DATA_ROOT, args.data_dir)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")

    train_images, train_masks = list_pairs(train_dir)
    val_images, val_masks = list_pairs(val_dir)
    print(f"Train pairs: {len(train_images)}, Val pairs: {len(val_images)}")

    train_cache_path = "/content/cache_train" if args.cache_to_disk else None
    val_cache_path = "/content/cache_val" if args.cache_to_disk else None

    train_ds = build_dataset(train_images, train_masks, args.batch_size, shuffle=True,
                              cache_path=train_cache_path)
    val_ds = build_dataset(val_images, val_masks, args.batch_size, shuffle=False,
                            cache_path=val_cache_path)

    print("Computing class weights from a sample of training masks...")
    class_weights = compute_class_weights(train_masks, config.NUM_CLASSES)
    with open(os.path.join(run_dir, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=2)

    print(f"Building architecture: {args.architecture}")
    if args.architecture == "unet_plus_plus":
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
    print(f"Model param count: {model.count_params():,}")

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model.load_weights(args.resume_from)

    loss_fn = dice_ce_loss(
        num_classes=config.NUM_CLASSES,
        dice_weight=config.DICE_WEIGHT,
        class_weights=class_weights,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[MeanIoUMetric(num_classes=config.NUM_CLASSES), "accuracy"],
    )

    best_ckpt_path = os.path.join(run_dir, "best_model.weights.h5")

    callbacks = [
        InEpochLogger(log_every_n_steps=args.log_every_n_steps),

        # Save the model with the best validation IoU
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt_path,
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),

        # Stop when validation IoU stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mean_iou",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),

        # Reduce LR when validation IoU plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_iou",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, "training_log.csv")
        ),
    ]
 
    try:
        import tensorboard 
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "tensorboard"))
        )
    except ImportError:
        print("tensorboard package not installed — skipping TensorBoard logging "
              "(training_log.csv and history.json still capture full metrics).")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False, 
    )

    final_model_path = os.path.join(run_dir, "final_model.weights.h5")
    model.save_weights(final_model_path)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    run_config = vars(args)
    run_config["img_size"] = config.IMG_SIZE
    run_config["num_classes"] = config.NUM_CLASSES
    run_config["dice_weight"] = config.DICE_WEIGHT
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nDone. Run outputs saved to: {run_dir}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()