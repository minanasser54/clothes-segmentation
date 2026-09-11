"""
inference.py
Usage:
    python inference.py --run-name unet_pp_v4_9classes --image path/to/photo.jpg

    # custom checkpoint / output location:
    python inference.py --run-name unet_pp_v4_9classes --checkpoint final_model.weights.h5 \\
        --image path/to/photo.jpg --output path/to/save_mask.png
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import config
from model import build_unet, build_unet_plus_plus


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on a single image.")
    parser.add_argument("--run-name", type=str, required=True,
                         help="Subfolder name under MODEL_ROOT containing the checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="best_model.weights.h5",
                         help="Checkpoint filename inside the run folder.")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to the input image.")
    parser.add_argument("--output", type=str, default=None,
                         help="Path to save the predicted mask PNG. "
                              "Defaults to '<image_name>_mask.png' next to the input image.")
    parser.add_argument("--save-overlay", action="store_true",
                         help="Also save a side-by-side image+mask overlay for a quick visual check.")
    return parser.parse_args()



def load_and_preprocess_image(image_path, img_size):
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    resized = cv2.resize(original_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    model_input = np.expand_dims(normalized, axis=0)

    return model_input, original_bgr


def predict_mask(model, model_input, original_size):
    logits = model.predict(model_input, verbose=0)
    pred_mask = np.argmax(logits[0], axis=-1).astype(np.uint8)  # (img_size, img_size)

    orig_h, orig_w = original_size
    pred_mask_resized = cv2.resize(
        pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    return pred_mask_resized


def make_overlay(original_bgr, pred_mask, alpha=0.5):
    mask_normalized = (pred_mask.astype(np.float32) / max(pred_mask.max(), 1) * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, mask_colored, alpha, 0)
    return overlay



def main():
    args = parse_args()

    run_dir = os.path.join(config.MODEL_ROOT, args.run_name)
    checkpoint_path = os.path.join(run_dir, args.checkpoint)

    if args.output is None:
        base, _ = os.path.splitext(args.image)
        output_path = f"{base}_mask.png"
    else:
        output_path = args.output

    print(f"Loading checkpoint: {checkpoint_path}")

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

    print(f"Reading image: {args.image}")
    model_input, original_bgr = load_and_preprocess_image(args.image, config.IMG_SIZE)
    original_size = original_bgr.shape[:2]  # (h, w)

    pred_mask = predict_mask(model, model_input, original_size)

    cv2.imwrite(output_path, pred_mask)
    print(f"Saved predicted mask to: {output_path}")

    if args.save_overlay:
        overlay_path = f"{os.path.splitext(output_path)[0]}_overlay.png"
        overlay = make_overlay(original_bgr, pred_mask)
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved overlay visualization to: {overlay_path}")


if __name__ == "__main__":
    main()