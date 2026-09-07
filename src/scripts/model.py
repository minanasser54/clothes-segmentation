"""
model.py
- build_unet: plain U-Net (original baseline)
- build_unet_plus_plus: U-Net++ (nested, dense skip pathways)
- defines the loss function (Dice + weighted Cross-Entropy hybrid) 
- mean IoU metric, shared by both architectures.

Usage:
    from model import build_unet_plus_plus, dice_ce_loss, MeanIoUMetric

    model = build_unet_plus_plus(num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE)
    model.compile(optimizer=..., loss=dice_ce_loss(num_classes=config.NUM_CLASSES),
                  metrics=[MeanIoUMetric(num_classes=config.NUM_CLASSES)])
"""

import tensorflow as tf
from tensorflow.keras import layers, Model



def double_conv_block(x, filters, name=None):
    """conv bn relu"""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       name=f"{name}_conv1" if name else None)(x)
    x = layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu1" if name else None)(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       name=f"{name}_conv2" if name else None)(x)
    x = layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu2" if name else None)(x)
    return x


def _match_spatial_size(x, target):
    """spatial size Resizing guard"""
    if x.shape[1] != target.shape[1] or x.shape[2] != target.shape[2]:
        x = layers.Resizing(target.shape[1], target.shape[2], interpolation="bilinear")(x)
    return x



def downsample_block(x, filters):
    skip = double_conv_block(x, filters)
    pooled = layers.MaxPool2D(pool_size=2)(skip)
    return skip, pooled


def upsample_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = _match_spatial_size(x, skip)
    x = layers.Concatenate()([x, skip])
    x = double_conv_block(x, filters)
    return x


def build_unet(num_classes, img_size, features=(64, 128, 256, 512), in_channels=3):
    """Plain U-Net baseline. Initial not used unet version"""
    inputs = layers.Input(shape=(img_size, img_size, in_channels))
    x = inputs
    skips = []
    for f in features:
        skip, x = downsample_block(x, f)
        skips.append(skip)
    x = double_conv_block(x, features[-1] * 2)
    for f, skip in zip(reversed(features), reversed(skips)):
        x = upsample_block(x, skip, f)
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same")(x)
    return Model(inputs=inputs, outputs=outputs, name="unet")


def build_unet_plus_plus(num_classes, img_size, features=(32, 64, 128, 256, 512), in_channels=3):
    """Build U-Net++ with a single output head wo deep supervision """
    depth = len(features)
    inputs = layers.Input(shape=(img_size, img_size, in_channels))
    nodes = {}

    # --- Encoder column (j = 0): plain downsampling path ---
    x = inputs
    for i in range(depth):
        node = double_conv_block(x, features[i], name=f"X{i}_0")
        nodes[(i, 0)] = node
        if i < depth - 1:
            x = layers.MaxPool2D(pool_size=2)(node)

    # --- Nested skip columns
    for j in range(1, depth):
        for i in range(0, depth - j):
            deeper_node = nodes[(i + 1, j - 1)]
            upsampled = layers.Conv2DTranspose(
                features[i], kernel_size=2, strides=2, padding="same",
                name=f"up_X{i}_{j}",
            )(deeper_node)

            same_level_nodes = [nodes[(i, k)] for k in range(j)]
            upsampled = _match_spatial_size(upsampled, same_level_nodes[0])

            concat = layers.Concatenate(name=f"concat_X{i}_{j}")(same_level_nodes + [upsampled])
            node = double_conv_block(concat, features[i], name=f"X{i}_{j}")
            nodes[(i, j)] = node

    # Final output comes from the shallowest, most-nested node: X[0, depth-1]
    final_node = nodes[(0, depth - 1)]
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same")(final_node)

    return Model(inputs=inputs, outputs=outputs, name="unet_plus_plus")


def dice_loss(y_true, y_pred, num_classes, smooth=1e-6):
    """Multi-class soft Dice loss, averaged over classes PRESENT IN THE BATCH.
    Args:
        y_true: int labels, shape (batch, H, W, 1) or (batch, H, W).
        y_pred: raw logits, shape (batch, H, W, num_classes).
    """

    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == 4 else y_true
    y_true_oh = tf.one_hot(y_true, depth=num_classes)

    y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

    axes = [1, 2]
    intersection = tf.reduce_sum(y_true_oh * y_pred_soft, axis=axes)
    ground_truth_sum = tf.reduce_sum(y_true_oh, axis=axes)
    union = ground_truth_sum + tf.reduce_sum(y_pred_soft, axis=axes)

    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    # Only include classes with at least one ground-truth pixel in this batch.
    present_mask = tf.cast(ground_truth_sum > 0, tf.float32)
    num_present = tf.reduce_sum(present_mask)
    num_present = tf.maximum(num_present, 1.0)

    dice = tf.reduce_sum(dice_per_class * present_mask) / num_present

    return 1.0 - dice


def weighted_ce_loss(y_true, y_pred, num_classes, class_weights=None):
    """Sparse categorical cross-entropy, optionally weighted per class.
    Args:
        y_true: int labels, shape (batch, H, W, 1) or (batch, H, W).
        y_pred: raw logits, shape (batch, H, W, num_classes).
        class_weights: optional 1D tensor/list of length num_classes.
    """
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == 4 else y_true

    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    if class_weights is not None:
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        sample_weights = tf.gather(weights_tensor, y_true)
        ce = ce * sample_weights

    return tf.reduce_mean(ce)


def dice_ce_loss(num_classes, dice_weight=0.7, class_weights=None):
    """total = dice_weight * dice_loss + (1 - dice_weight) * ce_loss"""
    def loss_fn(y_true, y_pred):
        d = dice_loss(y_true, y_pred, num_classes)
        ce = weighted_ce_loss(y_true, y_pred, num_classes, class_weights)
        return dice_weight * d + (1.0 - dice_weight) * ce

    return loss_fn



class MeanIoUMetric(tf.keras.metrics.Metric):
    """Wraps tf.keras.metrics.MeanIoU to accept raw logits + int labels """

    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == 4 else y_true
        y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.iou_metric.update_state(y_true, y_pred_labels, sample_weight)

    def result(self):
        return self.iou_metric.result()

    def reset_state(self):
        self.iou_metric.reset_state()


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs import config

    print("=" * 60)
    print("Building plain U-Net (baseline)...")
    unet = build_unet(
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        features=tuple(config.UNET_FEATURES),
    )
    unet.summary()
    print(f"Plain U-Net param count: {unet.count_params():,}")

    print("=" * 60)
    print("Building U-Net++ (primary model)...")
    unet_pp = build_unet_plus_plus(
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        features=tuple(getattr(config, "UNET_PP_FEATURES", (32, 64, 128, 256, 512))),
    )
    unet_pp.summary()
    print(f"U-Net++ param count: {unet_pp.count_params():,}")

    dummy_input = tf.random.uniform([2, config.IMG_SIZE, config.IMG_SIZE, 3])
    output = unet_pp(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"U-Net++ output shape: {output.shape}")
    assert output.shape == (2, config.IMG_SIZE, config.IMG_SIZE, config.NUM_CLASSES)
    print("U-Net++ builds and runs a forward pass correctly.")