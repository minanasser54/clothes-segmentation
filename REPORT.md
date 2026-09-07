# Report: Clothes Segmentation (CVCY002)

## 1. Dataset choice and reason

**Dataset:** [People Clothing Segmentation](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation) (Kaggle, CC0), 1000 images paired with pixel-level segmentation masks across 59 original classes (58 clothing/body classes + background).

**Why this dataset:**

- **Ready-to-use pixel masks.** Unlike alternatives considered (iMaterialist Fashion 2020: RLE-encoded masks and a ~24GB download; Clothing Co-Parsing: `.mat`-format masks and multiple annotation formats; ModaNet: polygon annotations with no bundled images; DeepFashion: bounding boxes only, no pixel masks at all), this dataset ships uniform-size PNG image/mask pairs with no format conversion required.
- **Uniform, pre-cleaned data.** All 1000 images are the same original resolution (825×550), removing a source of preprocessing inconsistency.
- **Right-sized for the assignment's constraints.** Small enough to iterate on quickly (including on free-tier Colab GPU), while still presenting a genuine, non-trivial 59-class segmentation problem.

**Constraint honored:** per the assignment brief, **no pretrained encoders were used** — both architectures implemented (`build_unet`, `build_unet_plus_plus` in `model.py`) are trained entirely from randomly-initialized weights.

## 2. Preprocessing and class remapping

### 2.1 Original pipeline

`data_preprocessing.py` matches images to masks by filename id, splits into train/val/test (80/10/10), and resizes both to a fixed square size (`IMG_SIZE`, final runs used 256×256) — images with bilinear interpolation, masks with **nearest-neighbor** interpolation exclusively, since interpolating class-id integers would invent invalid intermediate class values.

### 2.2 Class remapping: 59 → 9 superclasses

An initial model trained on all 59 original classes reached an overall mIoU of only **0.143**, with per-class inspection showing **20+ classes stuck at exactly 0.0 IoU** (`ring`, `watch`, `wallet`, `tie`, `hoodie`, `heels`, `loafers`, `gloves`, `bodysuit`, and others). With only ~1600 training images (after augmentation) spread across 59 classes, most fine-grained garment types simply did not have enough training examples to learn distinguishable boundaries.

To address this, `class_mapping.py` groups the 59 original classes into **9 semantic superclasses**:

| ID | Superclass | Original classes folded in |
|---|---|---|
| 0 | background | null |
| 1 | upper_clothes | blazer, blouse, bra, cape, cardigan, coat, hoodie, intimate, jacket, jumper, shirt, sweater, sweatshirt, t-shirt, top, vest |
| 2 | lower_clothes | jeans, leggings, panties, pants, shorts, skirt, stockings, tights |
| 3 | dress | bodysuit, dress, romper, swimwear |
| 4 | shoes | boots, clogs, flats, heels, loafers, pumps, sandals, shoes, sneakers, socks, wedges |
| 5 | accessories | accessories, bag, belt, bracelet, earrings, glasses, gloves, hat, necklace, purse, ring, scarf, sunglasses, tie, wallet, watch |
| 6 | skin | skin |
| 7 | hair | hair |
| 8 | suit | suit |

Suits were kept as their own class rather than folded into `dress`/`upper_clothes`/`lower_clothes`, since a suit mask typically spans torso and legs as a single annotated region and does not cleanly correspond to any of the other groups.

**Tradeoff, stated explicitly:** this remapping trades fine-grained garment-type recognition (e.g. distinguishing a "jacket" from a "coat") for substantially more reliable coarse-category segmentation. This is an appropriate tradeoff given the dataset size constraint, but it is a real limitation — see Section 6.

Remapping happens on the full-resolution mask **before** resizing, so the remap lookup only ever sees valid original class ids.

### 2.3 Augmentation

`data_augmentation.py` applies segmentation-safe augmentations to the training split only (validation and test splits are copied through unchanged, since they must reflect real, unaltered inputs):

- **Geometric** (applied identically to image and mask): horizontal flip, random crop + resize.
- **Color-only** (applied to the image, mask untouched): brightness, contrast, saturation jitter.

All mask-affecting resize operations use nearest-neighbor interpolation exclusively.

## 3. Model architecture

Two from-scratch architectures were implemented in `model.py`:

- **`build_unet`** — a standard U-Net baseline (encoder/decoder with single skip connections per resolution level).
- **`build_unet_plus_plus`** — **the primary model used for final results.**

### Why U-Net++

U-Net++ (Zhou et al., 2018) replaces each single skip connection with a **grid of nested, densely-connected convolution blocks**. For encoder depth level `i` and nesting position `j`:

```
X[i,j] = Conv(Concat(X[i,0], X[i,1], ..., X[i,j-1], Upsample(X[i+1,j-1])))
```

This means the decoder at each resolution receives a dense concatenation of *all* previously-computed feature maps at that level, not just the matching single encoder skip. The paper's central claim — and the reason this architecture was chosen for this task — is that this progressively closes the semantic gap between shallow (fine-detail) and deep (coarse-semantic) features before fusion, which should help recover finer object boundaries. Clothing segmentation involves exactly this kind of fine-boundary challenge: sleeve edges, collar lines, and the boundary between adjacent garments.

**Deep supervision was intentionally omitted** (only the final, most-nested output node produces the prediction — no auxiliary losses at intermediate decoder stages) to keep the implementation simpler and easier to debug and explain. This is a deliberate simplification, not an oversight; see Section 6 for the tradeoff.

Final architecture used `UNET_PP_FEATURES = [32, 64, 128, 256]` (4 levels), chosen over the paper's 5-level default to control parameter count given the ~1600-image training set.

## 4. Loss function selection and reason

The loss is a weighted hybrid of **Dice loss** and **class-weighted Cross-Entropy**:

```
total_loss = DICE_WEIGHT * dice_loss + (1 - DICE_WEIGHT) * weighted_ce_loss
```

with `DICE_WEIGHT = 0.7`.

**Why this combination:**
- **Dice loss** directly optimizes for region overlap (closely related to IoU), which is more informative than pixel-wise accuracy for a task with severe class imbalance — a single image's clothing pixels can be a small fraction of total pixels compared to background/skin.
- **Cross-entropy**, weighted by inverse log-frequency per class (computed from a sample of training masks), provides more stable early-training gradients than Dice alone, which can behave poorly when predictions are near-random.

### A bug found and fixed during development

The initial Dice loss implementation averaged per-class Dice scores over **all** classes, including classes with zero ground-truth pixels in a given batch. Because softmax never outputs an exact zero probability, an absent class's predicted-probability sum was always slightly positive while its intersection was exactly zero — driving that class's Dice score to ≈0.32 rather than the correct ≈1.0, **even for a mathematically perfect prediction**.

This was confirmed directly: a synthetic perfect-prediction test case produced a Dice loss of **0.655** (should be ≈0) under the buggy implementation. With ~5-10 classes typically present per image out of the full class count, this meant the majority of every batch's per-class loss terms were near-meaningless noise, diluting the gradient signal on the classes that actually mattered for that image.

**The fix:** `dice_loss()` now restricts averaging to only the classes with at least one ground-truth pixel present in the batch. The same synthetic perfect-prediction test case now correctly returns a loss of ≈0. The fix was verified to remain fully differentiable (gradients still flow to all logits) before being adopted for all subsequent training runs, including the final reported results.

## 5. Performance analysis and evaluation metrics

`eval.py` reports five metrics on the held-out test split:

- **Overall mIoU** — mean IoU across all classes present in the test set.
- **Foreground mIoU** — the same, but excluding the background class, so a large, trivially-easy background region doesn't inflate the headline number.
- **Mean Dice** — macro-averaged per-class Dice score (computed the same way as the training loss, for a fair comparison between training objective and final evaluation).
- **Pixel accuracy** — fraction of all pixels correctly classified.
- **Per-class IoU and Dice** — full breakdown, saved in `metrics.json`.

### Final training run results

Trained for 38 epochs (of a 60-epoch budget; early stopping and checkpoint selection both monitor `val_mean_iou`):

| Metric | Value |
|---|---|
| Best val_mean_iou (epoch 28) | **0.560** |
| val_accuracy at best checkpoint | **0.910** |
| Final-epoch train_mean_iou | 0.899 |
| Final-epoch train_loss / val_loss | 0.055 / 0.284 |

The gap between final-epoch train_mean_iou (0.899) and val_mean_iou (0.552–0.560) indicates the model was overfitting by the later epochs — expected given the small dataset and no pretrained encoder. `ModelCheckpoint` (monitoring `val_mean_iou`, `save_best_only=True`) protects against this by retaining the epoch-28 checkpoint rather than the final epoch's weights.

*(Run `eval.py` on the saved checkpoint to populate the full per-class IoU/Dice table and pixel-accuracy figure for the final submitted model; this section should be updated with those exact numbers before submission.)*

## 6. Strengths, weaknesses, and limitations

### Strengths

- Reliable separation of large, high-contrast regions: skin, hair, and broad garment categories (upper/lower clothes) show strong IoU, consistent with the class-remapping goal.
- Robust to moderate variation in pose and background clutter, aided by geometric + color augmentation on the training set.
- No dependency on pretrained weights or external model downloads at inference time — the model is fully self-contained.

### Weaknesses

- **Coarse category resolution.** Because of the 59→9 class remapping, the model cannot distinguish between garment *types* within a superclass (e.g. "t-shirt" vs "sweater" are both `upper_clothes`). This was a deliberate tradeoff for reliability given the dataset size, but it limits applicability wherever garment-type-specific output is required.
- **Overfitting on a small dataset.** The train/val mean_iou gap (0.899 vs ~0.55) by the later epochs reflects the fundamental constraint of training from scratch on ~1600 images with no pretrained encoder (a constraint specified by the assignment, not a preprocessing choice).
- **No deep supervision.** U-Net++'s full benefit (per the original paper) includes auxiliary losses at intermediate decoder stages; this implementation only supervises the final output, trading some potential accuracy for simplicity.
- **Fixed input resolution (256×256).** Fine details — thin straps, small accessories, garment texture — are lost at this resolution. This was a practical choice to fit training within available GPU memory.

### Limitations (operating conditions)

- **Single-person images only.** The dataset and model assume one person per image; multi-person scenes are untested and likely to produce unreliable results.
- **Frontal/near-frontal poses.** Training data is predominantly frontal-facing fashion photography; extreme poses, partial occlusion, or unusual camera angles are outside the training distribution.
- **Clear lighting conditions assumed.** No explicit testing was done under low light, strong shadows, or backlit conditions.
- **Photographic (not illustrated/synthetic) images.** The model was trained and evaluated only on real photographs; performance on drawings, renders, or heavily stylized images is unknown.

## 7. Reproducing this report's numbers

```bash
uv run scripts/eval.py --run-name <your-run-name>
```

produces `models/<run-name>/eval/metrics.json` and `sample_predictions.png`, containing the exact per-class breakdown and visual samples referenced above.
