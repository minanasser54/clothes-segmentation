# Clothes Segmentation

Semantic segmentation model that separates worn clothing from people in images — developed for Computer Vision Assignment **CVCY002**.

Given a photo of a person, the pipeline predicts a per-pixel class map distinguishing clothing regions (upper body, lower body, dress, shoes, accessories, suit) from skin, hair, and background.

## Project layout

```
clothes-segmentation/
├── pyproject.toml           # uv-managed dependencies
├── uv.lock
├── README.md                # this file
├── REPORT.md                # architecture, loss, evaluation, limitations
├── src/
│   ├── scripts/
│   │   ├── configs.py            # all fixed project settings (paths, image size, etc.)
│   │   ├── class_mapping.py      # 59-class → 9-superclass label mapping
│   │   ├── data_download.py      # pulls the raw dataset from Kaggle
│   │   ├── data_preprocessing.py # remaps masks, splits, resizes → Data/1_processed
│   │   ├── data_augmentation.py  # augments train split → Data/2_augmented
│   │   ├── model.py               # U-Net and U-Net++ architectures, loss, metric
│   │   ├── train.py               # training loop (CLI args for run-specific values)
│   │   ├── eval.py                # test-set evaluation (mIoU, Dice, pixel accuracy)
│   │   └── inference.py           # single-image prediction
│   ├── Data/                     # generated locally, not committed (see .gitignore)
│   └── models/                   # generated locally, not committed (see .gitignore)
```

## Local setup (Windows)

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
The commands below assume the repository is at `D:\CsCv\clothes-segmentation`.

```powershell
# Install uv if it is not already available
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install the project dependencies into the local .venv
cd D:\CsCv\clothes-segmentation
uv sync
```

Run the pipeline from `src`:

```powershell
cd D:\CsCv\clothes-segmentation\src
uv run scripts/data_download.py
uv run scripts/data_preprocessing.py
uv run scripts/data_augmentation.py
uv run scripts/train.py --run-name unet_pp_v1 --epochs 60 --batch-size 32 --lr 1e-3
uv run scripts/eval.py --run-name unet_pp_v1
uv run scripts/inference.py --run-name unet_pp_v1 --image Data\2_augmented\test\images\img_0018.png --save-overlay
```

Generated data is stored in `src\Data` and model outputs in `src\models`.

## Colab setup

Mount Google Drive, place or clone the repository there, and run `uv sync` from
the project root. Replace the project path below if your Drive folder differs.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install uv
%cd /content/drive/MyDrive/clothes-segmentation
!uv sync
%cd /content/drive/MyDrive/clothes-segmentation/src
```

Run the pipeline from `src`:

```bash
!uv run scripts/data_download.py
!uv run scripts/data_preprocessing.py
!uv run scripts/data_augmentation.py
!uv run scripts/train.py --run-name unet_pp_v1 --epochs 60 --batch-size 32 --lr 1e-3
!uv run scripts/eval.py --run-name unet_pp_v1
!uv run scripts/inference.py --run-name unet_pp_v1 --image Data/2_augmented/test/images/img_0018.png --save-overlay
```

### Kaggle access

`data_download.py` uses `kagglehub`, which needs Kaggle API credentials. Either:
- place your `kaggle.json` (from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token) at `~/.kaggle/kaggle.json`, or
- set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.

## Reproducing results

Run the pipeline stages in order using either the local or Colab commands
above. Each stage reads the previous stage's output from disk; there is no
in-memory hand-off between scripts.

### `train.py` CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 50 | Max training epochs (early stopping may end training sooner) |
| `--batch-size` | 16 | Training/validation batch size |
| `--lr` | 1e-3 | Initial learning rate (reduced automatically on plateau) |
| `--run-name` | `unet_run` | Output subfolder name under `MODEL_ROOT` |
| `--data-dir` | `2_augmented` | Which `Data/` subfolder to train from |
| `--architecture` | `unet_plus_plus` | `unet` or `unet_plus_plus` |
| `--resume-from` | — | Path to a checkpoint to resume training from |
| `--cache-to-disk` | off | Cache decoded images to local disk instead of RAM (use if you hit an OOM error) |
| `--early-stopping-patience` | 10 | Epochs without improvement before stopping |
| `--log-every-n-steps` | 10 | In-epoch progress logging frequency |

Every training run writes its checkpoint, logs, and configuration under
`src/models/<run-name>/` — nothing is overwritten between runs as long as
`--run-name` differs.

### `eval.py` output

Writes `models/<run-name>/eval/metrics.json` (Overall mIoU, Foreground mIoU, mean Dice, pixel accuracy, and a per-class IoU/Dice breakdown) and `sample_predictions.png` (a visual grid of image / ground truth / prediction).

### `inference.py` output

Given `--image path/to/photo.jpg`, saves `path/to/photo_mask.png` (the predicted per-pixel class map, resized back to the original image's resolution) and, with `--save-overlay`, a colorized overlay for a quick visual check.

## Results summary

The final model (`unet_plus_plus`, 9-superclass labels, 256×256 input) reached:

- **val_mean_iou (best checkpoint, epoch 28/38): 0.560**
- **val_accuracy (best checkpoint): 0.910**

Full metrics (including per-class breakdown) are in `REPORT.md`.

## See also

`REPORT.md` covers dataset choice, architecture rationale, loss function design (including a bug found and fixed in the Dice loss implementation), full evaluation results, and known system limitations.
