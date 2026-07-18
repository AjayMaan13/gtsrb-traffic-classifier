# road-sign-vision

A traffic-sign image classifier (GTSRB, 43 categories), rebuilt from a CS50AI course exercise into a full, 12-factor-compliant deep learning project — configurable via environment variables, three compared architectures, full evaluation, Grad-CAM explainability, and a live demo app.

**Live demo:** [huggingface.co/spaces/AjayMaan13/road-sign-vision](https://huggingface.co/spaces/AjayMaan13/road-sign-vision) &nbsp;·&nbsp; 
**Stack:** Python · TensorFlow / Keras · scikit-learn · Gradio · Hugging Face Spaces

Upload a photo of a traffic sign → get back a predicted category, a confidence score, and a Grad-CAM heatmap showing which part of the image the model actually looked at when deciding.

<p align="center">
  <img src="reports/gradcam_correct.png" width="600" alt="Grad-CAM heatmaps over correctly-classified traffic signs — the model attends to the sign itself, not the background.">
</p>

## Results

Three architectures trained and compared on the same train/val/test split. The purpose-built deep CNN won outright — higher accuracy than transfer learning, with a fraction of the parameters.

| Model | Params | Val accuracy | Test accuracy | Notes |
|-------|--------|--------------|----------------|-------|
| Baseline CNN | 809,387 | 95.72% | 95.79% | control (original CS50 model) |
| **Deep CNN + augmentation** | **401,387** | **99.12%** | **99.17%** | **best result, fewest params** |
| MobileNetV2 (transfer) | 2,427,499 | 94.57% | 94.24% | frozen feature extractor, no fine-tuning |

<p align="center">
  <img src="reports/confusion_matrix.png" width="440" alt="43×43 confusion matrix on the held-out test set.">
</p>

Full evaluation artifacts (confusion matrix, per-class precision/recall/F1, misclassification gallery, and Grad-CAM overlays) are in [`reports/`](reports/).

## What's built

- **Data pipeline** (`src/road_sign_vision/data.py`) — `tf.data`-based loading with a proper train/val/test split, batching, shuffling, and per-class image counts logged (GTSRB is meaningfully imbalanced across its 43 categories).
- **Three compared architectures** (`src/road_sign_vision/model.py`):
  - a small baseline CNN (the original CS50 model, kept as the control),
  - a deeper CNN with batch normalization and traffic-sign-safe data augmentation (small rotation/zoom/brightness only — no flips, since those would change a sign's meaning),
  - transfer learning on MobileNetV2 pretrained on ImageNet.
- **Experiment tracking** (`src/road_sign_vision/train.py`) — `EarlyStopping` + `ModelCheckpoint`, results logged to `reports/experiment_results.csv` for comparison.
- **Full evaluation** (`src/road_sign_vision/evaluate.py`) — confusion matrix, per-class precision/recall/F1, and a gallery of the model's most confidently-wrong predictions.
- **Explainability** (`src/road_sign_vision/gradcam.py`) — Grad-CAM heatmaps showing which pixels drove each prediction, for both correct and misclassified images.
- **Serving app** (`app/app.py`) — a stateless Gradio app that loads the saved model once and serves predictions + Grad-CAM overlays; built to run identically locally and on Hugging Face Spaces.
- **12-factor discipline throughout** — all config from environment variables (`.env.example`), structured logging to stdout, training (build) and serving (run) kept strictly separate, pinned dependencies.

## How to run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh          # downloads GTSRB into data/gtsrb/

python scripts/train.py                # trains, saves models/model.keras
python scripts/evaluate.py             # confusion matrix + classification report
python scripts/gradcam.py              # Grad-CAM galleries
python app/app.py                      # runs the demo app locally
```

All config (epochs, image size, batch size, which architecture to train, augmentation on/off, etc.) is set via environment variables — see `.env.example`. For example, to compare architectures:

```bash
EXPERIMENT=deep AUGMENT=true python scripts/train.py
EXPERIMENT=transfer FINE_TUNE=false python scripts/train.py
```

## Project structure

```
src/road_sign_vision/
├── config.py      # all settings read from environment variables
├── data.py        # tf.data pipeline: split, batch, shuffle, prefetch
├── model.py       # baseline / deep / transfer architectures
├── train.py       # training loop, callbacks, experiment logging
├── evaluate.py    # confusion matrix, per-class metrics, error gallery
└── gradcam.py     # Grad-CAM explainability
scripts/           # one-off admin commands (download, train, evaluate, gradcam)
app/               # Gradio serving app (stateless inference)
reports/           # generated figures and metrics
```

## Project flow

```mermaid
flowchart TD
    A["GTSRB raw .ppm images<br/>(43 category folders)"] --> B["One-time .ppm → .png cache"]
    B --> C["tf.data pipeline<br/>batch / shuffle / prefetch"]
    C --> D["Train split (70%)"]
    C --> E["Validation split (15%)"]
    C --> F["Test split (15%) — touched once"]

    D --> G["3 architectures trained + compared"]
    E --> G
    G --> G1["Baseline CNN"]
    G --> G2["Deep CNN + augmentation"]
    G --> G3["Transfer: MobileNetV2"]

    G1 --> H["Best model saved<br/>(EarlyStopping + ModelCheckpoint)"]
    G2 --> H
    G3 --> H

    F --> I["evaluate.py"]
    H --> I
    I --> I1["Confusion matrix"]
    I --> I2["Per-class precision/recall/F1"]
    I --> I3["Misclassification gallery"]

    H --> J["gradcam.py"]
    J --> J1["Grad-CAM heatmaps<br/>(correct + misclassified)"]

    H --> K["app/app.py — Gradio demo"]
    K --> L["Hugging Face Spaces<br/>(live, public demo)"]
```

## What this builds on

The original model architecture started as a CS50AI ("Introduction to Artificial Intelligence with Python") course exercise — a minimal CNN with no validation split, no experiment tracking, no evaluation beyond one accuracy number, and no way to inspect *why* it worked. Everything else in this repo — the data pipeline, architecture comparison, evaluation, explainability, and serving app — is original work built on top of that starting point.

## What I learned / limitations

- A three-way train/val/test split and watching validation loss during training catches overfitting that a simple train/test split misses.
- Class imbalance across GTSRB's 43 categories means accuracy alone is misleading — per-class precision/recall is what actually surfaces which categories the model struggles with.
- Grad-CAM is a cheap, effective sanity check: it confirmed the model attends to the actual sign rather than background clutter, for both correct and incorrect predictions.
- Transfer learning didn't automatically win here: a frozen, non-fine-tuned MobileNetV2 (2.4M params) underperformed a purpose-built deep CNN with only 401K params. ImageNet's natural-photo features don't transfer perfectly to small, low-resolution, symbol-like traffic-sign images without fine-tuning — a useful reminder not to assume transfer learning is always the answer.
- Limitation: the transfer-learning result reported here is feature-extraction only (frozen base); fine-tuning the base layers (`FINE_TUNE=true`) is implemented but not yet run to compare.

## Dataset & credits

- **Dataset:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html), J. Stallkamp et al. Used for research/educational purposes.
- **Starting point:** [CS50's Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/) (the `traffic` project).

## License

[MIT](LICENSE)
