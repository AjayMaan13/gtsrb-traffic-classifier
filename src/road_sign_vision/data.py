import logging
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from road_sign_vision.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_CATEGORIES,
    REPORTS_DIR,
    SEED,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)


def _log_class_counts(data_dir):
    """Log how many images are in each category folder, surfacing class imbalance."""
    for category in range(NUM_CATEGORIES):
        folder = os.path.join(data_dir, str(category))
        count = len(os.listdir(folder))
        logger.info("Category %s: %d images", category, count)


def _ensure_png_cache(data_dir):
    """
    GTSRB ships images as .ppm, a format image_dataset_from_directory can't
    read (it only supports bmp/gif/jpeg/png). Convert once into a sibling
    "<data_dir>_png" cache, then reuse it on later runs instead of
    re-converting every time.
    """
    cache_dir = data_dir.rstrip("/") + "_png"
    if os.path.isdir(cache_dir):
        return cache_dir

    logger.info("Converting .ppm images to .png in %s (one-time cache build)", cache_dir)
    for category in range(NUM_CATEGORIES):
        src_folder = os.path.join(data_dir, str(category))
        dst_folder = os.path.join(cache_dir, str(category))
        os.makedirs(dst_folder, exist_ok=True)
        for filename in os.listdir(src_folder):
            img = cv2.imread(os.path.join(src_folder, filename))
            out_name = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(dst_folder, out_name), img)

    return cache_dir


def get_datasets(data_dir=DATA_DIR):
    """
    Build train/validation/test tf.data.Datasets from `data_dir` (one
    subfolder per category, e.g. gtsrb/0, gtsrb/1, ...).

    Returns (train_ds, val_ds, test_ds, class_names) — roughly a 70/15/15 split.
    """
    _log_class_counts(data_dir)
    data_dir = _ensure_png_cache(data_dir)

    # BATCHING: image_dataset_from_directory groups images into batches of
    # BATCH_SIZE automatically, instead of loading the whole dataset as one
    # giant list like the old load_data() did.
    #
    # SHUFFLING: shuffle=True randomizes image order for the training set, so
    # each batch is a mix of categories rather than long runs of one category
    # (which would make the model repeatedly overfit to "whatever category is
    # currently in front of it").
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        shuffle=True,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",  # one-hot labels, matching model.py's categorical_crossentropy
    )
    class_names = train_ds.class_names

    # The remaining VAL_SPLIT fraction, not yet divided into val/test.
    # shuffle=False: validation/test order stays fixed and reproducible run
    # to run, unlike training data.
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    # Split that combined chunk batch-for-batch: first half -> validation
    # (watched during training), second half -> test (touched once, at the end).
    val_batches = val_test_ds.cardinality()
    val_ds = val_test_ds.take(val_batches // 2)
    test_ds = val_test_ds.skip(val_batches // 2)

    # NORMALIZATION: NOT done here anymore as of Phase 4. Different model
    # architectures need different pixel scaling (plain /255 for the baseline
    # and deep CNNs, MobileNetV2's own preprocess_input for transfer learning),
    # so each model in model.py now owns its own preprocessing as its first
    # layer(s). These datasets stay in raw [0, 255] so any model can use them.

    # PREFETCHING: let TensorFlow load/decode the next batch on the CPU while
    # the current batch is still training, instead of the model sitting idle
    # waiting on disk I/O between batches. AUTOTUNE picks the buffer size.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    logger.info(
        "Dataset ready: %d train batches, %d val batches, %d test batches (batch size %d)",
        train_ds.cardinality().numpy(),
        val_ds.cardinality().numpy(),
        test_ds.cardinality().numpy(),
        BATCH_SIZE,
    )

    return train_ds, val_ds, test_ds, class_names


def save_sample_grid(dataset, class_names, out_path=None):
    """Save a grid of one batch's images + labels, to sanity-check the pipeline."""
    out_path = out_path or os.path.join(REPORTS_DIR, "sample_grid.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    images, labels = next(iter(dataset.take(1)))
    plt.figure(figsize=(10, 10))
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        # Images are raw [0, 255] now (see get_datasets) — scale for display.
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = tf.argmax(labels[i]).numpy()
        plt.title(class_names[label_index])
        plt.axis("off")
    plt.savefig(out_path)
    plt.close()
    logger.info("Sample grid saved to %s", out_path)
