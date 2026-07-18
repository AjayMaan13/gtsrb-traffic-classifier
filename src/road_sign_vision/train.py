import csv
import json
import logging
import os
import time

import tensorflow as tf

from road_sign_vision.config import (
    AUGMENT,
    CLASS_NAMES_PATH,
    DATA_DIR,
    EPOCHS,
    EXPERIMENT,
    FINE_TUNE,
    MODEL_OUT,
    PATIENCE,
    RESULTS_CSV,
)
from road_sign_vision.data import get_datasets, save_sample_grid
from road_sign_vision.model import get_baseline_model, get_deep_model, get_transfer_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_model():
    if EXPERIMENT == "baseline":
        return get_baseline_model()
    if EXPERIMENT == "deep":
        return get_deep_model(augment=AUGMENT)
    if EXPERIMENT == "transfer":
        return get_transfer_model(fine_tune=FINE_TUNE)
    raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT!r} (expected baseline|deep|transfer)")


def build_callbacks():
    return [
        # Stop once validation loss stops improving, and roll back to the
        # best epoch's weights instead of whatever the last epoch produced.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        # Persist the best-seen model to disk as training progresses, so the
        # saved artifact is the best epoch, not necessarily the last one.
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_loss", save_best_only=True
        ),
    ]


def log_results(model_name, params, val_accuracy, test_accuracy, train_seconds, notes):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    is_new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["model", "params", "val_accuracy", "test_accuracy", "train_seconds", "notes"]
            )
        writer.writerow([model_name, params, f"{val_accuracy:.4f}", f"{test_accuracy:.4f}",
                          f"{train_seconds:.1f}", notes])
    logger.info("Result appended to %s", RESULTS_CSV)


def main():
    logger.info("Building datasets from %s", DATA_DIR)
    train_ds, val_ds, test_ds, class_names = get_datasets(DATA_DIR)

    # The serving app (Phase 7) needs to map a prediction index back to a
    # category name without shipping the whole dataset, so persist the
    # mapping once here, next to the model artifact.
    os.makedirs(os.path.dirname(CLASS_NAMES_PATH), exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    logger.info("Class names saved to %s", CLASS_NAMES_PATH)

    save_sample_grid(train_ds, class_names)

    logger.info("Experiment: %s (augment=%s, fine_tune=%s)", EXPERIMENT, AUGMENT, FINE_TUNE)
    model = build_model()
    params = model.count_params()

    logger.info("Training for up to %d epochs (early stopping patience=%d)", EPOCHS, PATIENCE)
    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=build_callbacks()
    )
    train_seconds = time.time() - start

    val_accuracy = max(history.history["val_accuracy"])
    loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    logger.info("Test loss: %.4f, test accuracy: %.4f", loss, test_accuracy)

    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    notes = f"augment={AUGMENT}" if EXPERIMENT == "deep" else (
        f"fine_tune={FINE_TUNE}" if EXPERIMENT == "transfer" else "control"
    )
    log_results(EXPERIMENT, params, val_accuracy, test_accuracy, train_seconds, notes)


if __name__ == "__main__":
    main()
