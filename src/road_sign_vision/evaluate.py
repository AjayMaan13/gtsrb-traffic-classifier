import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from road_sign_vision.config import DATA_DIR, MODEL_OUT, REPORTS_DIR
from road_sign_vision.data import get_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_predictions(model, test_ds):
    """
    Run the test set through the model once and collect, per image: the true
    label, the predicted label, the model's confidence, and the raw image
    (for the misclassification gallery).
    """
    y_true, y_pred, confidences, images_out = [], [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        batch_pred = np.argmax(probs, axis=1)
        batch_true = np.argmax(labels.numpy(), axis=1)
        batch_confidence = np.max(probs, axis=1)

        y_true.extend(batch_true.tolist())
        y_pred.extend(batch_pred.tolist())
        confidences.extend(batch_confidence.tolist())
        images_out.extend(images.numpy())

    return np.array(y_true), np.array(y_pred), np.array(confidences), images_out


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    # labels=range(...) forces a full NxN matrix even if some class is absent
    # from this particular test split (relevant with class imbalance).
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.xlabel("Predicted category")
    plt.ylabel("True category")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Confusion matrix saved to %s", out_path)


def save_classification_report(y_true, y_pred, class_names, out_path):
    report = classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),  # force all classes to appear, even if absent from this test split
        target_names=class_names,
        zero_division=0,
    )
    with open(out_path, "w") as f:
        f.write(report)
    logger.info("Classification report saved to %s", out_path)
    logger.info("\n%s", report)


def save_misclassified_gallery(y_true, y_pred, confidences, images, class_names, out_path, top_n=9):
    """
    Save the model's most *confident* mistakes — cases where it was sure of a
    wrong answer, which are usually the most informative failures to look at.
    """
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        logger.info("No misclassified images — skipping gallery.")
        return

    # Sort wrong predictions by confidence, descending, and take the worst.
    wrong_sorted = wrong[np.argsort(-confidences[wrong])]
    top = wrong_sorted[:top_n]

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(top):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx].astype("uint8"))
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        plt.title(f"true: {true_name}\npred: {pred_name} ({confidences[idx]:.2f})", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Misclassification gallery saved to %s", out_path)


def main():
    logger.info("Loading model from %s", MODEL_OUT)
    model = tf.keras.models.load_model(MODEL_OUT)

    logger.info("Building test dataset from %s", DATA_DIR)
    _, _, test_ds, class_names = get_datasets(DATA_DIR)

    logger.info("Running the test set once through the model")
    y_true, y_pred, confidences, images = collect_predictions(model, test_ds)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    save_classification_report(
        y_true, y_pred, class_names, os.path.join(REPORTS_DIR, "classification_report.txt")
    )
    save_misclassified_gallery(
        y_true, y_pred, confidences, images, class_names,
        os.path.join(REPORTS_DIR, "misclassified_gallery.png"),
    )


if __name__ == "__main__":
    main()
