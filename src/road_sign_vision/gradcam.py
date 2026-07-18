import logging
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from road_sign_vision.config import DATA_DIR, MODEL_OUT, REPORTS_DIR
from road_sign_vision.data import get_datasets
from road_sign_vision.evaluate import collect_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_last_spatial_layer(model):
    """
    Grad-CAM needs the last layer that still outputs a spatial feature map
    (shape (batch, height, width, channels), rank 4) — before Flatten/
    GlobalAveragePooling collapses it. Walking backwards works for all three
    Phase 4 architectures: for baseline/deep it lands on the last Conv2D-ish
    layer, and for transfer learning it lands on the nested MobileNetV2 layer
    itself, since Keras treats an applied sub-model as one layer whose output
    is MobileNetV2's own last feature map.
    """
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer
    raise ValueError("No spatial (4D) layer found — is this a CNN?")


def make_gradcam_heatmap(image, model, layer=None):
    """
    Returns (heatmap, predicted_class_index) for a single image.

    The heatmap is built from the gradient of the top predicted class's score
    with respect to the last spatial layer's feature maps: channels whose
    activations push the prediction up get a positive weight, and averaging
    the gradient over height/width turns each channel's importance into a
    single number.
    """
    layer = layer or _find_last_spatial_layer(model)
    img_batch = image[np.newaxis, ...].astype("float32")

    if isinstance(layer, tf.keras.Model):
        # The spatial layer is itself a nested sub-model (MobileNetV2, in the
        # transfer-learning architecture). Keras can't trace gradients through
        # a nested model's output in one combined tf.keras.Model the way it
        # can for a plain layer, so instead we manually replay the outer
        # model's layers in two halves — everything before the nested model,
        # then the nested model itself, then everything after — under a
        # single GradientTape watching the split point.
        outer_layers = model.layers
        split_index = outer_layers.index(layer)
        pre_layers = outer_layers[1:split_index]  # skip the InputLayer itself
        post_layers = outer_layers[split_index + 1:]

        x = img_batch
        for pre_layer in pre_layers:
            x = pre_layer(x)

        with tf.GradientTape() as tape:
            tape.watch(x)
            conv_output = layer(x)
            predictions = conv_output
            for post_layer in post_layers:
                predictions = post_layer(predictions)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, conv_output)
    else:
        grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_batch)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        # How much would the top class's score change if each pixel of the
        # feature maps changed? That's the gradient we pool into per-channel
        # importance weights.
        grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    # Weight each feature-map channel by how important it was, then sum
    # across channels into a single 2D heatmap.
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: only positive influence counts as "supporting" the prediction.
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else heatmap

    return heatmap.numpy(), int(pred_index.numpy())


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Resize the (small) heatmap up to image size and blend it over the original."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colormap = plt.get_cmap("jet")
    heatmap_colored = np.uint8(255 * colormap(heatmap_uint8)[:, :, :3])

    overlay = image.astype("float32") * (1 - alpha) + heatmap_colored.astype("float32") * alpha
    return overlay.astype("uint8")


def save_gradcam_gallery(model, images, indices, true_labels, pred_labels, class_names, out_path, top_n=9):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    selected = indices[:top_n]

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(selected):
        heatmap, _ = make_gradcam_heatmap(images[idx], model)
        overlay = overlay_heatmap(images[idx], heatmap)

        plt.subplot(3, 3, i + 1)
        plt.imshow(overlay)
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[pred_labels[idx]]
        plt.title(f"true: {true_name}\npred: {pred_name}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Grad-CAM gallery saved to %s", out_path)


def main():
    logger.info("Loading model from %s", MODEL_OUT)
    model = tf.keras.models.load_model(MODEL_OUT)

    logger.info("Building test dataset from %s", DATA_DIR)
    _, _, test_ds, class_names = get_datasets(DATA_DIR)

    logger.info("Running the test set once through the model")
    y_true, y_pred, confidences, images = collect_predictions(model, test_ds)

    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    wrong_by_confidence = wrong[np.argsort(-confidences[wrong])]

    save_gradcam_gallery(
        model, images, correct, y_true, y_pred, class_names,
        os.path.join(REPORTS_DIR, "gradcam_correct.png"),
    )
    if len(wrong_by_confidence) > 0:
        save_gradcam_gallery(
            model, images, wrong_by_confidence, y_true, y_pred, class_names,
            os.path.join(REPORTS_DIR, "gradcam_misclassified.png"),
        )
    else:
        logger.info("No misclassified images — skipping misclassified Grad-CAM gallery.")


if __name__ == "__main__":
    main()
