import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
import numpy as np
import tensorflow as tf

from road_sign_vision.config import IMG_HEIGHT, IMG_WIDTH, MODEL_OUT, PORT
from road_sign_vision.gradcam import make_gradcam_heatmap, overlay_heatmap

CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

# Loaded ONCE at startup, not inside predict() — 12-factor V/VI/IX: this app
# only ever does inference against an already-trained artifact, never trains,
# and holds no per-request state between calls.
model = tf.keras.models.load_model(MODEL_OUT)
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)


def predict(image):
    if image is None:
        return None, None

    resized = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH)).numpy()
    probs = model.predict(resized[np.newaxis, ...], verbose=0)[0]
    confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    heatmap, _ = make_gradcam_heatmap(resized, model)
    overlay = overlay_heatmap(resized, heatmap)

    return confidences, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a traffic sign photo"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top predictions"),
        gr.Image(label="Grad-CAM: where the model looked"),
    ],
    title="road-sign-vision",
    description="Upload a traffic sign photo to classify it (GTSRB, 43 categories) and see where the model focused when predicting.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
