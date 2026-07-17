import tensorflow as tf

from road_sign_vision.config import (
    FINE_TUNE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_CATEGORIES,
    TRANSFER_IMG_SIZE,
)

# Safe augmentations for traffic signs: small rotations/zoom/brightness only.
# No horizontal flip and no large rotation — those can turn one sign's meaning
# into a different (or nonsensical) one, e.g. flipping a "keep right" arrow.
_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
], name="augmentation")


def get_baseline_model():
    """
    The original CS50 CNN, as the control for comparison:
    Conv2D(32) -> MaxPool -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(softmax).
    """
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_deep_model(augment=True):
    """
    A deeper CNN: 3 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout),
    optionally with data augmentation applied before the first conv block.
    """
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs

    if augment:
        x = _augmentation(x)

    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="deep_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def get_transfer_model(fine_tune=FINE_TUNE):
    """
    MobileNetV2 pretrained on ImageNet as a frozen feature extractor, with a
    small classifier head trained on top. If `fine_tune` is True, unfreezes
    the base and continues training end-to-end at a low learning rate.

    MobileNetV2 requires inputs of at least 32x32; GTSRB images are resized
    to IMG_WIDTH/IMG_HEIGHT (30x30 by default), so we upscale to
    TRANSFER_IMG_SIZE before feeding the base model.
    """
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Resizing(TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE)(inputs)
    # MobileNetV2 expects its own specific preprocessing (scales pixels to
    # [-1, 1]) — not the plain /255 rescaling the other two models use.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(TRANSFER_IMG_SIZE, TRANSFER_IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = fine_tune

    x = base(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    # Fine-tuning uses a much smaller learning rate so it nudges the
    # pretrained weights instead of overwriting them.
    learning_rate = 1e-5 if fine_tune else 1e-3
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
