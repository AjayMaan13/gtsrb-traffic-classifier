import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from road_sign_vision.config import DATA_DIR, EPOCHS, MODEL_OUT, TEST_SIZE
from road_sign_vision.data import load_data
from road_sign_vision.model import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading data from %s", DATA_DIR)
    images, labels = load_data(DATA_DIR)

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    logger.info("Training for %d epochs", EPOCHS)
    model.fit(x_train, y_train, epochs=EPOCHS)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    logger.info("Test loss: %.4f, test accuracy: %.4f", loss, accuracy)

    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)


if __name__ == "__main__":
    main()
