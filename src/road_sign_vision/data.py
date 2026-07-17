import logging
import os

import cv2

from road_sign_vision.config import IMG_HEIGHT, IMG_WIDTH, NUM_CATEGORIES

logger = logging.getLogger(__name__)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        folder = os.path.join(data_dir, str(category))
        filenames = os.listdir(folder)
        logger.info("Loading category %s: %d images", category, len(filenames))
        for filename in filenames:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)

    logger.info("Loaded %d images across %d categories", len(images), NUM_CATEGORIES)
    return (images, labels)
