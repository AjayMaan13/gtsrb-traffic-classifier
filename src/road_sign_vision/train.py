import logging

from road_sign_vision.config import DATA_DIR, EPOCHS, MODEL_OUT
from road_sign_vision.data import get_datasets, save_sample_grid
from road_sign_vision.model import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Building datasets from %s", DATA_DIR)
    train_ds, val_ds, test_ds, class_names = get_datasets(DATA_DIR)

    save_sample_grid(train_ds, class_names)

    model = get_model()

    logger.info("Training for %d epochs", EPOCHS)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, accuracy = model.evaluate(test_ds, verbose=2)
    logger.info("Test loss: %.4f, test accuracy: %.4f", loss, accuracy)

    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)


if __name__ == "__main__":
    main()
