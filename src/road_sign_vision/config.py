import os

from dotenv import load_dotenv

load_dotenv()

EPOCHS = int(os.environ.get("EPOCHS", 10))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", 30))
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", 30))
NUM_CATEGORIES = int(os.environ.get("NUM_CATEGORIES", 43))
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.4))
DATA_DIR = os.environ.get("DATA_DIR", "data/gtsrb")
MODEL_OUT = os.environ.get("MODEL_OUT", "models/model.h5")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.3))
SEED = int(os.environ.get("SEED", 123))
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
