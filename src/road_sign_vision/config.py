import os

from dotenv import load_dotenv

load_dotenv()

EPOCHS = int(os.environ.get("EPOCHS", 10))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", 30))
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", 30))
NUM_CATEGORIES = int(os.environ.get("NUM_CATEGORIES", 43))
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.4))
DATA_DIR = os.environ.get("DATA_DIR", "data/gtsrb")
MODEL_OUT = os.environ.get("MODEL_OUT", "models/model.keras")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", 0.3))
SEED = int(os.environ.get("SEED", 123))
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")

# Phase 4 — experiment selection and tracking
EXPERIMENT = os.environ.get("EXPERIMENT", "baseline")  # baseline | deep | transfer
AUGMENT = os.environ.get("AUGMENT", "true").lower() == "true"
FINE_TUNE = os.environ.get("FINE_TUNE", "false").lower() == "true"
PATIENCE = int(os.environ.get("PATIENCE", 5))
TRANSFER_IMG_SIZE = int(os.environ.get("TRANSFER_IMG_SIZE", 96))
RESULTS_CSV = os.environ.get("RESULTS_CSV", "reports/experiment_results.csv")
