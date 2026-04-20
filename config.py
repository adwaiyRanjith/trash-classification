from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

NUM_CLASSES = 8
BATCH_SIZE = 32
LR = 1e-3


NUM_EPOCHS = 40

PATIENCE = 40