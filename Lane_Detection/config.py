
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'tusimple')
TRAIN_SET_DIR = os.path.join(DATA_DIR, 'train_set')
TEST_SET_DIR = os.path.join(DATA_DIR, 'test_set')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_CLASSES = 6  # Background + 5 lanes
IMG_HEIGHT = 288 # Resize target height (TuSimple is 720x1280)
IMG_WIDTH = 512  # Resize target width

# Checkpoints
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
