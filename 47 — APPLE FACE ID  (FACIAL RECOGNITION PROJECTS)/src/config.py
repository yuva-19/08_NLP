
# Model configuration
EMBEDDING_SIZE = 128
NUM_CLASSES = 5103

# Training configuration
BATCH_SIZE = 48
NUM_EPOCHS = 20

# Optimizer configuration
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4

# Scheduler configuration
MIN_LR = 1e-6

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY_BATCHES = 100

# Recovery configuration
AUTO_RESUME = True
