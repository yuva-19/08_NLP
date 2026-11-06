
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import (
    EMBEDDING_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MIN_LR,
    CHECKPOINT_DIR,
    AUTO_RESUME,
)

from src.transforms import get_train_transforms
from src.dataset import FaceRecognitionDataset
from src.face_recognition_model import FaceRecognitionModel


# ============================================================
# 1. PATHS AND DEVICE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_ROOT/ "data"/ "celeba"/ "raw"/ "img_align_celeba"
)

TRAIN_CSV = (
    PROJECT_ROOT/ "data"/ "celeba"/ "processed"/ "train.csv"
)

CHECKPOINT_DIR = Path(CHECKPOINT_DIR)

if not CHECKPOINT_DIR.is_absolute():
    CHECKPOINT_DIR = PROJECT_ROOT / CHECKPOINT_DIR

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "epoch_checkpoint.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

train_df = pd.read_csv(TRAIN_CSV)

# Convert original identity IDs into contiguous ArcFace labels.
original_identity_ids = sorted(
    int(identity)
    for identity in train_df["identity"].unique()
)

identity_to_class = {
    identity: class_index
    for class_index, identity in enumerate(original_identity_ids)
}

if len(identity_to_class) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} identities, "
        f"but found {len(identity_to_class)}."
    )

# Create training dataset.
train_dataset = FaceRecognitionDataset(
    train_df,
    IMAGE_DIR,
    get_train_transforms(),
)

# Create training DataLoader.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Training images:", len(train_dataset))
print("Training identities:", len(identity_to_class))
print("Batches per epoch:", len(train_loader))


# ============================================================
# 3. CREATE MODEL, LOSS, OPTIMIZER AND SCHEDULER
# ============================================================

model = FaceRecognitionModel(
    embedding_size=EMBEDDING_SIZE,
    num_classes=NUM_CLASSES,
).to(device)

criterion = nn.CrossEntropyLoss()

# Optimizer: SGD with momentum
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=MIN_LR,
)


# ============================================================
# 4. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(completed_epoch, global_step):

    checkpoint = {
        "completed_epoch": completed_epoch,
        "global_step": global_step,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),

        "identity_to_class": identity_to_class,
    }

    temporary_path = CHECKPOINT_PATH.with_name(
        CHECKPOINT_PATH.name + ".tmp"
    )

    torch.save(checkpoint, temporary_path)

    os.replace(
        temporary_path,
        CHECKPOINT_PATH
    )


# ============================================================
# 5. LOAD CHECKPOINT IF AVAILABLE
# ============================================================

start_epoch = 0
global_step = 0

if AUTO_RESUME and CHECKPOINT_PATH.exists():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # Ensure the identity mapping is unchanged.
    if checkpoint["identity_to_class"] != identity_to_class:
        raise ValueError(
            "Training identity mapping differs from checkpoint."
        )

    # Restore model parameters.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore optimizer state.
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # Restore learning-rate scheduler.
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # Continue from the next epoch.
    start_epoch = int(checkpoint["completed_epoch"])

    global_step = int(checkpoint["global_step"])

    print("\nCheckpoint loaded successfully!")

    print("Completed epochs:", start_epoch)
    print("Next epoch:", start_epoch + 1)
    print("Global training steps:", global_step)

else:

    print("\nStarting training from the beginning.")


# ============================================================
# 6. TRAINING LOOP
# ============================================================

if start_epoch >= NUM_EPOCHS:

    print("\nAll configured epochs are already completed.")

else:

    for epoch in range(start_epoch, NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        )

        for images, original_identities in progress_bar:

            # Convert original identity IDs into ArcFace labels.
            labels = torch.tensor(
                [
                    identity_to_class[int(identity)]
                    for identity in original_identities
                ],
                dtype=torch.long,
            )

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------
            # Forward pass
            # ------------------------------------

            optimizer.zero_grad(set_to_none=True)

            embeddings, logits = model(images, labels)

            loss = criterion(logits, labels)

            # ------------------------------------
            # Backpropagation
            # ------------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------------
            # Update training statistics
            # ------------------------------------

            global_step += 1

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        # ========================================
        # EPOCH COMPLETED
        # ========================================

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1} average loss: "
            f"{average_loss:.4f}"
        )

        # Update learning rate once per epoch.
        scheduler.step()

        print(
            "Learning rate:",
            optimizer.param_groups[0]["lr"],
        )

        # ========================================
        # SAVE CHECKPOINT
        # ========================================

        save_checkpoint(
            completed_epoch=epoch + 1,
            global_step=global_step,
        )

        print(
            f"Checkpoint saved after Epoch {epoch + 1}."
        )


# ============================================================
# 7. TRAINING COMPLETED
# ============================================================

print("\nTraining completed!")
