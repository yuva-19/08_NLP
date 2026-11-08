



import numpy as np
import torch
import torchvision.transforms as transforms

from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------
# 1. DATASET CLASS
# -----------------------------------------

class LandmarkDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        category_to_label,
        transform=None
    ):
        self.dataset = hf_dataset
        self.category_to_label = category_to_label
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"].convert("RGB")
        category = sample["category"]

        label = self.category_to_label[category]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# -----------------------------------------
# 2. PREPROCESSING FUNCTION
# -----------------------------------------

def preprocessing(
    dataset,
    target_images=500,
    batch_size=32,
    val_size=0.10,
    random_state=42
):

    # -------------------------------------
    # Train-validation split
    # -------------------------------------

    train_data = dataset["train"]

    categories = train_data["category"]

    train_indices, val_indices = train_test_split(
        np.arange(len(train_data)),
        test_size=val_size,
        random_state=random_state,
        stratify=categories
    )

    train_split = train_data.select(
        train_indices.tolist()
    )

    val_split = train_data.select(
        val_indices.tolist()
    )

    print("Original training samples:", len(train_data))
    print("Training samples:", len(train_split))
    print("Validation samples:", len(val_split))


    # -------------------------------------
    # Balance training dataset
    # -------------------------------------

    category_to_indices = defaultdict(list)

    for index, category in enumerate(train_split["category"]):
        category_to_indices[category].append(index)

    rng = np.random.default_rng(random_state)

    balanced_indices = []

    for category, indices in category_to_indices.items():

        selected_indices = rng.choice(
            indices,
            size=target_images,
            replace=len(indices) < target_images
        )

        balanced_indices.extend(
            selected_indices.tolist()
        )

    rng.shuffle(balanced_indices)

    balanced_train_data = train_split.select(
        balanced_indices
    )

    balanced_distribution = Counter(
        balanced_train_data["category"]
    )

    print("\nBalanced training samples:", len(balanced_train_data))
    print("Number of classes:", len(balanced_distribution))

    print("\nClass distribution:")

    for category, count in sorted(balanced_distribution.items()):
        print(f"{category}: {count}")


    # -------------------------------------
    # Training augmentation
    # -------------------------------------

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.3
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])


    # -------------------------------------
    # Validation preprocessing
    # -------------------------------------

    val_transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])


    # -------------------------------------
    # Label encoding
    # -------------------------------------

    categories = sorted(
        balanced_train_data.unique("category")
    )

    category_to_label = {
        category: index
        for index, category in enumerate(categories)
    }

    label_to_category = {
        index: category
        for category, index in category_to_label.items()
    }


    # -------------------------------------
    # Create PyTorch datasets
    # -------------------------------------

    train_dataset = LandmarkDataset(
        hf_dataset=balanced_train_data,
        category_to_label=category_to_label,
        transform=train_transform
    )

    val_dataset = LandmarkDataset(
        hf_dataset=val_split,
        category_to_label=category_to_label,
        transform=val_transform
    )


    # -------------------------------------
    # Create DataLoaders
    # -------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nPreprocessing completed!")
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    return (
    train_loader,
    val_loader,
    category_to_label,
    label_to_category
)
