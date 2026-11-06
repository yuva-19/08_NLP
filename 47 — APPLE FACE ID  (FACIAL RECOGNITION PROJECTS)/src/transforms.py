
from torchvision import transforms


def get_train_transforms():

    return transforms.Compose([
        transforms.Resize((112, 112)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=5),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def get_eval_transforms():

    return transforms.Compose([
        transforms.Resize((112, 112)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
