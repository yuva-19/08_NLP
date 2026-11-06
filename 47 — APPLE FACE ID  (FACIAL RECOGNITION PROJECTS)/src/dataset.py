
from torch.utils.data import Dataset
from PIL import Image

class FaceRecognitionDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self.image_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        identity = int(row["identity"])

        return image, identity
