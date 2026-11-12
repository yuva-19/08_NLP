
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):

    def __init__(self, dataframe, transform=None, label_mapping=None):

        if isinstance(dataframe, str):
            dataframe = pd.read_csv(dataframe)

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(row["image_path"]).convert("RGB")
        identity_id = int(row["identity_id"])

        if self.transform:
            image = self.transform(image)

        if self.label_mapping is not None:
            label = self.label_mapping[identity_id]
        else:
            label = identity_id

        return image, label
