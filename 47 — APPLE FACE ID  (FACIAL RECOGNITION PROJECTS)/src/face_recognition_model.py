
import torch
import torch.nn as nn

from src.mobilefacenet import MobileFaceNet
from src.arcface import ArcFace


class FaceRecognitionModel(nn.Module):

    def __init__(
        self,
        embedding_size=128,
        num_classes=5103
    ):
        super().__init__()

        # Feature extraction network
        self.backbone = MobileFaceNet(
            embedding_size=embedding_size
        )

        # ArcFace classification head
        self.arcface = ArcFace(
            embedding_size=embedding_size,
            num_classes=num_classes
        )

    def forward(self, images, labels):

        # Extract face embeddings
        embeddings = self.backbone(images)

        # Calculate ArcFace classification logits
        logits = self.arcface(embeddings, labels)

        return embeddings, logits
