
import torch.nn as nn

from .mobilefacenet import MobileFaceNet
from .arcface import ArcFace


class FaceRecognitionModel(nn.Module):

    def __init__(
        self,
        embedding_size=128,
        num_classes=5103
    ):

        super(FaceRecognitionModel, self).__init__()

        self.backbone = MobileFaceNet(
            embedding_size=embedding_size
        )

        self.arcface = ArcFace(
            embedding_size=embedding_size,
            num_classes=num_classes
        )

    def forward(self, images, labels=None):

        embeddings = self.backbone(images)

        if labels is None:
            return embeddings

        logits = self.arcface(embeddings, labels)

        return logits, embeddings
