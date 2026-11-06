
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):

    def __init__(
        self,
        embedding_size=128,
        num_classes=5103,
        scale=64.0,
        margin=0.5
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Learnable class weight matrix
        self.weight = nn.Parameter(
            torch.randn(num_classes, embedding_size)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):

        # Normalize embeddings and class weights
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        # Calculate sine from cosine
        sine = torch.sqrt(
            torch.clamp(1.0 - cosine ** 2, min=0.0)
        )

        # Apply angular margin
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability adjustment
        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.mm
        )

        # Modify only the target class
        one_hot = F.one_hot(
            labels,
            num_classes=self.num_classes
        ).float()

        output = (
            one_hot * phi +
            (1.0 - one_hot) * cosine
        )

        # Apply scaling factor
        output *= self.scale

        return output
