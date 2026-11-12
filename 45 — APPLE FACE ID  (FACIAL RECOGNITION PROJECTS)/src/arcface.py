
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_classes,
        scale=64.0,
        margin=0.5
    ):

        super(ArcFace, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_size)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)

        sine = torch.sqrt(
            torch.clamp(1.0 - cosine.pow(2), min=1e-7)
        )

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.mm
        )

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1.0
        )

        output = (
            one_hot * phi +
            (1.0 - one_hot) * cosine
        )

        output *= self.scale

        return output
