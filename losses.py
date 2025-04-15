import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5, normalize=False):
        """
        Computes the **Normalized Temperature-Scaled Cross-Entropy** Loss.

        Reference: https://theaisummer.com/simclr/

        Parameters
        ----------
        temperature: float
            Temperature scaling factor, which must be positive. Higher values reduce the variance in the outputs, softening the probability distribution.
        normalize: bool
            Whether the embeddings should be normalized before computing the loss. If the embeddings are already normalized, this should be set to False.
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, zi, zj):
        batch_size, embed_dim = zi.shape

        if self.normalize:
            zi = F.normalize(zi.clone(), p=2, dim=1)
            zj = F.normalize(zj.clone(), p=2, dim=1)

        sim_matrix = self.cosine_similarity(zi, zj)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        mask = self.negatives_mask(batch_size, device=sim_matrix.device)
        denominator = mask * torch.exp(sim_matrix / self.temperature)
        denominator = torch.sum(denominator, dim=1)
        losses = -torch.log(nominator / (denominator + 1e-6))
        return torch.sum(losses) / (2 * batch_size)

    @classmethod
    def cosine_similarity(cls, zi, zj):
        z = torch.cat([zi, zj], dim=0)
        return torch.mm(z, z.T)

    @classmethod
    @functools.cache
    def negatives_mask(cls, num, device):
        mask = 1 - torch.eye(2 * num)  # Everything but the diagonal
        for i in range(num):
            # Different augmentations of the same sample
            mask[i, num + i] = 0
            mask[num + i, i] = 0
        return mask.to(device)


class TripletLoss(nn.TripletMarginLoss):

    def __init__(self, margin=1.0, p=2.0):
        """
        Computes the **Triplet Loss** with hard negative mining.

        Parameters
        ----------
        margin: float
            Slack margin, which must be positive. Default: 1.
        p: float
            The norm degree for pairwise distance. Default: 2.
        """
        super().__init__(margin=margin, p=p)

    # noinspection PyMethodOverriding
    def forward(self, embeds, labels):
        anchors = embeds  # Every sample in the spectra acts as anchor.
        positives = self.draw_uniformly(anchors, labels, same_class=True)
        negatives = self.draw_closer_ones(anchors, labels, same_class=False)
        return super().forward(anchors, positives, negatives)

    @classmethod
    def draw_uniformly(cls, anchors, labels, same_class):
        li = [random.choice(anchors[labels == y if same_class else labels != y]) for y in labels]
        return torch.stack(li)

#     @classmethod
#     def draw_closer_ones(cls, anchors, labels, same_class, eps=1e-8):
#         proba = 1 / (torch.cdist(anchors, anchors) + eps)  # Selection likelihood inversely proportional to distance: closer samples are more likely to be selected
#         li = [random.choices(anchors[labels == y if same_class else labels != y], weights=proba[i, labels == y if same_class else labels != y])[0] for i, y in enumerate(labels)]
#         return torch.stack(li)

    @classmethod
    def draw_closer_ones(cls, anchors, labels, same_class, eps=1e-8):
        proba = 1 / (torch.cdist(anchors, anchors) + eps)
        li = []
        fallback_count = 0
        for i, y in enumerate(labels):
            valid_samples = anchors[labels == y if same_class else labels != y]
            if valid_samples.numel() > 0:  # Ensure there's at least one valid sample
                selected = random.choices(valid_samples, weights=proba[i, labels == y if same_class else labels != y])[0]
                li.append(selected)
            else:
                # Fallback to using the anchor itself
                li.append(anchors[i])
        return torch.stack(li)
