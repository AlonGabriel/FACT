import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            zi = F.normalize(zi, p=2, dim=1)
            zj = F.normalize(zj, p=2, dim=1)

        sim_matrix = self.cosine_similarity(zi, zj)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)

        mask = self.negatives_mask(batch_size, device=sim_matrix.device)
        denominator = mask * torch.exp(sim_matrix / self.temperature)
        denominator = torch.sum(denominator, dim=1)

        losses = -torch.log(nominator / denominator)
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
