from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from transformers import ClapConfig, ClapModel


@dataclass
class ModelOutputs:
    raw_outputs: torch.FloatTensor
    embeddings: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None


class FoundationModel(nn.Module, ABC):

    def __init__(self, backbone, proj_head=None, pred_head=None):
        super().__init__()
        self.backbone = backbone
        self.proj_head = proj_head
        self.pred_head = pred_head

    def forward(self, x):
        raw_outputs, embeddings = self.forward_backbone(x)
        if self.pred_head is None:
            return ModelOutputs(raw_outputs=raw_outputs, embeddings=embeddings)
        predictions = self.pred_head(embeddings if embeddings is not None else raw_outputs)
        return ModelOutputs(raw_outputs=raw_outputs, embeddings=embeddings, predictions=predictions)

    @abstractmethod
    def forward_backbone(self, x):
        pass

    @staticmethod
    @abstractmethod
    def from_base_model(base_model, projection_head, random_init, prediction_head):
        pass

    @staticmethod
    def mlp_head(d_input, hidden_channels):
        return ops.MLP(in_channels=d_input, hidden_channels=hidden_channels, norm_layer=nn.BatchNorm1d, dropout=0.5)


class CLAPBasedModel(FoundationModel):

    @staticmethod
    def from_base_model(base_model, projection_head, random_init, prediction_head):
        clap = ClapModel(ClapConfig.from_pretrained(base_model)) if random_init else ClapModel.from_pretrained(base_model)
        proj_head = clap.audio_projection if projection_head else None
        backbone = clap.audio_model
        d_input = backbone.config.projection_dim if projection_head else backbone.config.hidden_size
        pred_head = FoundationModel.mlp_head(d_input, prediction_head) if prediction_head else None
        return CLAPBasedModel(backbone, proj_head, pred_head)

    def forward_backbone(self, x):
        n, m = x.shape
        d_input = self.backbone.config.num_mel_bins
        multiple = (m // d_input) * d_input
        x = x[:, :multiple].reshape(n, 1, -1, d_input)
        raw_outputs = self.backbone(x).pooler_output
        proj_embeds = None
        if self.proj_head is not None:
            proj_embeds = self.proj_head(raw_outputs)
            proj_embeds = F.normalize(proj_embeds)
        return raw_outputs, proj_embeds


construct = {
    'laion/clap-htsat-unfused': CLAPBasedModel,
}


def construct_model(base_model, projection_head, random_init, prediction_head):
    return construct[base_model].from_base_model(base_model, projection_head, random_init, prediction_head)
