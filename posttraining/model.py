from logging import Logger
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from common.encoder import TransformerEncoderConfig, TransformerEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional

@dataclass
class ModelConfig:
    proj_dim: int

class Model(nn.Module):
    def __init__(
        self,
        encoder: TransformerEncoder,
        model_config: ModelConfig
    ):
        super(Model, self).__init__()
        self.encoder = encoder
        self.lhs = nn.Linear(encoder.config.d_model, model_config.proj_dim)
        self.rhs = nn.Linear(encoder.config.d_model, model_config.proj_dim)

    def forward(self, lhs, rhs, labels : Optional[torch.Tensor] = None):
        encoded_lhs = self.encoder.encode(**lhs)
        encoded_rhs = self.encoder.encode(**rhs)
        encoded_lhs = self.lhs(encoded_lhs)
        encoded_rhs = self.rhs(encoded_rhs)
        output = encoded_lhs * encoded_rhs
        output = output.sum(dim=1)
        if labels is None:
            return { "output" : output }
        loss = F.binary_cross_entropy_with_logits(output, labels.float())
        return { "output" : output, "loss" : loss }

def prepare_model(
    encoder_cfg : DictConfig, 
    model_cfg : DictConfig, 
    log : Logger,
    state_dict : Optional[str] = None, 
) -> Model:
    log.info("Preparing encoder config")
    encoder_config = TransformerEncoderConfig(
        num_layer = encoder_cfg.num_layer,
        num_head = encoder_cfg.num_head,
        d_model = encoder_cfg.d_model,
        d_ff = encoder_cfg.d_ff,
        dropout = encoder_cfg.dropout,
        max_len = encoder_cfg.max_len,
        vocab_size = encoder_cfg.vocab_size,
        num_kind = encoder_cfg.num_kind,
        pad_token = encoder_cfg.pad_token,
        pad_kind = encoder_cfg.pad_kind,
        kind_weight = encoder_cfg.kind_weight,
    )
    log.info("Preparing encoder")
    encoder = TransformerEncoder(encoder_config)
    log.info("Loading state dict")
    if state_dict is not None: 
        encoder.load_state_dict(torch.load(state_dict, weights_only=True))
    log.info("Preparing model config")
    model_config = ModelConfig(
        proj_dim = model_cfg.proj_dim
    )
    log.info("Preparing model")
    model = Model(encoder, model_config)
    return model

