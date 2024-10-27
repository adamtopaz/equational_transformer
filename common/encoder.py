from logging import Logger
from omegaconf import DictConfig
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

@dataclass
class TransformerEncoderConfig:
    num_layer : int
    num_head : int
    d_model : int
    d_ff : int
    dropout : float
    max_len : int
    vocab_size : int
    num_kind : int
    pad_token : int
    pad_kind : int
    kind_weight : float

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        config : TransformerEncoderConfig,
    ):
        super().__init__()
        self.config = config

        self.positional_encoding = nn.Embedding(
            self.config.max_len, 
            self.config.d_model
        )

        self.kind_embedding = nn.Embedding(
            self.config.num_kind,
            self.config.d_model
        )

        self.embedding = nn.Embedding(
            self.config.vocab_size, 
            self.config.d_model
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model, 
            nhead=self.config.num_head, 
            dim_feedforward=self.config.d_ff, 
            dropout=self.config.dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=self.config.num_layer
        )

        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.lm_head.weight = self.embedding.weight

        self.kind_head = nn.Linear(self.config.d_model, self.config.num_kind, bias=False)

        self.kind_head.weight = self.kind_embedding.weight

    def forward(
        self, 
        src : torch.Tensor, 
        kinds : torch.Tensor, 
        tgt : Optional[torch.Tensor] = None
    ):
        padding_mask = (src == self.config.pad_token)
        seq_len = src.size(1)
        pos = torch.arange(seq_len, device=src.device).expand(src.size(0), -1)
        pos = self.positional_encoding(pos)
        src = self.embedding(src)
        kind_embs = self.kind_embedding(kinds)
        src = src + pos + kind_embs
        encoder_output = self.encoder(
            src, 
            src_key_padding_mask=padding_mask, 
            is_causal=False
        )
        if tgt is not None:
            logits = self.lm_head(encoder_output)
            kind_logits = self.kind_head(encoder_output)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), 
                tgt.view(-1), 
                ignore_index=self.config.pad_token
            )
            kind_loss = F.cross_entropy(
                kind_logits.view(-1, self.config.num_kind),
                kinds.view(-1),
                ignore_index=self.config.pad_kind
            )
            return { 
                'encoder_output' : encoder_output, 
                'logits' : logits,
                'kind_logits' : kind_logits,
                'loss' : loss + self.config.kind_weight * kind_loss
            }
        return { 'encoder_output' : encoder_output }

    def encode(
        self,
        src : torch.Tensor,
        kinds : torch.Tensor
    ):
        encoder_output = self.forward(src, kinds, tgt=None)['encoder_output']
        mask = (src != self.config.pad_token).float().unsqueeze(-1)
        encoder_output = encoder_output * mask
        encoder_output = encoder_output.sum(dim=1) / mask.sum(dim=1)
        return encoder_output

def prepare_encoder(encoder_cfg : DictConfig, log : Logger) -> TransformerEncoder:
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
        kind_weight = encoder_cfg.kind_weight
    )
    log.info("Preparing encoder")
    encoder = TransformerEncoder(encoder_config)
    log.info("Encoder prepared")
    return encoder 
