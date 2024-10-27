from logging import Logger
from dataclasses import dataclass
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from common.tokenizer import prepare_tokenizer, Tokenizer
import os
import json
import torch

from typing import Optional, List, Dict

@dataclass
class PretrainingDatasetConfig:
    datadir : str
    datafile : str
    pad_kind : int
    lhs_kind : int
    rhs_kind : int
    max_datapoints : Optional[int]

class PretrainingDataset(Dataset):
    def __init__(
        self, 
        config : PretrainingDatasetConfig,
        tokenizer : Tokenizer
    ):
        self.config = config
        self.corpus = os.path.join(config.datadir, config.datafile)
        self.tokenizer = tokenizer
        self.data = []
        print("Loading corpus. This may take a while...")
        with open(self.corpus, 'r') as f:
            for l in tqdm(f):
                self.data.append(json.loads(l))
                if self.config.max_datapoints is not None and len(self.data) >= self.config.max_datapoints:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch : List[Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        max_len = max([len(x['lhs']) + len(x['rhs']) for x in batch])
        tokens_batch = []
        masked_tokens_batch = []
        kinds_batch = []
        for x in batch:
            lhs = x['lhs']
            rhs = x['rhs']
            tokens = self.tokenizer.tokenize(lhs + rhs, length=max_len, shuffle=False, mask=True)
            tokens_batch.append(tokens['tokens'])
            masked_tokens_batch.append(tokens['masked_tokens'])
            kinds = []
            kinds += [self.config.lhs_kind] * len(lhs)
            kinds += [self.config.rhs_kind] * len(rhs)
            kinds += [self.config.pad_kind] * (max_len - len(lhs) - len(rhs))
            kinds_batch.append(kinds)
        return {
            'tgt' : torch.tensor(tokens_batch),
            'src' : torch.tensor(masked_tokens_batch),
            'kinds' : torch.tensor(kinds_batch)
        }

def prepare_dataset(tokenizer_cfg : DictConfig, dataset_cfg : DictConfig, log : Logger) -> PretrainingDataset:
    log.info("Preparing tokenizer")
    tokenizer = prepare_tokenizer(tokenizer_cfg, log)
    log.info("Preparing dataset config")
    dataset_config = PretrainingDatasetConfig(
        datadir = dataset_cfg.datadir,
        datafile = dataset_cfg.datafile,
        pad_kind = dataset_cfg.pad_kind,
        lhs_kind = dataset_cfg.lhs_kind,
        rhs_kind = dataset_cfg.rhs_kind,
        max_datapoints = dataset_cfg.max_datapoints
    )
    log.info("Initializing dataset")
    dataset = PretrainingDataset(
        config = dataset_config,
        tokenizer = tokenizer
    )
    log.info("Dataset initialized")
    return dataset

