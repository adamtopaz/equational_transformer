from logging import Logger
from dataclasses import dataclass
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from common.tokenizer import prepare_tokenizer, Tokenizer
import os
import json
import torch
import random

from typing import Optional, List, Dict

@dataclass
class PosttrainingDatasetConfig:
    datadir: str
    equations: str
    train: str
    val: str
    test: str
    pad_kind: int
    lhs_kind: int
    rhs_kind: int
    max_datapoints: Optional[int]

class PosttrainingDataset(Dataset):
    def __init__(
        self,
        config: PosttrainingDatasetConfig,
        tokenizer: Tokenizer,
        tp : str
    ):
        self.config = config
        self.tp = tp
        if tp == "train":
            self.corpus = os.path.join(config.datadir, config.train)
        elif tp == "val":
            self.corpus = os.path.join(config.datadir, config.val)
        elif tp == "test":
            self.corpus = os.path.join(config.datadir, config.test)
        self.tokenizer = tokenizer
        self.equations = dict()
        with open(os.path.join(config.datadir, config.equations), 'r') as f:
            for l in f:
                j = json.loads(l)
                self.equations[j['name']] = {"lhs" : j['lhs'], "rhs" : j['rhs']}
        self.data = []
        print("Loading corpus. This may take a while...")
        with open(self.corpus, 'r') as f:
            for l in tqdm(f):
                j = json.loads(l)
                j['lhs'] = self.equations[j['lhs']]
                j['rhs'] = self.equations[j['rhs']]
                self.data.append(j)
                if self.config.max_datapoints is not None and len(self.data) >= self.config.max_datapoints:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch : List[Dict[str, Dict[str,List[str]]]]) -> Dict[str, torch.Tensor | Dict[str,torch.Tensor]]:
        lhs_tokens_batch = []
        lhs_kinds_batch = []
        rhs_tokens_batch = []
        rhs_kinds_batch = []
        labels_batch = []
        max_lhs_len = max([len(item['lhs']['lhs']) + len(item['lhs']['rhs']) for item in batch])
        max_rhs_len = max([len(item['rhs']['lhs']) + len(item['rhs']['rhs']) for item in batch])
        for item in batch:
            lhs_lhs, lhs_rhs = (item['lhs']['lhs'], item['lhs']['rhs']) if random.random() < 0.5 else (item['lhs']['rhs'], item['lhs']['lhs'])
            rhs_lhs, rhs_rhs = (item['rhs']['lhs'], item['rhs']['rhs']) if random.random() < 0.5 else (item['rhs']['rhs'], item['rhs']['lhs'])
            lhs_tokens = self.tokenizer.tokenize(lhs_lhs + lhs_rhs, max_lhs_len, shuffle=True, mask=False)['tokens']
            lhs_tokens_batch.append(lhs_tokens)

            lhs_kinds = []
            lhs_kinds += [self.config.lhs_kind] * len(lhs_lhs) 
            lhs_kinds += [self.config.rhs_kind] * len(lhs_rhs) 
            lhs_kinds += [self.config.pad_kind] * (max_lhs_len - len(lhs_lhs) - len(lhs_rhs))
            lhs_kinds_batch.append(lhs_kinds)

            rhs_tokens = self.tokenizer.tokenize(rhs_lhs + rhs_rhs, max_rhs_len, shuffle=True, mask=False)['tokens']
            rhs_tokens_batch.append(rhs_tokens)
            rhs_kinds = []
            rhs_kinds += [self.config.lhs_kind] * len(rhs_lhs)
            rhs_kinds += [self.config.rhs_kind] * len(rhs_rhs)
            rhs_kinds += [self.config.pad_kind] * (max_rhs_len - len(rhs_lhs) - len(rhs_rhs))
            rhs_kinds_batch.append(rhs_kinds)

            labels_batch.append(1 if item['isTrue'] else 0)
        return {
            "lhs" : {
                "src" : torch.tensor(lhs_tokens_batch),
                "kinds" : torch.tensor(lhs_kinds_batch),
            },
            "rhs" : {
                "src" : torch.tensor(rhs_tokens_batch),
                "kinds" : torch.tensor(rhs_kinds_batch),
            },
            "labels" : torch.tensor(labels_batch),
        }

def prepare_dataset(tokenizer_cfg : DictConfig, dataset_cfg : DictConfig, tp : str, log : Logger) -> PosttrainingDataset:
    log.info("Preparing tokenizer")
    tokenizer = prepare_tokenizer(tokenizer_cfg, log)
    log.info("Preparing dataset config")
    dataset_config = PosttrainingDatasetConfig(
        datadir = dataset_cfg.datadir,
        equations = dataset_cfg.equations,
        train = dataset_cfg.train,
        val = dataset_cfg.val,
        test = dataset_cfg.test,
        pad_kind = dataset_cfg.pad_kind,
        lhs_kind = dataset_cfg.lhs_kind,
        rhs_kind = dataset_cfg.rhs_kind,
        max_datapoints = dataset_cfg.max_datapoints
    )
    log.info("Initializing dataset")
    dataset = PosttrainingDataset(
        config = dataset_config,
        tokenizer = tokenizer,
        tp = tp
    )
    log.info("Dataset initialized")
    return dataset

