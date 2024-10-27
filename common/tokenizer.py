from logging import Logger
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

@dataclass
class TokenizerConfig:
    var_names : str
    pad_token : int
    mask_token : int
    mul_token : int
    num_special_token : int
    mask_rate : float
    vocab_size : int

class Tokenizer():
    def __init__(
        self, 
        config : TokenizerConfig
    ):
        self.config = config
        self.token_dict = {
            x : i + self.config.num_special_token for i, x in enumerate(self.config.var_names)
        }
        self.token_dict['pad'] = self.config.pad_token
        self.token_dict['mask'] = self.config.mask_token
        self.token_dict['mul'] = self.config.mul_token

    def var_perm(self, data : List[str]) -> List[str]:
        vars = [x for x in self.config.var_names]
        random.shuffle(vars)
        shuffle_dict = {x : y for x, y in zip(self.config.var_names, vars)}
        shuffle_dict['pad'] = 'pad'
        shuffle_dict['mask'] = 'mask'
        shuffle_dict['mul'] = 'mul'
        return [shuffle_dict[x] for x in data]

    def tokenize(
        self, 
        data : List[str], 
        length : Optional[int] = None, 
        mask : bool = False, 
        shuffle : bool = False
    ) -> Dict[str,List[int]]:
        if shuffle:
            data = self.var_perm(data)
        if length is not None:
            padded_data = []
            for i in range(length):
                if i < len(data):
                    padded_data.append(data[i])
                else:
                    padded_data.append('pad')
            data = padded_data
        masked_data = []
        for x in data:
            if mask and x != 'pad' and random.random() < self.config.mask_rate:
                masked_data.append('mask')
            else:
                masked_data.append(x)
        return {
            'tokens' : [self.token_dict[x] for x in data],
            'masked_tokens' : [self.token_dict[x] for x in masked_data]
        }

def prepare_tokenizer(config : DictConfig, log : Logger) -> Tokenizer:
    log.info("Preparing tokenizer config")
    tokenizer_config = TokenizerConfig(
        var_names = config.var_names,
        pad_token = config.pad_token,
        mask_token = config.mask_token,
        mul_token = config.mul_token,
        num_special_token = config.num_special_token,
        mask_rate = config.mask_rate,
        vocab_size = config.vocab_size
    )
    log.info("Initializing tokenizer")
    tokenizer = Tokenizer(tokenizer_config)
    log.info("Tokenizer initialized")
    return tokenizer
