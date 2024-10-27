import logging
from logging import Logger
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import os
import shutil
import json
import random

import requests

def download_file(url, path, log):
    log.info(f"Downloading {url} to {path}")
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status() 

            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 1024

            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                log.error("ERROR: Download incomplete.")
            return path
    except Exception as e:
        log.error(f"An error occurred: {e}")
        raise

def generate(cfg: DictConfig, log : Logger) -> None:
    log.info("Copying random tokenized equations to appropriate location.")
    source = cfg.dataprep.random_tokenized_equations
    target = os.path.join(cfg.pretraining.dataset.datadir, cfg.pretraining.dataset.datafile)
    shutil.copy(source, target)
    log.info("Done copying random tokenized equations.")

    log.info("Copying tokenized equations file to appropriate location.")
    source = cfg.dataprep.tokenized_equations
    target = os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.equations)
    shutil.copy(source, target)
    log.info("Done copying tokenized equations file.")

    log.info("Loading implications into memory.")
    implications = []
    with open(cfg.dataprep.implications) as f:
        for l in tqdm(f):
            implications.append(json.loads(l))
    log.info("Done loading implications into memory.")
    log.info("Computing used equations.")
    equations = set()
    for i in tqdm(implications):
        equations.add(i["lhs"])
        equations.add(i["rhs"])
    log.info("Done computing used equations.")
    
    log.info("Shuffling equations.")
    equations = list(equations)
    random.shuffle(equations)
    log.info("Done shuffling equations.")

    log.info("Splitting up equations into training and test/validation groups.")
    split = int(len(equations) * cfg.dataprep.split_ratio)
    train_equations = set(equations[:split])
    testval_equations = set(equations[split:])

    log.info("Computing training and test/validation implications.")
    train_implications = []
    testval_implications = []
    if cfg.dataprep.use_complement:
        log.info("Using complement for test/validation equations.")
        for i in tqdm(implications):
            if i["lhs"] in train_equations and i["rhs"] in train_equations:
                train_implications.append(i)
            else: 
                testval_implications.append(i)
    else:
        log.info("Not using complement for test/validation equations.")
        for i in tqdm(implications):
            if i["lhs"] in train_equations and i["rhs"] in train_equations:
                train_implications.append(i)
            elif i["lhs"] in testval_equations and i["rhs"] in testval_equations:
                testval_implications.append(i)
    log.info(f"Computed {len(train_implications)} training implications.")
    log.info(f"Computed {len(testval_implications)} test/validation implications.")
    log.info(f"Shuffling test/validation implications.")
    random.shuffle(testval_implications)
    log.info("Done shuffling test/validation implications.")
    log.info("Splitting up test and validation sets.")
    total = len(testval_implications)
    split = int(total * cfg.dataprep.val_ratio)
    validation_implications = testval_implications[:split]
    test_implications = testval_implications[split:]
    log.info(f"Computed {len(validation_implications)} validation implications.")
    log.info(f"Computed {len(test_implications)} test implications.")

    log.info("Writing training implications to appropriate location.")
    with open(os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.train), "w") as f:
        for i in tqdm(train_implications):
            f.write(json.dumps(i) + "\n")
    log.info("Done writing training implications to appropriate location.")
    log.info("Writing validation implications to appropriate location.")
    with open(os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.val), "w") as f:
        for i in tqdm(validation_implications):
            f.write(json.dumps(i) + "\n")
    log.info("Done writing validation implications to appropriate location.")
    log.info("Writing test implications to appropriate location.")
    with open(os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.test), "w") as f:
        for i in tqdm(test_implications):
            f.write(json.dumps(i) + "\n")
    log.info("Done writing test implications to appropriate location.")

def download(cfg : DictConfig, log : Logger):
    base_url = cfg.dataprep.base_url
    log.info(f"Using base URL: {base_url}")

    log.info("Downloading random tokenized equations.")
    url = os.path.join(base_url, cfg.dataprep.random_tokenized_equations)
    download_file(url, cfg.dataprep.random_tokenized_equations, log)
    log.info("Done downloading random tokenized equations.")

    log.info("Downloading tokenized equations.")
    url = os.path.join(base_url, cfg.dataprep.tokenized_equations)
    download_file(url, cfg.dataprep.tokenized_equations, log)
    log.info("Done downloading tokenized equations.")

    log.info("Downloading implications.")
    url = os.path.join(base_url, cfg.dataprep.implications)
    download_file(url, cfg.dataprep.implications, log)
    log.info("Done downloading implications.")

    log.info("Copying random tokenized equations to appropriate location.")
    source = cfg.dataprep.random_tokenized_equations
    target = os.path.join(cfg.pretraining.dataset.datadir, cfg.pretraining.dataset.datafile)
    shutil.copy(source, target)
    log.info("Done copying random tokenized equations.")

    log.info("Copying tokenized equations file to appropriate location.")
    source = cfg.dataprep.tokenized_equations
    target = os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.equations)
    shutil.copy(source, target)
    log.info("Done copying tokenized equations file.")

    log.info("Downloading train implications.")
    url = os.path.join(base_url, cfg.posttraining.dataset.train)
    download_file(url, os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.train), log)
    log.info("Done downloading train implications.")

    log.info("Downloading validation implications.")
    url = os.path.join(base_url, cfg.posttraining.dataset.val)
    download_file(url, os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.val), log)
    log.info("Done downloading validation implications.")

    log.info("Downloading test implications.")
    url = os.path.join(base_url, cfg.posttraining.dataset.test)
    download_file(url, os.path.join(cfg.posttraining.dataset.datadir, cfg.posttraining.dataset.test), log)
    log.info("Done downloading test implications.")

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info("Ensuring data directories exist.")
    os.makedirs(cfg.pretraining.dataset.datadir, exist_ok=True)
    os.makedirs(cfg.posttraining.dataset.datadir, exist_ok=True)
    log.info("Done ensuring data directories exist.")
    if cfg.dataprep.method == "generate":
        generate(cfg, log)
    elif cfg.dataprep.method == "download":
        download(cfg, log)

if __name__ == "__main__":
    main()
