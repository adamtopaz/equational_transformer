from logging import Logger
from omegaconf import DictConfig
from dataclasses import dataclass

from posttraining.model import Model, prepare_model
from posttraining.dataset import prepare_dataset
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from typing import Optional

class PLModel(L.LightningModule):
    def __init__(
        self,
        model : Model,
        learning_rate : float,
        weight_decay : float,
        warmup_epochs : float,
        num_epochs : int,
        epoch_size : int,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epoch_size = epoch_size
        self.num_epochs = num_epochs
        self.total_steps = self.epoch_size * self.num_epochs
        self.warmup_steps = int(self.warmup_epochs * self.epoch_size)
    def forward(self, **kwargs):
        return self.model(**kwargs)
    def training_step(self, batch):
        loss = self.model(**batch)['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch):
        loss = self.model(**batch)['loss']
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def eval_batch(self, batch):
        predictions = (self.model(**batch)['output'] >= 0.0).int() # Recall that the model outputs logits
        correct_predictions = (predictions == batch['labels']).int().sum()
        size = predictions.size(0)
        accuracy = correct_predictions.float() / size
        return { "correct_predictions": correct_predictions, "size": size, "accuracy": accuracy }
    def test_step(self, batch):
        eval = self.eval_batch(batch)
        self.log("test_accuracy", eval['accuracy'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return eval['accuracy']
    def configure_optimizers(self): 
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.001,
            total_iters=self.warmup_steps
        )
        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps-self.warmup_steps,
            eta_min=0
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[self.warmup_steps]
        )
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=scheduler,
                interval="step",
                frequency=1
            )
        )

@dataclass
class TrainerConfig:
    batch_size : int
    num_workers : int
    learning_rate : float
    weight_decay : float
    warmup_epochs : float
    num_epochs : int

def trainer(
    tokenizer_cfg : DictConfig,
    encoder_cfg : DictConfig,
    model_cfg : DictConfig,
    dataset_cfg : DictConfig,
    trainer_cfg : DictConfig,
    logdir : str,
    log : Logger,
    state_dict : Optional[str] = None,
):
    log.info("Preparing trainer config")
    trainer_config = TrainerConfig(
        batch_size=trainer_cfg.batch_size,
        num_workers=trainer_cfg.num_workers,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
        warmup_epochs=trainer_cfg.warmup_epochs,
        num_epochs=trainer_cfg.num_epochs
    )

    log.info("Preparing training dataset")
    train_dataset = prepare_dataset(tokenizer_cfg, dataset_cfg, "train", log)
    log.info("Preparing validation dataset")
    val_dataset = prepare_dataset(tokenizer_cfg, dataset_cfg, "val", log)
    log.info("Preparing testing dataset")
    test_dataset = prepare_dataset(tokenizer_cfg, dataset_cfg, "test", log)

    log.info("Preparing training dataloader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_cfg.num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    log.info("Preparing validation dataloader")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_cfg.num_workers,
        shuffle=True,
        collate_fn=val_dataset.collate_fn
    )

    log.info("Preparing testing dataloader")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_cfg.num_workers,
        shuffle=True,
        collate_fn=test_dataset.collate_fn
    )

    log.info("Preparing model")
    model = prepare_model(encoder_cfg, model_cfg, log, state_dict)

    log.info("Preparing lightning model")
    pl_model = PLModel(
        model=model,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        warmup_epochs=trainer_config.warmup_epochs,
        num_epochs=trainer_config.num_epochs,
        epoch_size=len(train_dataloader)
    )

    log.info("Preparing lightning logger")
    logger = TensorBoardLogger(logdir, name=None, version="logs")
    log.info("Preparing lightning learning rate monitor")
    lr_monitor = LearningRateMonitor(logging_interval='step') 
    log.info("Preparing lightning model checkpoint")
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(logdir,"checkpoints"),
        filename='checkpoint_{epoch}_{val_loss}',
        save_top_k=trainer_cfg.save_top_k,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=trainer_cfg.every_n_epochs,
        verbose=True,
        save_weights_only=False,
    )

    log.info("Preparing lightning trainer")
    trainer = L.Trainer(
        max_epochs=trainer_config.num_epochs,
        accelerator="gpu",
        logger=logger,
        callbacks=[lr_monitor, model_checkpoint],
        val_check_interval=trainer_cfg.val_check_interval,
        check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
        limit_val_batches=trainer_cfg.limit_val_batches,
    )

    log.info("Training")
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    log.info("Training complete")
    log.info("Saving model")
    torch.save(pl_model.model.state_dict(), f"{logdir}/model.pth")
    log.info(f"Saved model state dict to {logdir}/model.pth")
    log.info("Testing model")
    eval_output = trainer.test(pl_model, test_dataloader)
    log.info("Testing complete. Logging results")
    for e in eval_output:
        for k, v in e.items():
            log.info(f"{k}: {v}")

