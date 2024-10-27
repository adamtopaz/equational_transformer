from logging import Logger
from omegaconf import DictConfig
from dataclasses import dataclass

from common.encoder import prepare_encoder, TransformerEncoder
from pretraining.dataset import prepare_dataset
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerConfigType
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
import os

class PLModel(L.LightningModule):
    def __init__(
        self, 
        encoder : TransformerEncoder,
        learning_rate : float,
        weight_decay : float,
        warmup_epochs : float,
        num_epochs : int, 
        epoch_size : int,
    ):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epoch_size = epoch_size
        self.num_epochs = num_epochs
        self.total_steps = self.epoch_size * self.num_epochs
        self.warmup_steps = int(self.warmup_epochs * self.epoch_size)
    def forward(self, **kwargs):
        return self.encoder(**kwargs)
    def training_step(self, batch):
        loss = self.encoder(**batch)['loss']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
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
    dataset_cfg : DictConfig,
    trainer_cfg : DictConfig,
    logdir : str, 
    log : Logger
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

    log.info("Preparing dataset")
    dataset = prepare_dataset(tokenizer_cfg, dataset_cfg, log)

    log.info("Preparing dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_cfg.num_workers,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    log.info("Preparing encoder")
    encoder = prepare_encoder(encoder_cfg, log)

    log.info("Preparing lightning model")
    pl_model = PLModel(
        encoder = encoder,
        learning_rate = trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
        warmup_epochs = trainer_config.warmup_epochs,
        num_epochs = trainer_config.num_epochs,
        epoch_size = len(dataloader)
    )

    log.info("Preparing lightning logger")
    logger = TensorBoardLogger(logdir, name=None, version="logs")
    log.info("Preparing lightning learning rate monitor")
    lr_monitor = LearningRateMonitor(logging_interval='step') 

    log.info("Preparing lightning trainer")
    trainer = L.Trainer(
        max_epochs=trainer_config.num_epochs,
        accelerator="gpu",
        logger=logger,
        callbacks=[lr_monitor],
    )

    log.info("Training")
    trainer.fit(pl_model, dataloader)
    log.info("Training complete")
    log.info("Saving model")
    output_path = os.path.join(logdir,"encoder_state_dict.pth")
    torch.save(pl_model.encoder.state_dict(), output_path)
    log.info(f"Saved encoder state dict to {output_path}")
    return output_path

