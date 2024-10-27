import logging
import hydra
from omegaconf import DictConfig
from pretraining.trainer import trainer
from hydra.core.hydra_config import HydraConfig
import shutil

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    output_path = trainer(
        cfg.common.tokenizer,
        cfg.common.encoder,
        cfg.pretraining.dataset,
        cfg.pretraining.training,
        HydraConfig.get().run.dir,
        log
    )
    shutil.copy(output_path, "encoder_state_dict.pth")

if __name__ == "__main__":
    main()
