import logging
import hydra
from omegaconf import DictConfig
from posttraining.trainer import trainer
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    trainer(
        cfg.common.tokenizer,
        cfg.common.encoder,
        cfg.posttraining.model,
        cfg.posttraining.dataset,
        cfg.posttraining.training,
        HydraConfig.get().run.dir,
        log,
        cfg.posttraining.state_dict
    )

if __name__ == "__main__":
    main()
