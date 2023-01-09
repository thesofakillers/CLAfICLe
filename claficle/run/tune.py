"""Helper script for determining the best hyperparameters for a model."""
import os

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from claficle.models.base import BaseModel
from claficle.utils.general import run_script_preamble
from claficle.data.oscar import OSCARDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig):
    # sets seed, parses model name
    model: BaseModel
    model, cfg = run_script_preamble(cfg)

    # hardcode the language. Memory and time will be the roughly same for any language
    lang = "fr"

    # data
    oscar = OSCARDataModule(config=cfg.data, lang=lang, seed=cfg.seed)
    oscar.set_tokenizer(model.tokenizer)
    oscar.prepare_data()
    oscar.setup(stage=cfg.tune_mode)
    dataloader_func = {
        "profile_memory": oscar.memory_profile_dataloader,
        "profile_time": oscar.time_profile_dataloader,
    }[cfg.tune_mode]

    # set up pl trainer (tuner)
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_save_dir,
        name=f"tune-{cfg.tune_mode}",
    )
    trainer = pl.Trainer(
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        auto_scale_batch_size="power" if cfg.tune_mode == "profile_memory" else None,
    )

    print(f"Tuning {cfg.tune_mode}...")
    trainer.tune(
        model,
        train_dataloaders=dataloader_func("train"),
        val_dataloaders=dataloader_func("val"),
    )


if __name__ == "__main__":
    main()
