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

    # set up pl trainer (tuner)
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        job_type=f"tune-{cfg.tune_mode}",
        project="claficle",
        entity="giulio-uva",
    )
    trainer = pl.Trainer(
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        auto_scale_batch_size="binsearch" if cfg.tune_mode == "profile_memory" else None,
    )
    model.train_mode = cfg.trainer.train_mode

    # necessary hacks for profiling memory and batch size
    model.batch_size = oscar.cfg.batch_size
    model.collate_fn = oscar.collate_fn
    model.num_workers = oscar.cfg.num_workers
    print(f"Tuning {cfg.tune_mode}...")
    trainer.tune(model)


if __name__ == "__main__":
    main()
