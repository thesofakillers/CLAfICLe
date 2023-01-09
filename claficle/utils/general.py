"""General utils"""
from typing import Tuple

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from claficle.models.utils import NAME_TO_CLASS
from claficle.models.base import BaseModel


def run_script_preamble(cfg: DictConfig) -> Tuple[BaseModel, DictConfig]:
    pl.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))
    ModelClass: BaseModel = NAME_TO_CLASS[cfg.model.name]
    print("Loading pretrained model...")
    if cfg.model.pl_checkpoint:
        model = ModelClass.load_from_checkpoint(cfg.model.pl_checkpoint)
    else:
        model = ModelClass(cfg.model)
    # get possible additional preprocessing from model and set in benchmark cfg
    cfg.data.extra_proc_fn = cfg.model.extra_proc_fn
    # overwrite data cfg seed with eval cfg seed
    cfg.data.seed = cfg.seed

    return model, cfg
