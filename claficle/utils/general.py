"""General utils"""
import os
from typing import Tuple, List, TypeVar
import itertools

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from claficle.models.utils import NAME_TO_CLASS, get_model_preamble_post_init_kwargs
from claficle.models.base import BaseModel


def run_script_preamble(cfg: DictConfig) -> Tuple[BaseModel, DictConfig]:
    """
    sets the seed
    parses the model name and initializes/loads the model
    modifies the cfg accordingly
    """
    pl.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))
    ModelClass: BaseModel = NAME_TO_CLASS[cfg.model.name]
    print("Loading pretrained model...")
    if cfg.model.pl_checkpoint:
        model = ModelClass.load_from_checkpoint(
            os.path.join(cfg.model.checkpoint_dir, cfg.model.pl_checkpoint)
        )
    else:
        model = ModelClass(cfg.model)  # this loads non-pl checkpoints internally
    # get possible additional preprocessing from model and set in data cfg
    cfg.data.extra_proc_fn = cfg.model.extra_proc_fn
    # overwrite data cfg seed with run script seed
    cfg.data.seed = cfg.seed
    # if requested, run model.post_init(**kwargs)
    if cfg.model.preamble_post_init:
        model.post_init(**get_model_preamble_post_init_kwargs(cfg))
    print("Done.")
    return model, cfg


# used below
T = TypeVar("T")


def flatten_list_with_separator(
    unflattened_list: List[List[T]], separator: T
) -> List[T]:
    """
    Flattens a list of lists, inserting a separator between each list
    """
    return list(
        itertools.chain.from_iterable(
            (sublist + [separator]) for sublist in unflattened_list
        )
    )[:-1]
