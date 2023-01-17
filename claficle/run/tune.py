"""Helper script for determining the best hyperparameters for a model."""
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import transformers

from claficle.models.base import BaseModel
from claficle.utils.general import run_script_preamble
from claficle.data.oscar import OSCARDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig):
    # sets seed, parses model name
    model: BaseModel
    model, cfg = run_script_preamble(cfg)

    # hardcode the language to english. Memory and time will be the roughly same for any language
    lang = "en"

    # data
    oscar = OSCARDataModule(config=cfg.data, lang=lang, seed=cfg.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.causalLM_variant)
    oscar.set_tokenizer(tokenizer)  # necessary for collate_fn

    # set up pl trainer (tuner)
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        job_type=f"tune-{cfg.tune_mode}",
        project="claficle",
        entity="giulio-uva",
        mode="disabled" if cfg.trainer.disable_wandb else "online",
        group=script_host,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    timer = Timer(interval="epoch")
    if cfg.tune_mode == "profile_memory":
        trainer_kwargs = {
            "auto_scale_batch_size": "binsearch",
        }
    elif cfg.tune_mode == "profile_time":
        trainer_kwargs = {
            "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
            "val_check_interval": cfg.trainer.val_check_interval,
            "log_every_n_steps": cfg.trainer.val_check_interval,
            "callbacks": [timer],
        }
    else:
        raise ValueError(f"Unknown tune_mode: {cfg.tune_mode}")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        enable_checkpointing=False,
        precision=16,
        **trainer_kwargs,
    )
    model.train_mode = cfg.trainer.train_mode

    # necessary hacks for train and val loaders in model
    model.batch_size = oscar.cfg.batch_size
    model.collate_fn = oscar.collate_fn
    model.num_workers = oscar.cfg.num_workers
    if cfg.tune_mode == "profile_memory":
        print(f"Running {cfg.tune_mode}...")
        trainer.tune(model)
    elif cfg.tune_mode == "profile_time":
        print(f"Running {cfg.tune_mode}...")
        trainer.fit(model)

        wandb.log({"train_starttime_secs": timer.start_time("train")})
        wandb.log({"train_endtime_secs": timer.end_time("train")})
        wandb.log({"train_elapsed_secs": timer.time_elapsed("train")})

        wandb.log({"val_starttime_secs": timer.start_time("validate")})
        wandb.log({"val_endtime_secs": timer.end_time("validate")})
        wandb.log({"val_elapsed_secs": timer.time_elapsed("validate")})


if __name__ == "__main__":
    main()
