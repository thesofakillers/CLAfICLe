"""Distil a model into a vessel adapter"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import transformers
import pytorch_lightning as pl

from claficle.data.oscar import OSCARDataModule
from claficle.utils.general import run_script_preamble
from claficle.models.vessel import Vessel


@hydra.main(version_base=None, config_path="../conf", config_name="distil")
def main(cfg: DictConfig):
    # setting seed and initializing model
    model: Vessel
    model, cfg = run_script_preamble(cfg)

    # additional post initialization (activating adapter, freezing gpt2)
    model.post_init(seed=cfg.seed)

    # we are only doing vessel distillation in english
    lang = "en"

    # data
    oscar = OSCARDataModule(config=cfg.data, lang=lang, seed=cfg.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.causalLM_variant)
    oscar.prepare_data()
    oscar.set_tokenizer(tokenizer)
    oscar.setup("distillation")

    # trainer
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", "distillation", lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        entity="giulio-uva",
        project="claficle",
        job_type="distillation",
        mode="disabled" if cfg.trainer.disable_wandb else "online",
        group=script_host,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        log_model=False,  # don't log or upload artifacts
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # save best checkpoints
        dirpath=cfg.model.checkpoint_dir,
        filename=f"adapter_distillation_-v{cfg.seed}",
        monitor=f"{cfg.trainer.train_mode}/val/perplexity",
        mode="min",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=f"{cfg.trainer.train_mode}/val/perplexity", patience=4, mode="min"
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.trainer.clip_grad_norm,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,  # log every batch
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback],
        precision=16,
        deterministic=True,
    )
    model.train_mode = cfg.trainer.train_mode

    trainer.fit(model, oscar)


if __name__ == "__main__":
    main()
