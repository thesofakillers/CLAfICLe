"""Train our models"""
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import transformers
import pytorch_lightning as pl

from claficle.models.base import BaseModel
from claficle.utils.general import run_script_preamble
from claficle.data.oscar import OSCARDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if cfg.model.pl_checkpoint is None:
        raise ValueError("Must provide a (init) PL checkpoint to train")
    # sets seed, parses model name
    model: BaseModel
    model, cfg = run_script_preamble(cfg)

    # data
    oscar = OSCARDataModule(config=cfg.data, lang=cfg.model.target_lang, seed=cfg.seed)
    if cfg.tokenizer_name is None:
        if cfg.model.target_lang != "en":
            raise ValueError("Must provide a tokenizer path for non-English models")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model.causalLM_variant
        )
    else:
        tokenizer_path = os.path.join("checkpoints", "tokenizers", cfg.tokenizer_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    oscar.prepare_data()
    oscar.set_tokenizer(tokenizer)
    oscar.setup()

    # trainer
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", model.cfg.target_lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        entity="giulio-uva",
        project="claficle",
        job_type="train",
        mode="disabled" if cfg.trainer.disable_wandb else "online",
        group=script_host,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        log_model="all",  # log models to wandb during training rather than at the end
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=cfg.model.pl_checkpoint,
        monitor=f"{cfg.trainer.train_mode}/val/perplexity",
        mode="min",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
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
        log_every_n_steps=cfg.trainer.val_check_interval,
        callbacks=[checkpoint_callback],
        precision=16
    )
    model.train_mode = cfg.trainer.train_mode

    trainer.fit(model, oscar)
