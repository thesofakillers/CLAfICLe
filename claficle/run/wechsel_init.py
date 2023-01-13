"""
WECHSEL initializations involve the training of a tokenizer, and can therefore
be a lengthy process on their own. This script separates that process
"""
import os

from omegaconf import DictConfig, OmegaConf
import transformers
import pytorch_lightning as pl
import torch
import hydra

from claficle.data.oscar import OSCARDataModule
from claficle.models.gewechselt import Gewechselt


@hydra.main(version_base=None, config_path="../conf", config_name="wechsel_init")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # we'll need a Trainer instance to save a checkpoint
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", cfg.model.target_lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        job_type="wechsel_init",
        project="claficle",
        entity="giulio-uva",
        mode="disabled" if cfg.trainer.disable_wandb else "online",
        group=script_host,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.causalLM_variant)
    tokenizer.pad_token = tokenizer.eos_token

    oscar = OSCARDataModule(config=cfg.data, lang=cfg.model.target_lang, seed=cfg.seed)
    oscar.set_tokenizer(tokenizer)
    oscar.prepare_data()
    oscar.setup()

    # this will take a while
    model: Gewechselt = Gewechselt(cfg.model, oscar.train_dataset)

    # just so that we can save a PL checkpoint of the model
    trainer.predict(
        model,
        dataloaders=torch.utils.data.DataLoader(
            oscar.val_dataset_tokens.select([1]),
            batch_size=1,
            collate_fn=oscar.collate_fn,
        ),
        return_predictions=False,
    )

    # save the checkpoint
    prefix = (
        model.base_checkpoint
        if model.base_checkpoint is not None
        else model.causalLM_variant
    )
    trainer.save_checkpoint(
        "checkpoints/" f"{prefix}_{model.name}_{model.target_lang}_init.ckpt"
    )


if __name__ == "__main__":
    main()
