"""
A Gewechselt model: A model to which WECHSEL is applied
"""
from typing import Dict
import os

from omegaconf import DictConfig
from wechsel import WECHSEL, load_embeddings
from transformers import AutoTokenizer
import torch
import hydra
import datasets
import transformers
import numpy as np

from claficle.models.plain_gpt2 import PlainGPT2

langcode_to_lang: Dict[str, str] = {
    "en": "english",
    "de": "german",
    "fr": "french",
}


class Gewechselt(PlainGPT2):
    """
    GPT2 Model initialized using WECHSEL (Minixhofer et al. 2022)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def post_init(self, target_data: datasets.arrow_dataset.Dataset) -> AutoTokenizer:
        """Applies WECHSEL initialization. Returns the trained tokenizer"""
        print("Training target tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.causalLM_variant)
        target_tokenizer = tokenizer.train_new_from_iterator(
            target_data["text"], vocab_size=len(tokenizer), length=len(target_data)
        )
        print("Initializing WECHSEL...")
        wechsel = WECHSEL(
            load_embeddings(self.hparams.source_lang),
            load_embeddings(self.hparams.target_lang),
            bilingual_dictionary=langcode_to_lang[self.hparams.target_lang],
        )
        print("Generating target embeddings...")
        target_embeddings, info = wechsel.apply(
            tokenizer,
            target_tokenizer,
            self.lm.get_input_embeddings().weight.detach().numpy(),
        )
        print("Replacing source embeddings with target embeddings...")
        self.lm.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

        print("Done.")
        return target_tokenizer

    def configure_optimizers(self):
        """
        Adam with Cosine annealing to 0 by end of training with warmup
        peak learning rate of 5e-4
        """
        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.Adam(self.lm.parameters(), self.hparams.peak_lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # scheduler is called after each step
            },
        }


@hydra.main(version_base=None, config_path="../conf", config_name="wechsel_init")
def main(cfg: DictConfig):
    """
    Handles WECHSEL initialization, which can be a length process on its own as it
    onvolves the training of a tokenizer, among other computations.
    To avoid wasting training time, we run initialization separately and serialize
    the results, through this method
    """
    import pytorch_lightning as pl
    import wandb
    from claficle.data.oscar import OSCARDataModule
    from omegaconf import OmegaConf

    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    # we'll need a Trainer instance to save a pl checkpoint, this needs a logger
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
        enable_checkpointing=False,  # we handle this manually
    )

    oscar = OSCARDataModule(config=cfg.data, lang=cfg.model.target_lang, seed=cfg.seed)
    oscar.prepare_data()

    # this will take a while
    model: Gewechselt = Gewechselt(cfg.model)
    target_tokenizer = model.post_init(oscar.train_dataset)
    # save the tokenizer locally
    tokenizer_save_dir = os.path.join("checkpoints", "tokenizers")
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_save_dir, cfg.tokenizer_name)
    target_tokenizer.save_pretrained(tokenizer_path)
    # and upload it to wandb
    artifact = wandb.Artifact(
        name=cfg.tokenizer_name,
        type="tokenizer",
    )
    artifact.add_dir(tokenizer_path)
    wandb.log_artifact(artifact)

    # just so that we can save a PL checkpoint of the model
    trainer.predict(
        model,
        dataloaders=torch.utils.data.DataLoader(
            datasets.Dataset.from_list(
                [
                    {
                        "input_ids": torch.randint(0, int(5e4), size=(1024,)),
                        "attention_mask": torch.ones(1024, dtype=int),
                    }
                ]
            ),
            batch_size=1,
            collate_fn=oscar.collate_fn,
            num_workers=cfg.data.num_workers,
        ),
        return_predictions=False,
    )

    # save the checkpoint locally
    prefix = (
        cfg.model.base_checkpoint.split(".")[0]
        if cfg.model.base_checkpoint is not None
        else cfg.model.causalLM_variant
    )
    model_name = f"{prefix}_{cfg.model.name}_{cfg.model.target_lang}.ckpt"
    checkpoint_path = os.path.join("checkpoints", model_name)
    trainer.save_checkpoint(checkpoint_path)
    # and upload it to wandb
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        metadata=OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True),
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact, aliases=["init"])


if __name__ == "__main__":
    main()
