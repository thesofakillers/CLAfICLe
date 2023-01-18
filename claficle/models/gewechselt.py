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
import wandb

from claficle.models.plain_gpt2 import PlainGPT2
from claficle.utils.general import yield_batches_from_stream

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

    def post_init(
        self,
        target_data: datasets.IterableDataset,
        target_data_len: int,
        tokenizer_name: str,
    ):
        """
        Applies WECHSEL initialization.
        Serializes the trained tokenizer if it is not already present.
        """
        targ_tok_save_dir = os.path.join(self.hparams.checkpoint_dir, "tokenizers")
        os.makedirs(targ_tok_save_dir, exist_ok=True)
        target_tok_path = os.path.join(targ_tok_save_dir, tokenizer_name)

        source_tokenizer = AutoTokenizer.from_pretrained(self.hparams.causalLM_variant)

        if os.path.exists(target_tok_path):
            target_tokenizer = AutoTokenizer.from_pretrained(target_tok_path)
        else:
            print("Training target tokenizer...")
            target_tokenizer = source_tokenizer.train_new_from_iterator(
                yield_batches_from_stream(target_data.take(target_data_len), "text"),
                vocab_size=len(source_tokenizer),
                length=target_data_len,
            )
            # serializing and uploading to wandb
            target_tokenizer.save_pretrained(target_tok_path)
            artifact = wandb.Artifact(
                name=tokenizer_name,
                type="tokenizer",
            )
            artifact.add_dir(target_tok_path)
            wandb.log_artifact(artifact)

        print("Initializing WECHSEL...")
        wechsel = WECHSEL(
            load_embeddings(self.hparams.source_lang),
            load_embeddings(self.hparams.target_lang),
            bilingual_dictionary=langcode_to_lang[self.hparams.target_lang],
        )

        print("Generating target embeddings...")
        target_embeddings, info = wechsel.apply(
            source_tokenizer,
            target_tokenizer,
            self.lm.get_input_embeddings().weight.detach().numpy(),
        )

        print("Replacing source embeddings with target embeddings...")
        self.lm.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

        print("Done.")

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
    datasets.disable_caching()

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

    model: Gewechselt = Gewechselt(cfg.model)
    # this will take a while
    model.post_init(
        oscar.raw_dataset, int(2.4e6), cfg.tokenizer_name, cfg.model.checkpoint_dir
    )

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
    checkpoint_path = os.path.join(cfg.model.checkpoint_dir, model_name)
    trainer.save_checkpoint(checkpoint_path)
    # and upload it to wandb
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        metadata=OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True),
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact, aliases=["init", f"seed_{cfg.seed}"])


if __name__ == "__main__":
    main()
