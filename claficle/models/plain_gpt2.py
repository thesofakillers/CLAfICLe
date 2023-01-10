"""
PL wrapper around huggingface transformers gpt2
"""
from typing import Dict

from omegaconf import DictConfig
from torch import Tensor
import torch
from torch.utils.data import DataLoader
import datasets
import wandb

from claficle.models.base import BaseModel


class PlainGPT2(BaseModel):
    """
    PL wrapper around huggingface transformers gpt2
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        shared_step_output = self._shared_step(batch, batch_idx)
        return shared_step_output["loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        shared_step_output = self._shared_step(batch, batch_idx)
        # perplexity is just the exponentiation of cross entropy
        perplexity = torch.exp(shared_step_output["loss"].detach().cpu())
        self.log("val_perplexity", perplexity)

    def _shared_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """
        batch is a dict with keys "input_ids", "attention_mask" and optionally "labels"
        where each key is a tensor of shape (batch_size, seq_len)
        seq_len can vary between batches

        training mode can either be:
        - causal language modelling (clm)
        - MetaICL (meta-icl)
        - vessel (vessel)

        outputs a dict (or equivalent) with keys "loss" and "logits"
        """
        if self.train_mode == "clm":
            return self._clm_step(batch, batch_idx)
        else:
            raise NotImplementedError

    def _clm_step(self, batch: Dict[str, Tensor], batch_idx: int):
        """
        Causal language modelling step
        """
        output = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        return output

    def configure_optimizers(self):
        """placeholder optimizer"""
        return torch.optim.Adam(self.lm.parameters())

    def train_dataloader(self):
        """
        dummy dataloader only used in tune.py
        When used in a trainer with a DataModule
        with its own definition of train_dataloader, this method will be ignored
        """
        # dummy dataset of random ints
        n_datapoints = int(512 * 4)  # 4 batches worth of data
        return self._dummy_dataloader(n_datapoints, shuffle=True)

    def val_dataloader(self):
        """
        Same as train_dataloader but with 0.05 of the data
        """
        n_datapoints = int(512 * 4 * 0.005)  # 0.5 percent of 4 batches worth of data
        return self._dummy_dataloader(n_datapoints, shuffle=False)

    def _dummy_dataloader(self, n_datapoints, shuffle: bool):
        """helper method for train_dataloader and val_dataloader"""
        wandb.alert(
            title="Using dummy dataloader",
            text="This should only happen when running tune.py."
            "If this is not the case, something is wrong.",
            level=wandb.AlertLevel.WARN,
        )
        dataset = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(
                    0,
                    len(self.tokenizer),
                    size=(n_datapoints, self.tokenizer.model_max_length),
                ),
                "attention_mask": torch.ones(
                    (n_datapoints, self.tokenizer.model_max_length), dtype=int
                ),
            }
        )
        return DataLoader(
            dataset,
            shuffle=shuffle,
            pin_memory=True,
            # the following attributes are set after init, specifically for this
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
