"""
PL wrapper around huggingface transformers gpt2
"""
from typing import Dict

from omegaconf import DictConfig
from torch import Tensor
import torch

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
        loss = output.loss
        return loss

    def configure_optimizers(self):
        """placeholder optimizer"""
        return torch.optim.Adam(self.lm.parameters())
