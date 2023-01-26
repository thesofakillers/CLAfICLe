"""Frozen GPT2 + Unfrozen Adapter, to enable distillation into Adapter"""
from typing import Dict, Any
import os

from omegaconf import DictConfig

from claficle.models.plain_gpt2 import PlainGPT2


class Vessel(PlainGPT2):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def post_init(self, seed: int):
        """
        Either initializes or loads the adapter weights
        Depending on whether the file at self.hparams.adapter_checkpoint exists
        """
        self.adapter_path = os.path.join(
            self.hparams.checkpoint_dir,
            f"{self.hparams.adapter_checkpoint}-seed_{seed}",
        )
        # checkpoint loading branch
        if os.path.exists(self.adapter_path):
            self.lm.load_adapter(self.adapter_path, config="pfeiffer", set_active=True)
        # initialization branch
        else:
            self.lm.add_adapter(self.hparams.adapter_name, config="pfeiffer")

    def on_train_start(self):
        # freeze the GPT2 weights and only train the adapter
        self.lm.train_adapter(self.hparams.adapter_name)
        self.lm.set_active_adapters(self.hparams.adapter_name)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        - Calls save_adapter and saves the adapter
        - Removes the state dict from the PL checkpoint
        - This is not the intended use of on_save_checkpoint,
          but i have little options
        - We don't need to modify on_load_checkpoint since we will
          not be using it to load weights anyways.
        """
        self.lm.save_adapter(self.adapter_path, self.hparams.adapter_name)
        del checkpoint["state_dict"]
