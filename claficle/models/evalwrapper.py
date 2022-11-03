import pytorch_lightning as pl
import torch
from transformers import AutoModelForCausalLM
from omegaconf import DictConfig


class EvalWrapper(pl.LightningModule):
    # todo
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def load_checkpoint(self, checkpoint_path):
        # todo
        state_dict = torch.load(checkpoint_path)
        self.causal_lm = AutoModelForCausalLM.from_pretrained(
            self.cfg.causalLM_variant, state_dict=state_dict
        )

    def configure_approach(self):
        # todo
        pass

    def forward(self, x):
        # todo
        pass

    def test_step(self, batch, batch_idx, dataloader_idx):
        # todo
        pass

    def set_benchmark_metadata(self, metadata):
        """Contains info on the name and metrics for each dataloader in the benchmark"""
        self.benchmark_metadata = metadata
