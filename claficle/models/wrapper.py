import pytorch_lightning as pl
import torch
from transformers import AutoModelForCausalLM


class Wrapper(pl.LightningModule):
    # todo
    def __init__(self, causalLM_variant: str, approach: str):
        super().__init__()
        self.causalLM_variant = causalLM_variant
        self.approach = approach

    def load_checkpoint(self, checkpoint_path):
        # todo
        state_dict = torch.load(checkpoint_path)
        self.causal_lm = AutoModelForCausalLM.from_pretrained(
            self.causalLM_variant, state_dict=state_dict
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
        # todo
        pass
