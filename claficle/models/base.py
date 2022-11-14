from typing import Dict, List, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer
from torch import Tensor


class BaseModel(pl.LightningModule):
    # todo
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        # TODO: get log-lklhood from evaluand for each option completing the input
        # TODO: evaluate (max log-likelihood = prediction -> compare to label)
        # TODO: log
        pass

    def set_benchmark_metadata(self, metadata):
        """
        Contains info on the name and metrics for each dataloader in the benchmark
        """
        self.benchmark_metadata = metadata

    @staticmethod
    def pre_collate(batch: List[Dict]) -> List[Dict]:
        """
        Optional pre-collation processing to be passed to a dataloader
        By default we do nothing
        """
        return batch
