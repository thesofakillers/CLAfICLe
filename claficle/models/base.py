from typing import Dict, List
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer


class BaseModel(pl.LightningModule):
    # todo
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_step(self, batch: List[Dict], batch_idx: int, dataloader_idx: int):
        batch = self.prep_batch(batch)
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
