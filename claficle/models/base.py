from typing import Dict, List, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from claficle.models.utils import NAME_TO_CLASS


class BaseModel(pl.LightningModule):
    # todo
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self._set_evaluand()

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

    def prep_batch(self, batch: List[Dict]):
        proc_batch = self.pre_collate_proc(batch)
        coll_batch = self.collate_fn(proc_batch)
        return coll_batch

    def pre_collate_proc(self, batch: List[Dict]) -> List[Dict]:
        """
        Preprocesses batch before collation (e.g. translation)
        By default we do nothing
        """
        return batch

    def collate_fn(self, batch: List[Dict]):
        encodings = []
        labels = []
        for item in batch:
            # TODO: for each option, need to encode input + option directly
            # keeping track of which token ids are the completion and which aren't
            # note that tokenize can handle batching
            # note that we need to init tokenizer
            # note that we need to add separator either at end of input or at beginning
            # of each option
            encoding = self.tokenize(item["text"])
            encodings.append(encoding)
            labels = labels.append(item["label"])
        labels = torch.LongTensor(labels)
        pass
