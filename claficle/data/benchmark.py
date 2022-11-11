import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Dict

from torch.utils.data import DataLoader, SequentialSampler
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import datasets

from claficle.data.process import process_dataset


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, lang: str):
        super().__init__()
        self.cfg = config
        self.lang = lang
        self.raw_save_path: str = os.path.join(self.cfg.data_dir, "raw")
        self._metadata = {"lang": self.lang, "datasets": []}
        self._pre_collate_fn: Callable[
            [List[Dict]], List[Dict]
        ] = lambda batch: batch  # default no-op (can be set)

    def prepare_data(self):
        """takes care of downloading data"""
        print("Loading datasets...")
        # make use of multithreading to speed up by downloading in parallel
        thread_pool = ThreadPool(cpu_count())
        thread_pool.map(self._download_dataset, self.cfg.dataset_names)

    def setup(self, stage: Optional[str] = None):
        """
        processes each dataset, obtaining test split and relevant metric(s)
        """
        print("Processing datasets...")
        self._processed_datasets = []
        for dataset_name in self.cfg.dataset_names:
            dataset = self._load_raw_dataset(dataset_name)
            test_dataset, metrics, collection_name = process_dataset(
                dataset, self.lang, self.cfg, dataset_name
            )
            # test_dataset is None if dataset is not available in language
            if test_dataset is not None:
                # map dataset idx to name & metrics, so to track in LightningModule
                self._metadata["datasets"].append(
                    {"name": collection_name, "metrics": metrics}
                )
                self._processed_datasets.append(test_dataset)
        print("Done.")

    def get_metadata(self):
        return self._metadata

    def collate_fn(self, batch: List[Dict]):
        """Converts batch to list of dicts"""
        # apply any pre-collation processing first
        proc_batch: List[Dict] = self._pre_collate_fn(batch)

        encodings = []
        labels = []
        for item in proc_batch:
            # TODO: for each option, need to encode input + option directly
            # keeping track of which token ids are the completion and which aren't
            # note that tokenizer can handle batching and padding
            # note that we need to add separator either at end of input or at beginning
            # of each option
            encoding = self.tokenizer(item["text"])
            encodings.append(encoding)
            labels = labels.append(item["label"])
        labels = torch.LongTensor(labels)
        return batch

    def test_dataloader(self) -> List[DataLoader]:
        """Returns a test dataloader for each processed dataset"""
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=SequentialSampler(dataset),
                collate_fn=self.collate_fn,
            )
            for dataset in self._processed_datasets
        ]

    def _load_raw_dataset(self, dataset_name: str):
        # parses dataset name
        if ";" in dataset_name:
            collection: str
            subcollection: Optional[str]
            collection, subcollection = dataset_name.split(";")
        else:
            collection = dataset_name
            subcollection = None
        dataset_path = os.path.join(
            self.raw_save_path,
            collection,
            subcollection if subcollection is not None else "",
        )
        if os.path.exists(dataset_path):
            dataset = datasets.load_from_disk(dataset_path)
        else:
            dataset = datasets.load_dataset(collection, subcollection)
            # create save directory if it doesn't exist, and save to disk
            Path(dataset_path).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(dataset_path)
        return dataset

    def _download_dataset(self, dataset_name: str):
        """Downloads huggingface dataset"""
        self._load_raw_dataset(dataset_name)
        return

    def set_pre_collate_fn(self, pre_collate_fn: Callable[[List[Dict]], List[Dict]]):
        """
        Sets a pre-collate processing function, which is applied to each batch
        before collation
        """
        self._pre_collate_fn = pre_collate_fn

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


if __name__ == "__main__":
    # testing
    from pprint import pprint
    import yaml
    from omegaconf import OmegaConf

    with open("claficle/conf/benchmark/eval.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    cfg: DictConfig = OmegaConf.create(config)

    for lang in ["en", "fr", "de"]:
        benchmark = BenchmarkDataModule(cfg, lang)
        benchmark.prepare_data()
        benchmark.setup()
        pprint(benchmark.get_metadata())
