import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count
from typing import List, Optional

from torch.utils.data import DataLoader, SequentialSampler
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

    def test_dataloader(self) -> List[DataLoader]:
        """Returns a test dataloader for each processed dataset"""
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=SequentialSampler(dataset),
                collate_fn=lambda batch: batch,  # convert batch to list of dicts
            )
            for dataset in self._processed_datasets
        ]

    def _load_raw_dataset(self, dataset_name):
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

    def _download_dataset(self, dataset_name):
        """Downloads huggingface dataset"""
        self._load_raw_dataset(dataset_name)
        return


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
