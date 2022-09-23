import os
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count
from typing import List, Optional

from torch.utils.data import DataLoader, SequentialSampler
import pytorch_lightning as pl
from omegaconf import DictConfig
import datasets

from claficle.data.process import helper_by_name, process_dataset


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, lang: str):
        super().__init__()
        self.cfg = config
        self.lang = lang
        self.raw_save_path: str = os.path.join(self.cfg.data_dir, "raw")
        self._metadata = {"lang": self.lang, "datasets": []}

    def prepare_data(self):
        """takes care of downloading data"""
        # make use of multithreading to speed up by downloading in parallel
        thread_pool = ThreadPool(cpu_count())
        thread_pool.map(self._download_dataset, self.cfg.dataset_names)

    def setup(self, stage: Optional[str] = None):
        """
        processes each dataset, obtaining test split and relevant metric(s)
        """
        self._processed_datasets = []
        for dataset_name in self.cfg.dataset_names:
            dataset = self._load_dataset(dataset_name)
            test_dataset, metrics = process_dataset(
                dataset, self.lang, self.cfg, helper_by_name[dataset_name]
            )
            # map dataset idx to name & metrics, so we can track in LightningModule
            self._metadata["datasets"].append(
                {"name": dataset_name, "metrics": metrics}
            )
            self._processed_datasets.append(test_dataset)
        # sanity check, failing it should raise some red flags
        assert (
            len(self.cfg.dataset_names)
            == len(self._processed_datasets)
            == len(self._metadata["datasets"])
        ), "Mismatch in number of requested, processed, and tracked datasets"

    def get_metadata(self):
        return self._metadata

    def test_dataloader(self) -> List[DataLoader]:
        """Returns a test dataloader for each processed dataset"""
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=SequentialSampler(dataset),
            )
            for dataset in self._processed_datasets
        ]

    def _load_dataset(self, dataset_name):
        # parses dataset name
        if ";" in dataset_name:
            collection: str
            subcollection: Optional[str]
            collection, subcollection = dataset_name.split(";")
        else:
            collection = dataset_name
            subcollection = None
        return datasets.load_dataset(
            collection, subcollection, cache_dir=self.raw_save_path
        )

    def _download_dataset(self, dataset_name):
        """Downloads huggingface dataset"""
        print(f"Downloading {dataset_name}")
        self._load_dataset(dataset_name)
        return


if __name__ == "__main__":
    # testing
    import yaml
    from omegaconf import OmegaConf

    with open("claficle/conf/benchmark/eval.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    cfg: DictConfig = OmegaConf.create(config)

    benchmark = BenchmarkDataModule(cfg, "en")

    benchmark.prepare_data()