import os
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Iterable, List, Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
import datasets

import xlaicl.data.process as process


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, lang: str):
        super().__init__()
        self.config = config
        self.lang = lang
        self.raw_save_path: str = os.path.join(self.config.data_dir, "raw")
        self._process_by_name: Dict[str, Callable[[Any, str], Iterable]] = {
            "xglue;qam": process.xglue
        }

    def prepare_data(self):
        """takes care of downloading data"""
        # make use of multithreading to speed up by downloading in parallel
        thread_pool = ThreadPool(cpu_count())
        thread_pool.map(self._download_dataset, self.config.dataset_names)

    def setup(self, stage: Optional[str] = None):
        """
        Gets relevant test split
        Generates k-shot context
        prepend each input with k-shot context
        add options column to track options
        keep track of metadata throughout
        """
        for dataset_name in self.config.dataset_names:
            dataset = self._load_dataset(dataset_name)
            self._process_by_name[dataset_name](dataset, self.lang)

    def get_metadata(self):
        # TODO
        pass

    def test_dataloader(self) -> List[DataLoader]:
        # TODO
        return []

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

    with open("xlaicl/conf/benchmark/test.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    cfg: DictConfig = OmegaConf.create(config)

    benchmark = BenchmarkDataModule(cfg, "en")

    benchmark.prepare_data()
