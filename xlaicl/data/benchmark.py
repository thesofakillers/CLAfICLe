import os
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count

from typing import List, Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
import datasets


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.raw_save_path: str = os.path.join(self.config.data_dir, "raw")

    def prepare_data(self):
        """takes care of downloading data"""
        # make use of multithreading to speed up by downloading in parallel
        ThreadPool(cpu_count()).map(self._download_dataset, self.config.dataset_names)

    def setup(self, stage: Optional[str] = None):
        pass

    def get_metadata(self):
        pass

    def test_dataloader(self) -> List[DataLoader]:
        pass

    def _download_dataset(self, dataset_name):
        """Downloads huggingface dataset"""
        # parses dataset name
        if ";" in dataset_name:
            collection: str
            subcollection: Optional[str]
            collection, subcollection = dataset_name.split(";")
        else:
            collection = dataset_name
            subcollection = None
        # proceeds with download
        print(f"Downloading {dataset_name}")
        datasets.load_dataset(collection, subcollection, cache_dir=self.raw_save_path)
        return


if __name__ == "__main__":
    # testing
    import yaml
    from omegaconf import OmegaConf

    with open("xlaicl/conf/benchmark/test.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    cfg: DictConfig = OmegaConf.create(config)

    benchmark = BenchmarkDataModule(cfg)

    benchmark.prepare_data()
