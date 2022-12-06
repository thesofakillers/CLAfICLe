import os
from pathlib import Path
import json
from typing import Any, Dict, Generator, Optional

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import datasets
from tqdm import tqdm


class OSCARDataModule(pl.LightningDataModule):
    """
    PL DataModule responsible for OSCAR dataloading
    """

    def __init__(self, config: DictConfig, lang, seed):
        super().__init__()
        self.cfg = config
        self.is_setup = False
        self.save_dir = os.path.join(
            self.cfg.data_dir, "oscar", lang, str(self.cfg.sample_size_mb)
        )
        self.lang = lang
        pl.seed_everything(seed)

    def prepare_data(self):
        """Take care of downloading data"""
        if self.is_setup:
            return
        # if already saved, don't need to download
        if os.path.exists(self.save_dir):
            return
        # otherwise, download and save
        else:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            if self.cfg.sample_size_mb is None:
                dataset: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                    "oscar", f"unshuffled_deduplicated_{self.lang}", split="train"
                )
                # manually split into train and test and save to disk
                dataset = dataset.train_test_split(test_size=self.cfg.val_percent)
                dataset.save_to_disk(self.save_dir)
            else:
                subsample_size = self.cfg.sample_size_mb * 1024 * 1024
                dataset: datasets.iterable_dataset.IterableDataset = (
                    datasets.load_dataset(
                        "oscar",
                        f"unshuffled_deduplicated_{self.lang}",
                        split="train",
                        streaming=True,
                    )
                )
                dataset_iter = iter(dataset)
                # save subsample_size bytes of data as train
                datastream_to_file(
                    dataset_iter, self.save_dir, "train.json", subsample_size
                )
                # save remaining subsample_size * val_percent bytes of data as val
                datastream_to_file(
                    dataset_iter,
                    self.save_dir,
                    "validation.json",
                    subsample_size * self.cfg.val_percent,
                )

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.load_dataset(
                self.save_dir, split="train", cache_dir=self.save_dir
            )
        if stage == "validate" or stage is None:
            self.val_dataset = datasets.load_dataset(
                self.save_dir, split="validation", cache_dir=self.save_dir
            )
        self.is_setup = True


def datastream_to_file(
    data_stream: Generator[Dict, Any, Any],
    save_dir: str,
    file_name: str,
    total_size_bytes: int,
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        size = 0
        bar = tqdm(total=total_size_bytes)

        while size < total_size_bytes:
            entry = next(data_stream)

            entry_size = len(entry["text"].encode("utf-8"))
            size += entry_size

            bar.update(entry_size)

            f.write(f"{json.dumps(entry)}\n")


@hydra.main(version_base=None, config_path="../conf/data/", config_name="oscar_base")
def main(cfg: DictConfig):
    """downloads and processes the data for benchmark for each of the available languages"""
    print(cfg)
    oscar = OSCARDataModule(cfg, "fr", 1)
    oscar.prepare_data()
    oscar.setup()


if __name__ == "__main__":
    main()
