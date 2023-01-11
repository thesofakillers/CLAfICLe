import os
from pathlib import Path
import json
from typing import Any, Dict, Generator, Optional, List

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import datasets
import transformers
from tqdm import tqdm
import torch

from claficle.utils.general import flatten_list_with_separator


class OSCARDataModule(pl.LightningDataModule):
    """
    PL DataModule responsible for OSCAR dataloading

    Note: run `set_tokenizer(tokenizer)` before running self.setup()
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
                    "oscar-corpus/OSCAR-2201",
                    f"unshuffled_deduplicated_{self.lang}",
                    split="train",
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
                print(f"Downloading {self.cfg.sample_size_mb}MB of train data")
                datastream_to_file(
                    dataset_iter, self.save_dir, "train.json", subsample_size
                )
                # save remaining subsample_size * val_percent bytes of data as val
                print(
                    f"Downloading {self.cfg.sample_size_mb * self.cfg.val_percent}MB"
                    " of val data"
                )
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
            self.train_dataset_tokens = self.train_dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=self.train_dataset.column_names,
            )
        if stage == "validate" or stage is None:
            self.val_dataset = datasets.load_dataset(
                self.save_dir, split="validation", cache_dir=self.save_dir
            )
            self.val_dataset_tokens = self.val_dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=self.val_dataset.column_names,
            )
        self.is_setup = True

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.pad_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map["pad_token"]
        )
        self.max_seq_length = min(1024, tokenizer.model_max_length)
        self.vocab_size = len(self.tokenizer)

    def tokenize_fn(self, batch):
        output = self.tokenizer(
            batch["text"],
            truncation=False,
        )

        # concatenate every sample into one single list, separating throughout
        concat_input_ids = flatten_list_with_separator(
            output["token_ids"], self.tokenizer.eos_token_id
        )
        concat_attention_mask = flatten_list_with_separator(output["attention_mask"], 0)

        # determine how much to pad to make divisible by max_seq_length
        num_pad = (
            self.max_seq_length - (len(concat_input_ids) % self.max_seq_length)
        ) % self.max_seq_length

        # pad to make divisible by max_seq_length if necessary
        if num_pad != 0:
            concat_input_ids += [self.pad_token_id] * num_pad
            concat_attention_mask += [0] * num_pad

        # convert to LongTensors and shape them into (batch_size, max_seq_length)
        input_ids = torch.LongTensor(concat_input_ids).view(-1, self.max_seq_length)
        attention_mask = torch.LongTensor(concat_attention_mask).view(
            -1, self.max_seq_length
        )
        # note that the batch size here could be different from specified batch size

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Converts a list of dictionaries of tensors into a dictionary of tensors
        """
        # converting to dict of tensors
        return transformers.default_data_collator(features)


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


@hydra.main(version_base=None, config_path="../conf/data/", config_name="oscar")
def main(cfg: DictConfig):
    """downloads and processes the data for benchmark for each of the available languages"""
    print(cfg)
    # use +lang={en/fr/de} in the cli to get cfg.lang
    oscar = OSCARDataModule(cfg, cfg.lang, 1)
    oscar.prepare_data()
    oscar.setup()


if __name__ == "__main__":
    main()
