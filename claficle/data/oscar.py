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
import wandb
import numpy as np

from claficle.utils.general import flatten_list_with_separator


class OSCARDataModule(pl.LightningDataModule):
    """
    PL DataModule responsible for OSCAR dataloading

    Note: run `set_tokenizer(tokenizer)` before running self.setup()
    """

    def __init__(self, config: DictConfig, lang: str, seed: str):
        super().__init__()
        self.cfg = config
        self.is_setup = False
        self.raw_save_dir = os.path.join(
            self.cfg.data_dir, "raw", "oscar", lang, str(self.cfg.sample_size_mb)
        )
        self.processed_save_dir = os.path.join(
            self.cfg.data_dir, "processed", "oscar", lang, str(self.cfg.sample_size_mb)
        )
        self.lang = lang
        pl.seed_everything(seed)

    def prepare_data(self):
        """Take care of downloading data"""
        if self.is_setup:
            return
        # if already saved, don't need to download
        if os.path.exists(self.raw_save_dir):
            return
        # otherwise, download and save
        else:
            Path(self.raw_save_dir).mkdir(parents=True, exist_ok=True)
            if self.cfg.sample_size_mb is None:
                dataset: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                    "oscar-corpus/OSCAR-2201",
                    f"unshuffled_deduplicated_{self.lang}",
                    split="train",
                )
                # manually split into train and test and save to disk
                dataset = dataset.train_test_split(test_size=self.cfg.val_frac)
                dataset.save_to_disk(self.raw_save_dir)
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
                    dataset_iter, self.raw_save_dir, "train.json", subsample_size
                )
                # save remaining subsample_size * val_frac bytes of data as val
                print(
                    f"Downloading {self.cfg.sample_size_mb * self.cfg.val_frac}MB"
                    " of val data"
                )
                datastream_to_file(
                    dataset_iter,
                    self.raw_save_dir,
                    "validation.json",
                    subsample_size * self.cfg.val_frac,
                )

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        if stage == "fit" or stage is None:
            self.train_dataset, self.train_dataset_tokens = self.setup_split("train")
        if stage == "validate" or stage is None:
            self.val_dataset, self.val_dataset_tokens = self.setup_split("validation")
        else:
            raise ValueError(f"Invalid stage: {stage}")
        self.is_setup = True

    def setup_split(self, split: str):
        dataset = datasets.load_dataset(
            self.raw_save_dir, split=split, cache_dir=self.processed_save_dir
        )
        # load from disk if we already tokenized:
        processed_path = os.path.join(self.processed_save_dir, f"{split}_tokenized")
        if os.path.exists(processed_path):
            print("Dataset already tokenized. Loading from disk")
            dataset_tokens = datasets.load_from_disk(processed_path)
        else:
            dataset_tokens = dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
            )
            # save to disk for next time
            os.makedirs(processed_path, exist_ok=True)
            dataset_tokens.save_to_disk(processed_path)
        return dataset, dataset_tokens

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
            output["input_ids"], self.tokenizer.eos_token_id
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

        # convert to np, reshape (batch_size, max_seq_length), back to list of lists
        input_ids = np.array(concat_input_ids).reshape(-1, self.max_seq_length).tolist()
        attention_mask = (
            np.array(concat_attention_mask).reshape(-1, self.max_seq_length).tolist()
        )

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
    """
    downloads and processes OSCAR for each of the available languages
    """
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    wandb.init(
        project="claficle",
        entity="giulio-uva",
        job_type="oscar",
        config=cfg,
        mode="disabled" if cfg.disable_wandb else "online",
        group=script_host,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    # use lang={en/fr/de} in the cli to set cfg.lang
    oscar = OSCARDataModule(cfg, cfg.lang, 1)
    oscar.set_tokenizer(tokenizer)
    oscar.prepare_data()
    oscar.setup()


if __name__ == "__main__":
    main()
