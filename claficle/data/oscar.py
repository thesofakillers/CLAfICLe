import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import datasets
from datasets import Dataset, DatasetDict
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
            self.cfg.data_dir, "raw", "oscar", lang, str(self.cfg.num_samples)
        )
        self.processed_save_dir = os.path.join(
            self.cfg.data_dir, "processed", "oscar", lang, str(self.cfg.num_samples)
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
            data_stream: datasets.iterable_dataset.IterableDataset = (
                datasets.load_dataset(
                    "oscar-corpus/OSCAR-2201",
                    self.lang,
                    split="train",
                    streaming=True,
                )
            )

            num_val_samples = int(self.cfg.num_samples * self.cfg.val_frac)

            # generator necessary for Dataset.from_generator below
            def gen(start, stop, total, desc):
                for i in itertools.islice(
                    tqdm(iter(data_stream), total=total, desc=desc), start, stop
                ):
                    yield i

            # make a dataset dict and save it directly to disk
            DatasetDict(
                {
                    "train": Dataset.from_generator(
                        gen,
                        gen_kwargs={
                            "start": 0,
                            "stop": self.cfg.num_samples + 1,
                            "total": self.cfg.num_samples,
                            "desc": "Train stream",
                        },
                    ),
                    "validation": Dataset.from_generator(
                        gen,
                        gen_kwargs={
                            "start": self.cfg.num_samples,
                            "stop": self.cfg.num_samples + num_val_samples + 1,
                            "total": self.cfg.num_samples + num_val_samples,
                            "desc": "Validation stream",
                        },
                    ),
                }
            ).save_to_disk(self.raw_save_dir, fs="deprecated")

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
        # just the raw dataset for this split
        dataset = datasets.load_from_disk(os.path.join(self.raw_save_dir))[split]
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
            dataset_tokens.save_to_disk(processed_path, fs="deprecated")
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
    def collate_fn(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Converts a list of dictionaries of tensors into a dictionary of tensors
        """
        # converting to dict of tensors
        return transformers.default_data_collator(features)


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
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
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
