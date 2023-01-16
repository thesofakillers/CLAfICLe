import os
from pathlib import Path
from typing import Dict, Optional, List
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

datasets.disable_caching()


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
            dataset_dict = datasets.load_from_disk(self.raw_save_dir)
            self.train_dataset = dataset_dict["train"]
            self.val_dataset = dataset_dict["validation"]
            return
        # otherwise, download and save
        else:
            Path(self.raw_save_dir).mkdir(parents=True, exist_ok=True)
            data_stream: datasets.iterable_dataset.IterableDataset = (
                datasets.load_dataset(
                    "oscar",
                    f"unshuffled_deduplicated_{self.lang}",
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

            self.train_dataset = Dataset.from_generator(
                gen,
                gen_kwargs={
                    "start": 0,
                    "stop": self.cfg.num_samples + 1,
                    "total": self.cfg.num_samples,
                    "desc": "Train stream",
                },
            )
            self.val_dataset = Dataset.from_generator(
                gen,
                gen_kwargs={
                    "start": self.cfg.num_samples,
                    "stop": self.cfg.num_samples + num_val_samples + 1,
                    "total": self.cfg.num_samples + num_val_samples,
                    "desc": "Validation stream",
                },
            )
            # collect into a dataset dict and save them to disk
            dataset_dict = DatasetDict(
                {"train": self.train_dataset, "validation": self.val_dataset}
            )
            dataset_dict.save_to_disk(self.raw_save_dir, fs="deprecated")
            return

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        if stage == "fit" or stage is None:
            self.train_dataset_tokens = self.setup_split("train")
        if stage == "validate" or stage is None:
            self.val_dataset_tokens = self.setup_split("validation")
        else:
            raise ValueError(f"Invalid stage: {stage}")
        self.is_setup = True

    def setup_split(self, split: str):
        # load from disk if we already tokenized:
        processed_path = os.path.join(self.processed_save_dir, f"{split}_tokenized")
        if os.path.exists(processed_path):
            print("Dataset already tokenized. Loading from disk")
            dataset_tokens = datasets.load_from_disk(processed_path)
        else:
            dataset = self.train_dataset if split == "train" else self.val_dataset
            dataset_tokens = dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
            )
            # save to disk for next time
            os.makedirs(processed_path, exist_ok=True)
            dataset_tokens.save_to_disk(processed_path, fs="deprecated")
        return dataset_tokens

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        # see https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
        self.tokenizer.pad_token = self.tokenizer.eos_token
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


@hydra.main(version_base=None, config_path="../conf/", config_name="setup_data")
def main(cfg: DictConfig):
    """
    downloads and/or processes OSCAR for each of the available languages

    NOTE:
    Downloading and processing may need to happen separately.
    This is because in French and German, we train the tokenizer using
    the raw training data. This training occurs in run/wechsel_init.py.
    Once the tokenizer is trained, we can proceed with dataset tokenization.
    We can omit processing by not passing the tokenizer path in the cfg.
    Running the script a second time will not re-download the data.
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
    oscar = OSCARDataModule(cfg.data, cfg.lang, 1)
    oscar.prepare_data()

    # optionally, load the tokenizer and perform tokenization
    if cfg.tokenizer_name is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            os.path.join("checkpoints", "tokenizers", cfg.tokenizer_name)
        )
        oscar.set_tokenizer(tokenizer)
        oscar.setup()
    # for english, we can always do the tokenization
    elif cfg.tokenizer_name is None and cfg.lang == "en":
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large")
        oscar.set_tokenizer(tokenizer)
        oscar.setup()


if __name__ == "__main__":
    main()
