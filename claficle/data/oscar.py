import os
from typing import Dict, Optional, List
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import datasets
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
        self.processed_save_dir = os.path.join(
            self.cfg.data_dir, "processed", "oscar", lang, str(self.cfg.num_tokens)
        )
        self.lang = lang
        pl.seed_everything(seed)

    def prepare_data(self):
        """Take care of downloading data"""
        if self.is_setup:
            return
        self.raw_dataset = datasets.load_dataset(
            "oscar",
            f"unshuffled_deduplicated_{self.lang}",
            split="train",
            download_mode="force_redownload",  # to prevent cache from being created
            streaming=True,  # steaming means lazy loading anyway.
        )

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return
        # these are updated when calling setup_split(split, ...)
        self.entry_batches = {"train": 0, "validation": 0}
        self.train_dataset_tokens = self.setup_split("train", 0, self.cfg.num_tokens)
        num_val_tokens = int(self.cfg.num_tokens * self.cfg.val_frac)
        self.val_dataset_tokens = self.setup_split(
            "validation", self.entry_batches["train"], num_val_tokens
        )
        if stage == "debug":
            # train for much less time (10 batches of data, instead of 5000)
            self.train_dataset_tokens = self.train_dataset_tokens.select(
                range(int(512 * 10))
            )
            self.val_dataset_tokens = self.val_dataset_tokens.select(
                range(int(512 * 10 * 0.005))
            )
        self.is_setup = True

    def setup_split(self, split: str, start_batch: int, total_tokens: int):
        # load from disk if we already tokenized:
        processed_path = os.path.join(self.processed_save_dir, f"{split}_tokenized")
        if os.path.exists(processed_path):
            print("Dataset already tokenized. Loading from disk")
            dataset_tokens = datasets.load_from_disk(processed_path)
        else:
            dataset_tokens = datasets.Dataset.from_generator(
                self.token_generator,
                gen_kwargs={
                    "start_batch": start_batch,
                    "total": total_tokens,
                    "split": split,
                },
            )

            # save to disk for next time
            os.makedirs(processed_path, exist_ok=True)
            dataset_tokens.save_to_disk(processed_path, fs="deprecated")
        return dataset_tokens

    def token_generator(self, start_batch: int, total: int, split: str):
        """
        Generator for tokenizing dataset
        Using batch sizes of 1000
        Optionally skips to the start_batch of our raw_dataset
        Then tokenizes each incoming batch, yielding each element.
        This continues until (roughly) `total` tokens have been yielded
        """
        # first, setting up generator in case we need to skip a few batches
        entry_generator = itertools.islice(
            tqdm(
                self.raw_dataset.iter(batch_size=1000),
                total=start_batch,
                desc="Skipping to the right starting point",
            ),
            start_batch,
            None,
        )

        tokens_generated = 0
        with tqdm(total=total, desc=f"{split} tokens") as pbar:
            for batch in entry_generator:
                if tokens_generated > total:
                    return
                self.entry_batches[split] += 1
                tokenized_batch = self.tokenize_fn(batch)
                batch_size = len(tokenized_batch["input_ids"])
                num_tokens = batch_size * self.max_seq_length  # approximately
                pbar.update(num_tokens)
                tokens_generated += num_tokens
                for input_ids, attention_mask in zip(
                    tokenized_batch["input_ids"], tokenized_batch["attention_mask"]
                ):
                    yield {"input_ids": input_ids, "attention_mask": attention_mask}

    def tokenize_fn(self, batch):
        output = self.tokenizer(batch["text"], truncation=False)

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
    datasets.disable_caching()
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
            os.path.join(cfg.checkpoint_dir, "tokenizers", cfg.tokenizer_name)
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
