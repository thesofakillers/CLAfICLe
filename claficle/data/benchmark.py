import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool  # multithreading for IO operations
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Dict, Tuple

from torch.utils.data import DataLoader, SequentialSampler
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from omegaconf import DictConfig
import datasets

from claficle.data.process import process_dataset


class BenchmarkDataModule(pl.LightningDataModule):
    """
    PL DataModule responsible for dataloaders for various datasets used
    for our multitask benchmarking.

    Note: run `set_tokenizer(tokenizer)` before asking for a dataloader

    PS: you may also wish to run `set_pre_collate_fn(fn)`
    to apply any pre-collation processing
    """

    def __init__(self, config: DictConfig, lang: str):
        super().__init__()
        self.cfg = config
        self.lang = lang
        self.raw_save_path: str = os.path.join(self.cfg.data_dir, "raw")
        self._metadata = {"lang": self.lang, "datasets": []}
        self._pre_collate_fn: Callable[
            ..., List[Dict]
        ] = lambda batch: batch  # default no-op (can be set)
        self.is_setup = False

    def prepare_data(self):
        """takes care of downloading data"""
        if self.is_setup:
            return
        print("Loading datasets...")
        # make use of multithreading to speed up by downloading in parallel
        thread_pool = ThreadPool(cpu_count())
        thread_pool.map(self._download_dataset, self.cfg.dataset_names)

    def setup(self, stage: Optional[str] = None):
        """
        processes each dataset, obtaining test split and relevant metric(s)
        """
        if self.is_setup:
            return
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
        self.is_setup = True

    def get_metadata(self):
        return self._metadata

    def collate_fn(self, batch: List[Dict]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        For each input, encodes it and concatenates it with each
        of available (encoded) options
        Padding is applied in the process
        token_type_ids (0 is input, 1 is option, 2 is padding) are tracked throughout
        Batches labels into LongTensor

        Returns Tuple of (input_ids, token_type_ids, labels)
        Dimensions of ((B x O x S), (B x O x S), (B, ))
        where B is batch size, O is number of options, S is max sequence length in B
        """
        # apply any pre-collation processing first
        pre_collate_kwargs = {
            "src_lang": self.lang,
        }
        proc_batch: List[Dict] = self._pre_collate_fn(batch=batch, **pre_collate_kwargs)

        # batch encode the inputs
        input_encodings = self.tokenizer(
            [x["input"] for x in proc_batch], truncation=True
        )["input_ids"]

        # we then go through batch to concatenate each option to a given input
        batch_concats = []
        batch_tok_type_ids = []
        batch_labels = []
        for input_encoding, item in zip(input_encodings, proc_batch):
            batch_labels.append(item["label"])
            input_tok_type_ids = [0 for _ in input_encoding]

            # encode each option, prefixed by separator
            option_encodings = self.tokenizer(
                [self.cfg.separator + option for option in item["options"]],
                truncation=True,
            )["input_ids"]

            # we then concatenate each option to our current input encoding
            tok_type_ids = []
            concat_encodings = []
            for option_encoding in option_encodings:
                concatenated = input_encoding + option_encoding
                tok_type_id = input_tok_type_ids + [1 for _ in option_encoding]
                # truncate from left side to see most recent tokens if necessary
                concatenated = concatenated[-self.max_seq_length :]
                tok_type_id = tok_type_id[-self.max_seq_length :]
                # and add to options
                concat_encodings.append(torch.LongTensor(concatenated))
                tok_type_ids.append(torch.LongTensor(tok_type_id))

            # here we are padding across options
            concat_encodings = pad_sequence(
                concat_encodings, batch_first=False, padding_value=self.pad_token_id
            )
            tok_type_ids = pad_sequence(
                tok_type_ids, batch_first=False, padding_value=2
            )

            batch_concats.append(concat_encodings)
            batch_tok_type_ids.append(tok_type_ids)

        # here we pad across the batch
        batch_concats = pad_sequence(
            batch_concats, batch_first=True, padding_value=self.pad_token_id
        ).permute(0, 2, 1)
        batch_tok_type_ids = pad_sequence(
            batch_tok_type_ids, batch_first=True, padding_value=2
        ).permute(0, 2, 1)

        return batch_concats, batch_tok_type_ids, torch.LongTensor(batch_labels)

    def test_dataloader(self) -> List[DataLoader]:
        """Returns a test dataloader for each processed dataset"""
        return [
            DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=SequentialSampler(dataset),
                collate_fn=self.collate_fn,
            )
            for dataset in self._processed_datasets
        ]

    def _load_raw_dataset(self, dataset_name: str):
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

    def _download_dataset(self, dataset_name: str):
        """Downloads huggingface dataset"""
        self._load_raw_dataset(dataset_name)
        return

    def set_pre_collate_fn(self, pre_collate_fn: Callable[[List[Dict]], List[Dict]]):
        """
        Sets a pre-collate processing function, which is applied to each batch
        before collation
        """
        self._pre_collate_fn = pre_collate_fn

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.pad_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map["pad_token"]
        )
        self.max_seq_length = tokenizer.model_max_length


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
