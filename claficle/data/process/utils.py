from typing import Callable

from datasets.arrow_dataset import Dataset
import numpy as np


def prepare_kshot_str(
    k_shot_subdataset: Dataset, separator: str, preparer: Callable, optioner: Callable
) -> str:
    k_shot_str = ""

    for i, example in enumerate(k_shot_subdataset):
        prep_example = preparer(example, separator)
        prep_example = optioner(prep_example)
        k_shot_str += (
            f"{prep_example['input']}"
            f"{separator}"
            f"{prep_example['options'][prep_example['label']]}"
            f"{separator*3}"
        )

    return k_shot_str


def prepend_kshot(example, k_shot_str):
    example["input"] = k_shot_str + example["input"]
    return example


def prepare_and_process(
    example, k_shot_str: str, separator: str, preparer: Callable, optioner: Callable
) -> dict:
    # prepare so we have 'input' field
    prepared_example = preparer(example, separator)
    # prepend k-shot context to 'input'
    processed_example = prepend_kshot(prepared_example, k_shot_str, separator)
    # add 'options' field
    proc_example_with_options = optioner(processed_example)
    return proc_example_with_options


def get_k_shot_subset(dataset: Dataset, k: int):
    data_len = len(dataset)
    k_indices = np.random.choice(data_len, k, replace=False)
    k_shot = dataset.select(k_indices)
    return k_shot, k_indices


class ProcessHelper:
    """Base class defining skeleton for (sub)dataset helpers"""

    @staticmethod
    def get_k_source(dataset, lang):
        raise NotImplementedError

    k_from_test = False

    @staticmethod
    def get_test_split(dataset, lang):
        raise NotImplementedError

    @staticmethod
    def get_options(example):
        raise NotImplementedError

    remove_cols = None

    @staticmethod
    def prepare_example(example, separator):
        raise NotImplementedError

    is_classification = True

    rename_cols = {}

    @staticmethod
    def language_available(dataset_name, lang):
        """Checks if language is available for dataset"""
        raise NotImplementedError
