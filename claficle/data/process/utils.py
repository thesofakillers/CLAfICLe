from typing import Callable

from datasets.arrow_dataset import Dataset
import numpy as np


def prepare_kshot_str(
    k_shot_subdataset: Dataset, separator: str, preparer: Callable
) -> str:
    k_shot_str = ""

    for i, example in enumerate(k_shot_subdataset):
        prep_example = preparer(example, separator)
        k_shot_str += (
            f"{prep_example['input']}{separator}{prep_example['label']}{separator*3}"
        )

    return k_shot_str


def prepend_kshot(example, k_shot_str):
    example["input"] = k_shot_str + example["input"]
    return example


def prepare_and_process(example, k_shot_str: str, separator: str, preparer: Callable):
    prepared_example = preparer(example, separator)
    processed_example = prepend_kshot(prepared_example, k_shot_str, separator)
    return processed_example


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

    k_from_test = None

    @staticmethod
    def get_test_split(dataset, lang):
        raise NotImplementedError

    @staticmethod
    def get_options(dataset):
        raise NotImplementedError

    remove_columns = None

    @staticmethod
    def prepare_example(example, separator):
        raise NotImplementedError

    is_classification = True
