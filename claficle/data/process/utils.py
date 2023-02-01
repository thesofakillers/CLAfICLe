from typing import Callable, Dict, List

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
    """
    Prepares an example s.t. it has an input field
    Prepends the input with the k-shot context
    Adds an option field
    """
    prepared_example = preparer(example, separator)  # 'input' field
    processed_example = prepend_kshot(prepared_example, k_shot_str)  # k-shot context
    proc_example_with_options = optioner(processed_example)  # 'options' field
    return proc_example_with_options


def get_k_shot_subset(dataset: Dataset, k: int, rng: np.random.Generator):
    data_len = len(dataset)
    k_indices = rng.choice(data_len, k, replace=False)
    k_shot = dataset.select(k_indices)
    return k_shot, k_indices


class ProcessHelper:
    """Base class defining skeleton for (sub)dataset helpers"""

    @staticmethod
    def get_k_source(dataset, lang):
        """Gets the split from where the k-shot context data is sampled"""
        raise NotImplementedError

    # whether the k-shot context originates from the same split as text
    k_from_test = False

    @staticmethod
    def get_test_split(dataset, lang):
        """Gets the split from where the test data is"""
        raise NotImplementedError

    @staticmethod
    def get_options(example: Dict) -> Dict:
        """
        Edits an example such that there is a column 'options' with the options for
        that example
        """
        raise NotImplementedError

    remove_cols = None

    @staticmethod
    def prepare_example(example, separator):
        """
        Edits an example such that there is a column 'input' with
        the appropriate combination of fields
        """
        raise NotImplementedError

    # true -> use accuracy; false -> use F1. TODO: rename this variable
    is_classification = True

    # source name : target name
    rename_cols = {}

    @staticmethod
    def language_available(dataset_name, lang):
        """
        Checks if language is available for dataset
        Returns bool and language-free collection name
        """
        raise NotImplementedError
