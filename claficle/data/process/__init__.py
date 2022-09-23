from typing import Any, Dict, Tuple, List

from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
import numpy as np

import claficle.data.process.utils as utils

from claficle.data.process.xglue import NCHelper, PAWSXHelper, QAMHelper, QADSMHelper

helper_by_name: Dict[str, Dict] = {
    "xglue;qam": QAMHelper,
    "xglue;qadsm": QADSMHelper,
    "xglue;nc": NCHelper,
    "xglue;paws-x": PAWSXHelper,
}


def process_dataset(
    dataset: Dataset,
    lang: str,
    cfg: DictConfig,
    Helper: utils.ProcessHelper,
) -> Tuple[Any, List[str]]:
    """
    Gets relevant splits
    Generates k-shot context from non-test portion of data
    Prepends each test input with k-shot context
    Adds options column to track options
    Returns processed test dataset and relevant metrics
    """
    for source, target in Helper.rename_cols:
        dataset = dataset.rename_column(source, target)
    k_shot_source = Helper.get_k_source(dataset, lang)
    k_shot, k_indices = utils.get_k_shot_subset(k_shot_source, cfg.k_shot)
    k_shot_string = utils.prepare_kshot_str(
        k_shot, cfg.separator, Helper.prepare_example
    )

    if Helper.k_from_test:
        test_indices = np.setdiff1d(np.arange(len(k_shot_source)), k_indices)
        test_split = k_shot_source.select(test_indices)
    else:
        test_split = Helper.get_test_split(dataset, lang)

    processed_test_split = test_split.map(
        utils.prepare_and_process,
        remove_columns=Helper.remove_cols,
        fn_kwargs={
            "k_shot_str": k_shot_string,
            "separator": cfg.separator,
            "preparer": Helper.prepare_example,
            "optioner": Helper.get_options,
        },
    )
    if Helper.is_classification:
        metrics = ["f1"]
    else:
        metrics = ["accuracy"]

    return processed_test_split, metrics
