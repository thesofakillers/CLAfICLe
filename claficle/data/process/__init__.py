from typing import Any, Dict, Tuple, List

from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
import numpy as np

import claficle.data.process.utils as utils

from claficle.data.process import xglue, xcsr

helper_by_name: Dict[str, Dict] = {
    "xglue;qam": xglue.QAMHelper,
    "xglue;qadsm": xglue.QADSMHelper,
    "xglue;nc": xglue.NCHelper,
    "xglue;paws-x": xglue.PAWSXHelper,
    "xglue;xnli": xglue.XNLIHelper,
    "xcsr;X-CSQA-en": xcsr.CSQAHelper,
    "xcsr;X-CSQA-fr": xcsr.CSQAHelper,
    "xcsr;X-CSQA-de": xcsr.CSQAHelper,
}


def process_dataset(
    dataset: Dataset, lang: str, cfg: DictConfig, dataset_name: str
) -> Tuple[Any, List[str]]:
    """
    Gets relevant splits
    Generates k-shot context from non-test portion of data
    Prepends each test input with k-shot context
    Adds options column to track options
    Returns processed test dataset and relevant metrics
    """
    Helper = helper_by_name[dataset_name]
    if not Helper.language_available(dataset_name, lang):
        return None, []
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
