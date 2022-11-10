import os
from pathlib import Path
from typing import Any, Dict, Tuple, List

from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
import datasets
import numpy as np

import claficle.data.process.utils as utils

from claficle.data.process import xglue, xcsr, hatecheck, winox, swissjudge, amazon

helper_by_name: Dict[str, Dict] = {
    "xglue;qam": xglue.QAMHelper,
    "xglue;qadsm": xglue.QADSMHelper,
    "xglue;nc": xglue.NCHelper,
    "xglue;paws-x": xglue.PAWSXHelper,
    "xglue;xnli": xglue.XNLIHelper,
    "xcsr;X-CSQA-en": xcsr.CSQAHelper,
    "xcsr;X-CSQA-fr": xcsr.CSQAHelper,
    "xcsr;X-CSQA-de": xcsr.CSQAHelper,
    "xcsr;X-CODAH-en": xcsr.CODAHHelper,
    "xcsr;X-CODAH-fr": xcsr.CODAHHelper,
    "xcsr;X-CODAH-de": xcsr.CODAHHelper,
    "Paul/hatecheck": hatecheck.EnglishHelper,
    "Paul/hatecheck-german": hatecheck.NonEnglishHelper,
    "Paul/hatecheck-french": hatecheck.NonEnglishHelper,
    "demelin/wino_x;lm_en_de": winox.WinoXHelper,
    "demelin/wino_x;lm_en_fr": winox.WinoXHelper,
    "swiss_judgment_prediction;mt_en": swissjudge.SwissJudgeHelper,
    "swiss_judgment_prediction;de": swissjudge.SwissJudgeHelper,
    "swiss_judgment_prediction;fr": swissjudge.SwissJudgeHelper,
    "amazon_reviews_multi;en": amazon.AmazonHelper,
    "amazon_reviews_multi;de": amazon.AmazonHelper,
    "amazon_reviews_multi;fr": amazon.AmazonHelper,
}


def process_dataset(
    processed_test_split: Dataset, lang: str, cfg: DictConfig, dataset_name: str
) -> Tuple[Any, List[str], str]:
    """
    Gets relevant splits
    Generates k-shot context from non-test portion of data
    Prepends each test input with k-shot context
    Adds options column to track options
    Returns processed test dataset and relevant metrics
    """
    Helper = helper_by_name[dataset_name]

    if Helper.is_classification:
        metrics = ["f1"]
    else:
        metrics = ["accuracy"]

    collection_name, language_available = Helper.language_available(dataset_name, lang)
    if not language_available:
        return None, [], collection_name

    dataset_path = os.path.join(cfg.data_dir, "processed", collection_name, lang)
    if os.path.exists(dataset_path):
        processed_test_split = datasets.load_from_disk(dataset_path)
        return processed_test_split, metrics, collection_name

    for source, target in Helper.rename_cols.items():
        processed_test_split = processed_test_split.rename_column(source, target)
    k_shot_source = Helper.get_k_source(processed_test_split, lang)
    k_shot, k_indices = utils.get_k_shot_subset(k_shot_source, cfg.k)
    k_shot_string = utils.prepare_kshot_str(
        k_shot, cfg.separator, Helper.prepare_example, Helper.get_options
    )

    if Helper.k_from_test:
        test_indices = np.setdiff1d(np.arange(len(k_shot_source)), k_indices)
        test_split = k_shot_source.select(test_indices)
    else:
        test_split = Helper.get_test_split(processed_test_split, lang)

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
    # create save directory if it doesn't exist, and save to disk
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    processed_test_split.save_to_disk(dataset_path)

    return processed_test_split, metrics, collection_name
