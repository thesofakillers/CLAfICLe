import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, List, Optional
from googletrans.gtoken import time

from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
import datasets
import numpy as np

import claficle.data.process.utils as utils

from claficle.data.process import xglue, xcsr, hatecheck, winox, swissjudge, amazon
from claficle.data.process.utils import translate_bulk, translate_single_text


def translate_kshot(k_shot: str, fn_kwargs: Dict) -> str:
    src_lang, dest_lang, separator = (
        fn_kwargs["src_lang"],
        fn_kwargs["dest_lang"],
        fn_kwargs["separator"],
    )
    if src_lang == dest_lang:
        return k_shot
    k_shot = k_shot.strip()  # translate single text expects stripped string
    translated = translate_single_text(k_shot, src_lang, dest_lang, separator)
    # since  we stripped, we need to add the final separator back
    translated += separator * 3
    return translated


def translate_options(options: List, fn_kwargs: Dict):
    src_lang, dest_lang = fn_kwargs["src_lang"], fn_kwargs["dest_lang"]
    if src_lang == dest_lang:
        return options
    return translate_bulk(options, src_lang, dest_lang)


def translate_batch(
    batch, src_lang: str, dest_lang: str, separator: str, processed_options=None
):
    # no work necessary
    if src_lang == dest_lang:
        return batch
    # since we already translated the k-shot context, we only need to translate the input
    split_batch: List[List[str]] = [x.split(separator * 3) for x in batch["input"]]
    inputs = [batch_elems[-1] for batch_elems in split_batch]
    # no need to translate if inputs are all ''
    if all([x == "" for x in inputs]):
        trans_inputs = inputs
    else:
        trans_inputs = None
        while trans_inputs is None:
            try:
                trans_inputs = translate_bulk(inputs, src_lang, dest_lang)
            except Exception as e:
                print("Error translating input, retrying in 2 seconds...")
                print(e)
                time.sleep(2)
                pass
    batch["input"] = [  # when done, we need to rejoin the k-shot context
        (separator * 3).join([*batch_elems[:-1], x])
        for batch_elems, x in zip(split_batch, trans_inputs)
    ]
    # options
    if processed_options is not None:
        batch["options"] = [processed_options for _x in batch["input"]]
    else:
        trans_options = None
        while trans_options is None:
            try:
                trans_options = [
                    translate_bulk(options, src_lang, dest_lang)
                    for options in batch["options"]
                ]
            except Exception as e:
                print("Error translating options, retrying in 2 seconds...")
                print(e)
                time.sleep(2)
                pass
        batch["options"] = trans_options
    return batch


# maps dataset name to helper class
HELPER_BY_NAME: Dict[str, Dict] = {
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

EXTRA_FN_BY_NAME: Dict[Optional[str], Callable[..., Any]] = {
    None: None,
    "translate": translate_batch,
}

EXTRA_FN_OPS_BY_NAME: Dict[Optional[str], Callable[..., Any]] = {
    None: None,
    "translate": translate_options,
}

EXTRA_FN_KSHOT_BY_NAME: Dict[Optional[str], Callable[..., Any]] = {
    None: None,
    "translate": translate_kshot,
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
    Helper = HELPER_BY_NAME[dataset_name]
    rng = np.random.default_rng(cfg.seed)

    if Helper.is_classification:
        metrics = ["f1"]
    else:
        metrics = ["accuracy"]

    collection_name, language_available = Helper.language_available(dataset_name, lang)
    if not language_available:
        return None, [], collection_name
    print(f"Processing {collection_name}")

    # extra proc fns
    extra_proc_fn: Optional[Callable] = EXTRA_FN_BY_NAME[cfg.extra_proc_fn]
    extra_proc_fn_options: Optional[Callable] = EXTRA_FN_OPS_BY_NAME[cfg.extra_proc_fn]
    extra_proc_fn_kshot: Optional[Callable] = EXTRA_FN_KSHOT_BY_NAME[cfg.extra_proc_fn]

    # where to save/load processed data
    dataset_path = os.path.join(
        cfg.data_dir,
        "processed",
        collection_name,
        f"seed_{cfg.seed}",
        cfg.extra_proc_fn if cfg.extra_proc_fn is not None else "",
        lang,
    )
    if os.path.exists(dataset_path):  # no need to process again
        processed_test_split = datasets.load_from_disk(dataset_path)
        return processed_test_split, metrics, collection_name

    # column renaming
    for source, target in Helper.rename_cols.items():
        processed_test_split = processed_test_split.rename_column(source, target)

    # k-shot context handling
    k_shot_source = Helper.get_k_source(processed_test_split, lang)
    k_shot, k_indices = utils.get_k_shot_subset(k_shot_source, cfg.k, rng)
    k_shot_string = utils.prepare_kshot_str(
        k_shot, cfg.separator, Helper.prepare_example, Helper.get_options
    )
    if extra_proc_fn_kshot is not None:  # additional k_shot processing if requested
        k_shot_string = extra_proc_fn_kshot(
            k_shot_string,
            fn_kwargs={"src_lang": lang, "dest_lang": "en", "separator": cfg.separator},
        )

    # handling splits
    if Helper.k_from_test:
        test_indices = np.setdiff1d(np.arange(len(k_shot_source)), k_indices)
        test_split = k_shot_source.select(test_indices)
    else:
        test_split = Helper.get_test_split(processed_test_split, lang)

    # finally, processing
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

    # additional processing if needed
    if extra_proc_fn is not None:
        # if we're doing classification, options are same for every example
        if Helper.is_classification:
            processed_options = extra_proc_fn_options(
                processed_test_split[0]["options"],
                fn_kwargs={"src_lang": lang, "dest_lang": "en"},
            )
        else:
            processed_options = None
        processed_test_split = processed_test_split.map(
            extra_proc_fn,
            fn_kwargs={
                "src_lang": lang,
                "dest_lang": "en",
                "separator": cfg.separator,
                "processed_options": processed_options,
            },
            batched=True,
            batch_size=cfg.batch_size,
        )
    # create save directory if it doesn't exist, and save to disk
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    processed_test_split.save_to_disk(dataset_path)

    return processed_test_split, metrics, collection_name
