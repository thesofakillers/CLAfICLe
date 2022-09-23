def prepare_qam_example(example, separator):
    example["input"] = example["question"] + separator + example["answer"]
    return example


qam_kwargs = {
    "get_k_source": lambda dataset, lang: dataset[f"test.{lang}"],
    "k_from_test": True,
    "get_test_split": None,
    "get_options": lambda test_split: tuple(
        range(test_split.features["label"].num_classes)
    ),
    "remove_columns": ["question", "answer"],
    "preparer": prepare_qam_example,
}
