from claficle.data.process.utils import ProcessHelper


class SwissJudgeHelper(ProcessHelper):
    @staticmethod
    def language_available(dataset_name, lang):
        collection_name, parsed_lang = dataset_name.split(";")
        parsed_lang = parsed_lang[-2:]
        return (collection_name, parsed_lang == lang)

    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["train"]

    k_from_test = True

    @staticmethod
    def get_options(example):
        example["options"] = tuple(range(2))
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["text"]
        return example

    remove_cols = [
        "text",
        "id",
        "year",
        "language",
        "region",
        "canton",
        "legal area",
        "source_language",
    ]
