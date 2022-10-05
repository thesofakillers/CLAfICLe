from claficle.data.process.utils import ProcessHelper


class SwissJudgeHelper(ProcessHelper):
    @staticmethod
    def language_available(dataset_name, lang):
        parsed_lang = dataset_name.split(";")[-1][-2:]
        return parsed_lang == lang

    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["validation"]

    k_from_test = False

    @staticmethod
    def get_test_split(dataset, lang):
        return dataset["test"]

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
