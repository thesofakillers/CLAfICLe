from claficle.data.process.utils import ProcessHelper


class XGLUEHelper(ProcessHelper):
    @staticmethod
    def get_k_source(dataset, lang):
        return dataset[f"validation.{lang}"]

    @staticmethod
    def get_test_split(dataset, lang):
        return dataset[f"test.{lang}"]

    is_classification = True

    @staticmethod
    def get_options(example):
        example["options"] = tuple(range(2))
        return example

    @staticmethod
    def language_available(dataset_name, lang):
        if lang in {"en", "fr", "de"}:
            return True
        else:
            return False


class QAMHelper(XGLUEHelper):
    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["question"] + separator + example["answer"]
        return example

    remove_cols = ["question", "answer"]


class QADSMHelper(XGLUEHelper):
    @staticmethod
    def prepare_example(example, separator):
        example["input"] = (
            example["query"]
            + separator
            + example["ad_title"]
            + separator
            + example["ad_description"]
        )
        return example

    rename_cols = {"relevance_label": "label"}

    remove_cols = ["query", "ad_title", "ad_description"]


class NCHelper(XGLUEHelper):
    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["news_title"] + separator + example["news_body"]
        return example

    rename_cols = {"news_category": "label"}

    remove_cols = ["news_title", "news_body"]

    @staticmethod
    def get_options(example):
        example["options"] = tuple(range(10))
        return example


class PAWSXHelper(XGLUEHelper):
    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["sentence1"] + separator + example["sentence2"]
        return example

    remove_cols = ["sentence1", "sentence2"]


class XNLIHelper(XGLUEHelper):
    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["premise"] + separator + example["hypothesis"]
        return example

    remove_cols = ["premise", "hypothesis"]

    @staticmethod
    def get_options(example):
        example["options"] = tuple(range(3))
        return example
