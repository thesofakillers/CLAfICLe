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
    def get_options(dataset):
        return tuple(range(dataset.features["relevance_label"].names))


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
