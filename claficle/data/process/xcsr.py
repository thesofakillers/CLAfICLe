from claficle.data.process.utils import ProcessHelper


class XCSRHelper(ProcessHelper):
    @staticmethod
    def language_available(dataset_name, lang):
        dataset_lang = dataset_name[-2:]
        if dataset_lang == lang:
            return True
        else:
            return False

    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["validation"]

    k_from_test = True


class CSQAHelper(XCSRHelper):
    @staticmethod
    def get_options(example):
        labels, text = tuple(example["question"]["choices"].values())
        example["options"] = [f"{label}: {text}" for label, text in zip(labels, text)]

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["question"]["stem"]

    rename_cols = {"answerKey": "label"}

    is_classification = False

    remove_cols = ["lang", "id", "question"]
