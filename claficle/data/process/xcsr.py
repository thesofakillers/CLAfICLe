from claficle.data.process.utils import ProcessHelper


class XCSRHelper(ProcessHelper):
    rename_cols = {"answerKey": "label"}

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

    @staticmethod
    def get_options(example):
        example["options"] = example["question"]["choices"]["text"]
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["question"]["stem"]
        example["label"] = ord(example["label"].lower()) - ord("a")
        return example

    is_classification = False


class CSQAHelper(XCSRHelper):

    remove_cols = ["lang", "id", "question", "stem"]


class CODAHHelper(XCSRHelper):

    remove_cols = ["lang", "id", "question", "question_tag", "stem"]
