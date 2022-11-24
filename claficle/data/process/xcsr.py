from claficle.data.process.utils import ProcessHelper


class XCSRHelper(ProcessHelper):
    rename_cols = {"answerKey": "label"}

    @staticmethod
    def language_available(dataset_name, lang):
        dataset_lang = dataset_name[-2:]
        collection_name = dataset_name[5:-3]
        lang_avail = dataset_lang == lang
        return (f"xcsr-{collection_name}", lang_avail)

    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["validation"]

    k_from_test = True

    is_classification = False


class CSQAHelper(XCSRHelper):

    remove_cols = ["lang", "id", "question"]

    @staticmethod
    def get_options(example):
        example["options"] = [
            text if text != "" else "-"
            for text in example["question"]["choices"]["text"]
        ]
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["question"]["stem"]
        example["label"] = ord(example["label"].lower()) - ord("a")
        return example


class CODAHHelper(XCSRHelper):

    remove_cols = ["lang", "id", "question", "question_tag"]

    @staticmethod
    def get_options(example):
        example["options"] = example["question"]["choices"]["text"]
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = ""
        example["label"] = ord(example["label"].lower()) - ord("a")
        return example
