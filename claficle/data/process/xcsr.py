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
        return dataset["train"]

    @staticmethod
    def get_test_split(dataset, lang):
        return dataset["validation"]

    @staticmethod
    def get_options(example):

