from claficle.data.process.utils import ProcessHelper


class AmazonHelper(ProcessHelper):
    rename_cols = {}

    @staticmethod
    def language_available(dataset_name, lang):
        collection_name, parsed_lang = dataset_name.split(";")
        return (collection_name, parsed_lang == lang)

    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["validation"]

    k_from_test = False

    @staticmethod
    def get_test_split(dataset, lang):
        return dataset["test"]

    @staticmethod
    def get_options(example):
        example["options"] = [str(x) for x in range(1, 6)]
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["review_body"]
        example["label"] = example["stars"] - 1
        return example

    remove_cols = [
        "review_body",
        "stars",
        "review_id",
        "product_id",
        "reviewer_id",
        "review_title",
        "language",
        "product_category",
    ]
