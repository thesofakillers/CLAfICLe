from claficle.data.process.utils import ProcessHelper


def prepare_qam_example(example, separator):
    example["input"] = example["question"] + separator + example["answer"]
    return example


class XGLUEHelper(ProcessHelper):
    @staticmethod
    def get_k_source(dataset, lang):
        return dataset[f"validation.{lang}"]

    k_from_test = False

    @staticmethod
    def get_test_split(dataset, lang):
        return dataset[f"test.{lang}"]

    @staticmethod
    def get_options(dataset):
        return tuple(range(dataset.features["label"].num_classes))

    remove_columns = ["question", "answer"]

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["question"] + separator + example["answer"]
        return example

    is_classification = True
