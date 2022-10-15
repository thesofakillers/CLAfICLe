from claficle.data.process.utils import ProcessHelper


class WinoXHelper(ProcessHelper):
    rename_cols = {"answer": "label"}

    @staticmethod
    def language_available(dataset_name, lang):
        subsplit = dataset_name.split(";")[-1]
        return (lang in {"en", "de"} and subsplit == "lm_en_de") or (
            lang == "fr" and subsplit == "lm_en_fr"
        )

    @staticmethod
    def get_k_source(dataset, lang):
        dataset = dataset["test"]
        # need to do renaming and prelim removing here since langs are split by columns
        dataset = dataset.rename_columns(
            {
                f"context_{lang}": "context",
                f"option_1{lang}": "option_1",
                f"option_2{lang}": "option_2",
            }
        )
        dataset.remove_columns(
            [
                col
                for col in dataset.column_names
                if col not in {"context", "option_1", "option_2", "answer"}
            ]
        )
        return dataset

    k_from_test = True

    @staticmethod
    def get_options(example):
        example["options"] = (example["option_1"], example["option_2"])
        return example

    @staticmethod
    def prepare_example(example, separator):
        example["input"] = example["context"]
        example["label"] = example["label"] - 1
        return example

    remove_cols = ["context", "option_1", "option_2"]

    is_classification = False
