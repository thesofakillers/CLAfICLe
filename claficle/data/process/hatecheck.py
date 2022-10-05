from claficle.data.process.utils import ProcessHelper


class HatecheckHelper(ProcessHelper):
    @staticmethod
    def get_k_source(dataset, lang):
        return dataset["test"]

    k_from_test = True

    @staticmethod
    def get_options(example):
        example["options"] = ("hateful", "non-hateful")
        return example

    rename_cols = {"test_case": "input", "label_gold": "label"}
    is_classification = True

    remove_cols = [
        "case_templ",
        "functionality",
        "ref_case_id",
        "ref_templ_id",
        "target_ident",
        "templ_id",
    ]

    @staticmethod
    def prepare_example(example, separator):
        example["label"] = {"hateful": 0, "non-hateful": 1}[example["label"]]


class EnglishHelper(HatecheckHelper):
    @staticmethod
    def language_available(dataset_name, lang):
        return lang == "en" and dataset_name == "Paul/hatecheck"

    remove_cols = HatecheckHelper.remove_cols + [
        "case_id",
        "direction",
        "focus_lemma",
        "focus_words",
    ]


class NonEnglishHelper(HatecheckHelper):
    @staticmethod
    def language_available(dataset_name, lang):
        return (lang == "de" and dataset_name == "Paul/hatecheck-german") or (
            lang == "fr" and dataset_name == "Paul/hatecheck-french"
        )

    remove_cols = HatecheckHelper.remove_cols + [
        "disagreement_in_case",
        "disagreement_in_template",
        "gender_female",
        "gender_male",
        "label_annotated",
        "label_annotated_maj",
        "mhc_case_id",
    ]
