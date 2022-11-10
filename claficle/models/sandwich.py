"""Sandwich model: lang -> english -> lang"""
from googletrans import Translator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
from claficle.models.base import BaseModel


class Sandwich(BaseModel):

    """
    Uses Google Translate API to use BREAD_LANG inputs on a FILL_LANG model
    BREAD_LANG -> FILL_LANG_MODEL -> BREAD_LANG
    """

    def __init__(self, bread: str, model: str, fill: str = "en"):
        """Initialization

        :param bread: desired input and output language (ISO-639-1)
        :param model: HF Hub name or path to the model to use as the filling
        :param fill: language the language model is capable of (ISO-639-1)

        """
        self.bread = bread
        self.fill = fill
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._translator = Translator()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """Translates a piece of text"""
        return self._translator.translate(text, src=src_lang, dest=dest_lang).text

    def generate(self, bread_input: str) -> str:
        """
        Translates BREAD to FILL  and passes to FILL model
        Translates FILL model output to BREAD
        """
        fill_input: str = self.translate(
            bread_input, src_lang=self.bread, dest_lang=self.fill
        )

        fill_input_tokens = self.tokenizer.encode(fill_input, return_tensors="pt")
        fill_output_tokens = self.model.generate(fill_input_tokens)[0]
        fill_output = self.tokenizer.decode(
            fill_output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        bread_output: str = self.translate(
            fill_output, src_lang=self.fill, dest_lang=self.bread
        )

        return bread_output


if __name__ == "__main__":
    test_sandwich = Sandwich(bread="it", model="google/t5-small-lm-adapt", fill="en")

    test_output = test_sandwich.generate(
        "'Ho fame, che cosa mi suggerisci da mangiare?''Prova il "
    )

    print(test_output)
