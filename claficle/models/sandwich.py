"""Sandwich model: lang -> english -> lang"""
from typing import Dict, List, Optional

from googletrans import Translator
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
import torch

from claficle.models.base import BaseModel

translator = Translator()


def translate_batch(texts: List[str], src_lang: str, dest_lang: str) -> List[str]:
    """Translate a batch of strings"""
    res = translator.translate(texts, src=src_lang, dest=dest_lang)
    return [trans.text for trans in res]


def translate_single_text(
    text: str, src_lang: str, dest_lang: str, separator: str
) -> str:
    """Translate a single string. Will chunk string if too long"""
    if len(text) > 4000:
        chunks = text.split(separator * 3)
        trans_chunks = translate_batch(chunks, src_lang, dest_lang)
        text = (separator * 3).join(trans_chunks)
    else:
        text = translator.translate(text, src=src_lang, dest=dest_lang)
    return text


class Sandwich(BaseModel):

    """
    Uses Google Translate API to use BREAD_LANG inputs on a FILL_LANG model
    BREAD_LANG -> FILL_LANG_MODEL -> BREAD_LANG
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def run_causal_model(self, input_ids, attention_mask):
        return self.lm(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def pre_collate(batch: List[Dict], **kwargs) -> List[Dict]:
        """Translates text from `src_lang` to `dest_lang` language"""
        default_kwargs = {"src_lang": "en", "dest_lang": "en", "separator": "\n"}
        kwargs = {**default_kwargs, **kwargs}
        src_lang, dest_lang, separator = (
            kwargs["src_lang"],
            kwargs["dest_lang"],
            kwargs["separator"],
        )
        if src_lang == dest_lang:
            return batch
        for item in batch:
            item["input"] = translate_single_text(
                item["input"], src_lang, dest_lang, separator
            )
            item["options"] = translate_batch(item["options"], src_lang, dest_lang)

        return batch

    def load_non_pl_checkpoint(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = None

        self.lm = AutoModelForCausalLM.from_pretrained(
            self.hparams.causalLM_variant, state_dict=state_dict
        )


@hydra.main(version_base=None, config_path="../conf/model", config_name="sandwich")
def main(cfg: DictConfig):
    # testing
    print(cfg)
    if cfg.pl_checkpoint:
        model = Sandwich.load_from_checkpoint(cfg.checkpoint_path)
    else:
        model = Sandwich(cfg)
        model.load_non_pl_checkpoint(cfg.checkpoint_path)

    print(model)


if __name__ == "__main__":
    main()
