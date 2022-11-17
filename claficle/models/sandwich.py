"""Sandwich model: lang -> english -> lang"""
from typing import Dict, List, Optional

from googletrans import Translator
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
import torch

from claficle.models.base import BaseModel

translator = Translator()


def translate(text: str, src_lang: str, dest_lang: str) -> str:
    """Translates a piece of text"""
    return translator.translate(text, src=src_lang, dest=dest_lang).text


class Sandwich(BaseModel):

    """
    Uses Google Translate API to use BREAD_LANG inputs on a FILL_LANG model
    BREAD_LANG -> FILL_LANG_MODEL -> BREAD_LANG
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    @staticmethod
    def run_causal_model(self, input_ids, attention_mask):
        return self.lm(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def pre_collate(batch: List[Dict], **kwargs) -> List[Dict]:
        """Translates text from `src_lang` to `dest_lang` language"""
        src_lang, dest_lang = kwargs["src_lang"], kwargs["dest_lang"]
        if src_lang == dest_lang:
            return batch
        for item in batch:
            item["input"] = translate(
                item["input"], src_lang=src_lang, dest_lang=dest_lang
            )
            item["options"] = [
                translate(
                    str(option),
                    src_lang=src_lang,
                    dest_lang=dest_lang,
                )
                for option in item["options"]
            ]
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
