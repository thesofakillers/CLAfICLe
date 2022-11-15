"""Sandwich model: lang -> english -> lang"""
from typing import Dict, List

from googletrans import Translator
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
import torch
from claficle.models.base import BaseModel


class Sandwich(BaseModel):

    """
    Uses Google Translate API to use BREAD_LANG inputs on a FILL_LANG model
    BREAD_LANG -> FILL_LANG_MODEL -> BREAD_LANG
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._translator = Translator()
        state_dict = torch.load(config.lm_checkpoint_path)
        self.lm = AutoModelForCausalLM.from_pretrained(
            config.causalLM_variant, state_dict=state_dict
        )

    def translate(self, text: str, src_lang: str, dest_lang: str) -> str:
        """Translates a piece of text"""
        return self._translator.translate(text, src=src_lang, dest=dest_lang).text

    def run_causal_model(self, input_ids, attention_mask):
        return self.lm(input_ids=input_ids, attention_mask=attention_mask)

    def pre_collate(
        self, batch: List[Dict], src_lang: str, dest_lang: str = "en"
    ) -> List[Dict]:
        """Translates text from `bread` to `fill` language"""
        for item in batch:
            item["input"] = self.translate(
                item["input"], src_lang=src_lang, dest_lang=dest_lang
            )
            item["options"] = [
                self.translate(
                    str(option),
                    src_lang=src_lang,
                    dest_lang=dest_lang,
                )
                for option in item["options"]
            ]
        return batch


if __name__ == "__main__":
    # generating checkpoint
    import yaml
    from omegaconf import OmegaConf

    with open("claficle/conf/model/sandwhich.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    cfg: DictConfig = OmegaConf.create(config)

    sandwich = Sandwich(config=cfg)

    torch.save(sandwich.state_dict(), cfg.checkpoint_path)
