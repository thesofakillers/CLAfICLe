"""Sandwich model: lang -> english -> lang"""
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig

from claficle.models.base import BaseModel
from claficle.data.process.utils import translate_bulk, translate_single_text


class Sandwich(BaseModel):

    """
    Translates incoming data to English
    before passing to an English language model
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
            item["options"] = translate_bulk(item["options"], src_lang, dest_lang)

        return batch


@hydra.main(version_base=None, config_path="../conf/model", config_name="sandwich")
def main(cfg: DictConfig):
    # testing
    print(cfg)
    if cfg.pl_checkpoint:
        model = Sandwich.load_from_checkpoint(cfg.checkpoint_path)
    else:
        model = Sandwich(cfg)

    print(model)


if __name__ == "__main__":
    main()
