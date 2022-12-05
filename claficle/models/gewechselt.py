"""
A Gewechselt model: A model to which WECHSEL is applied
"""
from typing import Tuple

from omegaconf import DictConfig
from wechsel import WECHSEL, load_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
import torch

from claficle.models.base import BaseModel

langcode_to_lang: Dict[str, str] = {
    "en": "english",
    "de": "german",
    "fr": "french",
}


class Gewechselt(BaseModel):
    """
    Model initialized using WECHSEL (Minixhofer et al. 2022)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def custom_init(
        self, tokenizer, lm, config: DictConfig
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Applies WECHSEL initialization"""
        target_tokenizer = tokenizer.train_new_from_iterator(
            # TODO: datamodule for 4GB of OSCAR
            load_dataset("oscar", "unshuffled_deduplicated_sw", split="train")["text"],
            vocab_size=len(tokenizer),
        )
        wechsel = WECHSEL(
            load_embeddings(config.source_lang),
            load_embeddings(config.target_lang),
            bilingual_dictionary=langcode_to_lang[config.target_lang],
        )
        target_embeddings, info = wechsel.apply(
            tokenizer,
            target_tokenizer,
            lm.get_input_embeddings().weight.detach().numpy(),
        )
        lm.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

        return target_tokenizer, lm

    def run_causal_model(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError
