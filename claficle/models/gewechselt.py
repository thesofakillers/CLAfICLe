"""
A Gewechselt model: A model to which WECHSEL is applied
"""
from omegaconf import DictConfig
import wechsel

from claficle.models.base import BaseModel


class Gewechselt(BaseModel):
    """
    Model initialized using WECHSEL (Minixhofer et al. 2022)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def initialize(self, config: DictConfig) -> AutoTokenizer:
        """Applies WECHSEL initialization"""
        source_tokenizer = AutoTokenizer.from_pretrained(config.causalLM_variant)
        lm = AutoModelForCausalLM.from_pretrained(config.causalLM_variant)
        # TODO: the rest of the wechsel tutorial

    def run_causal_model(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError
