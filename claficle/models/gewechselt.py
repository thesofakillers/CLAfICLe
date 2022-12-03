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
        # TODO

    def run_causal_model(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError
