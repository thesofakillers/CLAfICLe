from omegaconf import DictConfig
from claficle.models.sandwich import Sandwich
from claficle.models.plain_gpt2 import PlainGPT2
from claficle.models.gewechselt import Gewechselt
from claficle.models.vessel import Vessel

NAME_TO_CLASS = {
    "sandwich": Sandwich,
    "plain_gpt2": PlainGPT2,
    "gewechselt": Gewechselt,
    "vessel": Vessel,
}


def get_model_preamble_post_init_kwargs(cfg: DictConfig):
    NAME_TO_KWARGS = {"vessel": {"seed": cfg.seed}}
    try:
        kwargs = NAME_TO_KWARGS[cfg.model.name]
    except KeyError:
        raise ValueError(
            "This model class either does not have a post_init or its post_init is not "
            "expected in preamble"
        )
    return kwargs
