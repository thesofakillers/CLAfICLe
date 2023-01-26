from claficle.models.sandwich import Sandwich
from claficle.models.plain_gpt2 import PlainGPT2
from claficle.models.gewechselt import Gewechselt
from claficle.models.vessel import Vessel

NAME_TO_CLASS = {
    "sandwich": Sandwich,
    "plain_gpt2": PlainGPT2,
    "gewechselt": Gewechselt,
    "vessel": Vessel
}
