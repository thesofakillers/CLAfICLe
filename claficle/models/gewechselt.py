"""
A Gewechselt model: A model to which WECHSEL is applied
"""
from typing import Tuple, Dict

from omegaconf import DictConfig
from wechsel import WECHSEL, load_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hydra
import datasets

from claficle.models.base import PlainGPT2

langcode_to_lang: Dict[str, str] = {
    "en": "english",
    "de": "german",
    "fr": "french",
}


class Gewechselt(PlainGPT2):
    """
    GPT2 Model initialized using WECHSEL (Minixhofer et al. 2022)
    """

    def __init__(self, config: DictConfig, target_data: datasets.arrow_dataset.Dataset):
        super().__init__(config, target_data=target_data)

    def custom_init(
        self,
        tokenizer,
        lm,
        config: DictConfig,
        target_data: datasets.arrow_dataset.Dataset,  # this is a kwarg
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Applies WECHSEL initialization"""
        print("Training target tokenizer...")
        target_tokenizer = tokenizer.train_new_from_iterator(
            target_data["text"],
            vocab_size=len(tokenizer),
        )
        wechsel = WECHSEL(
            load_embeddings(config.source_lang),
            load_embeddings(config.target_lang),
            bilingual_dictionary=langcode_to_lang[config.target_lang],
        )
        print("Generating target embeddings...")
        target_embeddings, info = wechsel.apply(
            tokenizer,
            target_tokenizer,
            lm.get_input_embeddings().weight.detach().numpy(),
        )
        print("Replacing source embeddings with target embeddings...")
        lm.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
        print("Done.")

        return target_tokenizer, lm


@hydra.main(
    version_base=None, config_path="../conf/model/", config_name="base_gewechselt"
)
def main(cfg: DictConfig):
    """used for testing: initializes model"""
    from claficle.data.oscar import OSCARDataModule
    import yaml
    from yaml.loader import SafeLoader

    print(cfg)
    cfg.name = "gewechselt"
    cfg.causalLM_variant = "distilgpt2"
    cfg.target_lang = "fr"

    # load separate data config from ../conf/data/oscar.yaml
    with open("claficle/conf/data/oscar.yaml", "r") as f:
        data_cfg = yaml.load(f, Loader=SafeLoader)
    data_cfg = DictConfig(data_cfg)

    data_cfg.sample_size_mb = 1024

    oscar_fr = OSCARDataModule(data_cfg, cfg.target_lang, 1)
    oscar_fr.prepare_data()
    oscar_fr.setup()

    model = Gewechselt(cfg, oscar_fr.train_dataset)  # noqa: F841


if __name__ == "__main__":
    main()
