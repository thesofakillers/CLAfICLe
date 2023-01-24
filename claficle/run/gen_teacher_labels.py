import os

import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from claficle.data.oscar import OSCARDataModule
from claficle.models.plain_gpt2 import PlainGPT2


@hydra.main(version_base=None, config_path="../conf/", config_name="gen_teacher_labels")
def main(cfg: DictConfig):
    datasets.disable_caching()
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    wandb.init(
        project="claficle",
        entity="giulio-uva",
        job_type="teacher_labels",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode="disabled" if cfg.disable_wandb else "online",
        group=script_host,
    )
    lang = "en"
    oscar = OSCARDataModule(cfg.data, lang, cfg.seed)
    oscar.prepare_data()

    # instantiate the teacher and set it in the oscar instance
    metaicl_teacher = PlainGPT2(cfg.model)
    oscar.set_teacher(metaicl_teacher)

    oscar.setup("distillation")


if __name__ == "__main__":
    main()
