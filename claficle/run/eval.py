"""Evaluates model on benchmark of tasks"""
import os

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from claficle.data.benchmark import BenchmarkDataModule
from claficle.models.base import BaseModel
from claficle.utils.general import run_script_preamble


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    # sets seed, parses model name
    model: BaseModel
    model, cfg = run_script_preamble(cfg)

    lang = cfg.lang
    # separate benchmarks by language
    benchmark = BenchmarkDataModule(config=cfg.data, lang=lang)
    benchmark.prepare_data()
    benchmark.setup()
    # so that the model knows names and metrics of dataloaders before testing
    model.set_benchmark_metadata(benchmark.get_metadata())

    # TODO: handle tokenizer
    benchmark.set_tokenizer(model.tokenizer)

    # set up pl trainer (tester)
    log_save_dir = os.path.join(
        cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
    )
    os.makedirs(log_save_dir, exist_ok=True)
    script_host = "slurm" if "SLURM_JOB_ID" in os.environ else "local"
    logger = pl.loggers.WandbLogger(
        save_dir=log_save_dir,
        project="claficle",
        entity="giulio-uva",
        job_type="eval",
        mode="disabled" if cfg.disable_wandb else "online",
        group=script_host,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        log_model=False,  # don't log or upload artifacts
    )
    trainer = pl.Trainer(
        logger=logger,
        enable_progress_bar=cfg.trainer.progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )
    print(f"Evaluating in {lang}...")
    trainer.test(model, datamodule=benchmark)
    print("Done.")


if __name__ == "__main__":
    main()
