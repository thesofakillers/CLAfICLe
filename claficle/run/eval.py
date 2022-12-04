"""Evaluates model on benchmark of tasks"""
import os

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from claficle.data.benchmark import BenchmarkDataModule
from claficle.models.utils import NAME_TO_CLASS
from claficle.models.base import BaseModel


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))
    ModelClass: BaseModel = NAME_TO_CLASS[cfg.model.name]
    print("Loading model from checkpoint...")
    if cfg.model.pl_checkpoint:
        model = ModelClass.load_from_checkpoint(cfg.model.pl_checkpoint)
    else:
        model = ModelClass(cfg.model)
    # get possible additional preprocessing from model and set in benchmark cfg
    cfg.benchmark.extra_proc_fn = cfg.model.extra_proc_fn
    # overwrite benchmark cfg seed with eval cfg seed
    cfg.benchmark.seed = cfg.seed

    # separate benchmarks by language
    bmark_by_lang = {}
    langs = ["en", "de", "fr"]
    lang_flags = {lang: cfg[lang] for lang in langs}
    assert any(lang_flags.values()), "At least one of en, de, fr must be True"
    for lang, flag in lang_flags.items():
        if flag is True:
            print(f"Setting up data for evaluation in {lang}...")
            bmark_by_lang[lang] = BenchmarkDataModule(config=cfg.benchmark, lang=lang)
            bmark_by_lang[lang].prepare_data()
            bmark_by_lang[lang].setup()
    for lang in langs:
        if lang in bmark_by_lang:
            benchmark = bmark_by_lang[lang]
        else:
            continue
        # so that the model knows names and metrics of dataloaders before testing
        model.set_benchmark_metadata(benchmark.get_metadata())
        benchmark.set_tokenizer(model.tokenizer)
        if cfg.benchmark.extra_proc_fn is None:
            benchmark.set_pre_collate_fn(ModelClass.pre_collate)

        # set up pl trainer (tester)
        log_save_dir = os.path.join(
            cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
        )
        logger = pl.loggers.TensorBoardLogger(
            save_dir=log_save_dir,
            name="eval",
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
