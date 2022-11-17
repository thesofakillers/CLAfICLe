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
        model = ModelClass.load_from_checkpoint(cfg.model.checkpoint_path)
    else:
        model = ModelClass(cfg.model)
        model.load_non_pl_checkpoint(cfg.model.checkpoint_path)
    # separate benchmarks by language
    bmark_by_lang = {}
    langs = ["en", "de", "fr"]
    lang_flags = {lang: cfg[lang] for lang in langs}
    assert any(lang_flags.values()), "At least one of en, de, fr must be True"
    for lang, flag in lang_flags.items():
        if flag is True:
            benchmark = BenchmarkDataModule(config=cfg.benchmark, lang=lang)
            bmark_by_lang[lang] = benchmark
    trainer = pl.Trainer(cfg.trainer)
    for lang in langs:
        if lang in bmark_by_lang:
            print(f"Setting up data for evaluation in {lang}...")
            benchmark = bmark_by_lang[lang]
        else:
            continue
        # so that the model knows names and metrics of dataloaders before testing
        log_save_dir = os.path.join(cfg.log_dir, cfg.model.name, lang)
        logger = pl.loggers.TensorBoardLogger(
            save_dir=log_save_dir,
            name=f"{cfg.model.name}_{lang}",
        )
        model.set_benchmark_metadata(benchmark.get_metadata())
        benchmark.set_tokenizer(model.tokenizer)
        benchmark.set_pre_collate_fn(ModelClass.pre_collate)

        print(f"Evaluating in {lang}...")
        trainer.test(model, datamodule=benchmark, logger=logger)
    print("Done.")


if __name__ == "__main__":
    main()
