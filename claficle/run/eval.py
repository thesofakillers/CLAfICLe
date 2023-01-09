"""Evaluates model on benchmark of tasks"""
import os

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from claficle.data.benchmark import BenchmarkDataModule
from claficle.models.base import BaseModel
from claficle.utils.general import run_script_preamble


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    # sets seed, parses model name
    model: BaseModel
    model, cfg = run_script_preamble(cfg)

    # separate benchmarks by language
    bmark_by_lang = {}
    langs = ["en", "de", "fr"]
    lang_flags = {lang: cfg[lang] for lang in langs}
    assert any(lang_flags.values()), "At least one of en, de, fr must be True"
    for lang, flag in lang_flags.items():
        if flag is True:
            print(f"Setting up data for evaluation in {lang}...")
            bmark_by_lang[lang] = BenchmarkDataModule(config=cfg.data, lang=lang)
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
        if cfg.data.extra_proc_fn is None:
            benchmark.set_pre_collate_fn(model.pre_collate)

        # set up pl trainer (tester)
        log_save_dir = os.path.join(
            cfg.trainer.log_dir, cfg.model.name, f"seed_{cfg.seed}", lang
        )
        os.makedirs(log_save_dir, exist_ok=True)
        logger = pl.loggers.WandbLogger(
            save_dir=log_save_dir,
            job_type="eval",
            project="claficle",
            entity="giulio-uva",
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
