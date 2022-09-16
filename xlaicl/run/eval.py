"""Evaluates model on benchmark of tasks"""
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from xlaicl.data.benchmark import BenchmarkDataModule
from xlaicl.models.wrapper import Wrapper


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = Wrapper.load_from_checkpoint(cfg.checkpoint_path)
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
    for _lang, benchmark in bmark_by_lang.items():
        # so that the model knows names and metrics of dataloaders before testing
        model.set_benchmark_metadata(benchmark.get_metadata())
        trainer.test(model, datamodule=benchmark)


if __name__ == "__main__":
    main()
