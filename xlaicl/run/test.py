"""Evaluates model on benchmark of tasks"""
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from xlaicl.data.benchmark import BenchmarkDataModule
from xlaicl.models.wrapper import Wrapper


@hydra.main(version_base=None, config_path="../conf", config_name="test")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = Wrapper.load_from_checkpoint(cfg.checkpoint_path)
    # separate benchmarks by language
    benchmarks = []
    langs = ["en", "de", "fr"]
    lang_flags = {lang: cfg[lang] for lang in langs}
    assert any(lang_flags.values()), "At least one of en, de, fr must be True"
    for lang, flag in lang_flags.items():
        if flag is True:
            benchmark = BenchmarkDataModule(config=cfg.benchmark, lang=lang)
            benchmarks.append(benchmark)
    # so that the model knows names and metrics of dataloaders
    model.set_benchmark_metadata(benchmark.get_metadata())
    trainer = pl.Trainer(cfg.trainer)
    for benchmark in benchmarks:
        trainer.test(model, datamodule=benchmark)


if __name__ == "__main__":
    main()
