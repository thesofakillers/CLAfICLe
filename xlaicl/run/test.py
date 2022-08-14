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
    benchmark = BenchmarkDataModule(cfg.benchmark)
    # so that the model knows names and metrics of dataloaders
    model.set_benchmark_metadata(benchmark.metadata)
    trainer = pl.Trainer(cfg.trainer)
    trainer.test(model, datamodule=benchmark)


if __name__ == "__main__":
    main()
