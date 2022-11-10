import pytorch_lightning as pl
from omegaconf import DictConfig
from claficle.models.utils import NAME_TO_CLASS


class BaseModel(pl.LightningModule):
    # todo
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self._set_evaluand()

    def _set_evaluand(self):
        """returns an instance the model we wish to evaluate"""
        Evaluand_Class: pl.LightningModule = NAME_TO_CLASS[self.hparams.name]
        self.evaluand = Evaluand_Class.load_from_checkpoint(
            self.hparams.checkpoint_path
        )

    def test_step(self, batch, batch_idx, dataloader_idx):
        # todo
        # get data (batch)
        # pre-collate processing (defined in evaluand)
        # collate
        # get log-likelihood from evaluand for each option completing the input
        # evaluate (max log-likelihood = prediction -> compare to label)
        # log
        pass

    def set_benchmark_metadata(self, metadata):
        """Contains info on the name and metrics for each dataloader in the benchmark"""
        self.benchmark_metadata = metadata
