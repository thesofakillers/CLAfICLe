from typing import Dict, List, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer
from torch import Tensor
import torch.nn.functional as F
import torchmetrics.functional as TF


class BaseModel(pl.LightningModule):
    # todo
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.truncation_side = "left"
        # see https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        # parse batch
        concats, tok_type_ids, labels = batch
        batch_size, num_options, seq_len = concats.shape

        # attention_mask: don't pay attention to padding
        attention_mask = (tok_type_ids != 2).long()

        # reshape for hf causal LM, new shape is (batch_size * num_options, seq_len)
        reshaped_concats = concats.reshape(-1, concats.size(-1))
        reshaped_attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))

        # (batch_size * num_options, seq_len, vocab_size)
        output_logits: Tensor = self.run_causal_model(
            input_ids=reshaped_concats, attention_mask=reshaped_attention_mask
        )

        # remove last element from logits and first element from inputs to get targets
        # see https://github.com/facebookresearch/MetaICL/issues/9
        output_logits = output_logits[:, :-1, :]
        reshaped_concats = reshaped_concats[:, 1:]

        # flatten further, for cross_entropy
        flattened_logits = output_logits.reshape(-1, output_logits.size(-1))
        flattened_targets = reshaped_concats.reshape(-1)

        # compute loss
        flattened_losses = F.cross_entropy(
            flattened_logits, flattened_targets, reduction="none"
        )
        # reshape back to (batch_size, num_options, seq_len-1)
        losses = flattened_losses.reshape(batch_size, num_options, -1)

        # target_mask: which ids correspond to concatenated options
        target_mask = (tok_type_ids == 1).long()
        # just like before, remove the first element
        target_mask = target_mask[:, 1:]

        # apply target mask to loss to only consider neg log likelihood of concat options
        masked_losses = losses * target_mask

        # we can finally get our predictions (batch_size,)
        preds = masked_losses.sum(dim=-1).argmin(dim=-1)

        # TODO log
        self.log("test metric", TF.f1_score(preds, labels, num_classes=num_options))
        pass

    def set_benchmark_metadata(self, metadata):
        """
        Contains info on the name and metrics for each dataloader in the benchmark
        """
        self.benchmark_metadata = metadata

    @staticmethod
    def pre_collate(batch: List[Dict]) -> List[Dict]:
        """
        Optional pre-collation processing to be passed to a dataloader
        By default we do nothing
        """
        return batch

    def run_causal_model(self, input_ids, attention_mask):
        """
        Runs the causal model on the input.
        To be implemented in inheriting classes.
        """
        raise NotImplementedError
