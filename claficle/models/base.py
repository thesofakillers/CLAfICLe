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
        """
        Performs inference, computes relevant metrics and logs

        batch: ((B x O x S), (B x O x S), (B, ))
            B: batch size; O: number of options; S: max sequence length
        batch_idx: int
        dataloader_idx: int describing which dataloader is being used
            to be paired with self.metadata
        """

        # parse batch
        concats, tok_type_ids, labels = batch

        preds, losses = self.do_inference(concats, tok_type_ids)

        # TODO log
        self.log("test metric", TF.f1_score(preds, labels))

    def compute_loss(self, raw_logits, token_inputs, target_mask):
        """
        Computes the negative log likelihood loss
        so to minimize  - log p(target | context)

        raw_logits: (batch_size * num_options, seq_len, vocab_size)
        token_inputs: (batch_size * num_options, seq_len)
        target_mask: (batch_size, num_options, seq_len)
        """
        # remove last element from logits and first element from inputs to get targets
        # see https://github.com/facebookresearch/MetaICL/issues/9
        raw_logits = raw_logits[:, :-1, :]
        token_inputs = token_inputs[:, 1:]
        target_mask = target_mask[:, 1:]

        # flatten further, for cross_entropy
        flattened_logits = raw_logits.reshape(-1, raw_logits.size(-1))
        flattened_targets = token_inputs.reshape(-1)

        # compute loss
        flattened_losses = F.cross_entropy(
            flattened_logits, flattened_targets, reduction="none"
        )
        # reshape back to (batch_size, num_options, seq_len-1)
        losses = flattened_losses.reshape(*target_mask.shape[:-1], -1)

        # apply target mask to loss to only consider neg log likelihood of concat options
        masked_losses = losses * target_mask
        # average over number of targets, getting (batch_size, num_options)
        loss = masked_losses.sum(axis=-1) / target_mask.sum(axis=-1)

        return loss

    def do_inference(self, concats, tok_type_ids):
        """
        Runs the model on the input and returns the predictions and losses.

        concats: tensor of shape (batch_size, num_options, seq_len)
            consisting in the encoding of the concatenation of the context with each
            of the options
        tok_type_ids: tensor of shape (batch_size, num_options, seq_len)
            0 where the context is, 1 where the option is, 2 where the padding is
        """
        batch_size, num_options, seq_len = concats.shape

        # attention_mask: don't pay attention to padding
        attention_mask = (tok_type_ids != 2).long()
        # target_mask: which ids correspond to concatenated options
        target_mask = (tok_type_ids == 1).long()

        # reshape for hf causal LM, new shape is (batch_size * num_options, seq_len)
        reshaped_concats = concats.reshape(-1, concats.size(-1))
        reshaped_attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))

        # (batch_size * num_options, seq_len, vocab_size)
        output_logits: Tensor = self.run_causal_model(
            input_ids=reshaped_concats, attention_mask=reshaped_attention_mask
        )

        # preds where conditioned NLL is lowest
        losses = self.compute_loss(output_logits, reshaped_concats, target_mask)
        preds = losses.sum(dim=-1).argmin(dim=-1)

        return preds, losses

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
