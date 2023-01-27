import os
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from functools import partial

import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
from torch import Tensor
import torch.nn.functional as F
import torchmetrics.functional as TF
import torch


class BaseModel(pl.LightningModule):
    """
    Abstract class from which to inherit from
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.curr_dataloader_name: Optional[str] = None
        self.metric_to_fn = {
            "f1": TF.classification.multiclass_f1_score,
            "accuracy": TF.classification.multiclass_accuracy,
        }
        self.lm = self.initialize_lm(config)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def initialize_lm(self, config: DictConfig) -> AutoModelForCausalLM:
        """
        LM initialization, allowing for loading weights from fine-tuned checkpoint
        """
        lm = AutoModelForCausalLM.from_pretrained(config.causalLM_variant)
        # optionally load a state dict if using fine-tuned model as starting point
        if config.base_checkpoint is not None:
            ckpt = torch.load(
                os.path.join(config.checkpoint_dir, config.base_checkpoint)
            )
            checkpoint_file_ext = config.base_checkpoint.split(".")[-1]
            if checkpoint_file_ext == "ckpt":
                # need to get rid of 'lm.' prefix for pl_lightning checkpoints
                state_dict = OrderedDict(
                    [(k[3:], v) for k, v in ckpt["state_dict"].items()]
                )
            elif checkpoint_file_ext == "pt":
                state_dict = ckpt
            else:
                raise ValueError("Expected .ckpt or .pt file extension")
            lm.load_state_dict(state_dict)
        return lm

    def post_init(self, **kwargs):
        """
        Optional method that can be called after initialization
        for additional initialization steps. Separate from __init__
        to avoid being called when loading from checkpoint.
        """
        raise NotImplementedError

    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Performs inference, computes relevant metrics and logs

        batch: ((B x O x S), (B x O x S), (B, ))
            B: batch size; O: number of options; S: max sequence length
        batch_idx: int
        dataloader_idx: int describing which dataloader is being used
            to be paired with self.metadata
        """
        torch.set_grad_enabled(False)  # because I don't trust PL to do this
        # parse batch
        concats, tok_type_ids, labels = batch

        preds, losses = self.do_metaicl_inference(concats, tok_type_ids)

        dataloader_config = self.benchmark_metadata["datasets"][dataloader_idx]
        num_classes = dataloader_config["num_classes"]
        for metric in dataloader_config["metrics"]:
            score = self.metric_to_fn[metric](
                preds=preds, target=labels, num_classes=num_classes
            )
            self.log(
                f"{dataloader_config['name']}/test/{metric}",
                score,
                add_dataloader_idx=False,
                on_step=True,
                on_epoch=True,
            )

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        dataloader_config = self.benchmark_metadata["datasets"][dataloader_idx]
        dataloader_name = dataloader_config["name"]
        if self.curr_dataloader_name != dataloader_name:
            self.curr_dataloader_name = dataloader_name
            print(f"Evaluating {dataloader_name}...")

    def compute_loss(self, raw_logits, token_inputs, target_mask):
        """
        Computes the negative log likelihood loss
        so to minimize  - log p(target | context)

        raw_logits: (batch_size * num_options, seq_len, vocab_size)
        token_inputs: (batch_size * num_options, seq_len)
        target_mask: (batch_size, num_options, seq_len)

        returns: loss, of shape (batch_size, num_options)
        """
        # remove last element from logits and first element from inputs to get targets
        # see https://github.com/facebookresearch/MetaICL/issues/9
        raw_logits = raw_logits[:, :-1, :]
        token_inputs = token_inputs[:, 1:]
        target_mask = target_mask[:, :, 1:]

        # flatten further, for cross_entropy; .view() cant be used because not contig
        flattened_logits = raw_logits.reshape(-1, raw_logits.size(-1))
        flattened_targets = token_inputs.reshape(-1)

        # compute loss
        flattened_losses = F.nll_loss(
            flattened_logits, flattened_targets, reduction="none"
        )
        # reshape back to (batch_size, num_options, seq_len-1)
        losses = flattened_losses.view(*target_mask.shape[:-1], -1)

        # apply target mask to loss to only consider neg log likelihood of concat options
        masked_losses = losses * target_mask
        # average over number of targets, getting (batch_size, num_options)
        loss = masked_losses.sum(axis=-1) / target_mask.sum(axis=-1)

        return loss

    def do_metaicl_inference(self, concats, tok_type_ids):
        """
        Runs the model on MetaICL-like input and returns the predictions and losses.

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
        reshaped_concats = concats.view(-1, concats.size(-1))
        reshaped_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        # (batch_size * num_options, seq_len, vocab_size)
        output_logits: Tensor = self.run_causal_model_for_metaicl(
            input_ids=reshaped_concats, attention_mask=reshaped_attention_mask
        )

        # preds where conditioned NLL is lowest
        losses = self.compute_loss(output_logits, reshaped_concats, target_mask)

        preds = losses.argmin(dim=-1)

        return preds, losses

    def set_benchmark_metadata(self, metadata):
        """
        Contains info on the name and metrics for each dataloader in the benchmark
        """
        self.benchmark_metadata = metadata

    def run_causal_model_for_metaicl(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        Runs the causal model on MetaICL-like input.
        To be implemented in inheriting classes.

        input_ids is of shape (pseudo_batch_size, seq_len)
        attention_mask is of shape (pseudo_batch_size, seq_len)

        where pseudo_batch_size is the result of flattening
        e.g. pseudo_batch_size = batch_size * num_options

        Returns `batch_logits`, a tensor of shape
        (pseudo_batch_size, seq_len, vocab_size)
        """
        return self.lm(input_ids=input_ids, attention_mask=attention_mask).logits

    @staticmethod
    def pre_collate(batch: List[Dict], **kwargs) -> List[Dict]:
        """
        Optional pre-collation processing to be passed to a dataloader
        By default we do nothing
        """
        return batch
