from logging import getLogger
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from ...config import BertSumExtConfig

logger = getLogger(__name__)


class BertSumExt(BertPreTrainedModel):
    config_class = BertSumExtConfig

    def __init__(self, config: BertSumExtConfig):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(config.base_model_name_or_path)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.encoder.num_attention_heads,
                dim_feedforward=config.encoder.intermediate_size,
                dropout=config.encoder.attention_probs_dropout_prob,
                activation=config.encoder.hidden_act,
            ),
            config.encoder.num_hidden_layers,
            nn.LayerNorm(config.hidden_size, eps=config.encoder.layer_norm_eps),
        )
        self.classifier = nn.Linear(config.hidden_size, 1, bias=True)
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

        if config.encoder.initializer_range != 0.0:
            for p in self.encoder.layers.parameters():
                p.data.uniform_(
                    -config.encoder.initializer_range, config.encoder.initializer_range
                )
        if config.encoder.xavier_initialization:
            for p in self.encoder.layers.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        cls_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[Optional[torch.Tensor]], SequenceClassifierOutput]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs,
        )

        sequence_output = outputs[0].transpose(0, 1)
        cls_output = self.encoder(
            sequence_output,
            src_key_padding_mask=cls_mask.bool() ^ True
            if cls_mask is not None
            else None,
        )
        cls_output = cls_output.transpose(0, 1)

        logits = self.classifier(cls_output).squeeze(2)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
            if cls_mask is not None:
                loss = (loss * cls_mask.float()).sum(1).mean()
            else:
                loss = loss.sum(1).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
