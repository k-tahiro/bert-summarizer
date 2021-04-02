from logging import getLogger
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from transformers import BertConfig, BertLMHeadModel, BertPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ...embeddings import PositionalEncoding
from ...loss import LabelSmoothingLoss

logger = getLogger(__name__)


class BertSumAbsDecoder(BertLMHeadModel):
    def __init__(self, config: BertConfig):
        super(BertPreTrainedModel, self).__init__(config)

        self.embeddings = nn.Sequential(
            nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            ),
            PositionalEncoding(config.hidden_size, dropout=config.hidden_dropout_prob),
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.attention_probs_dropout_prob,
                activation=config.hidden_act,
            ),
            config.num_hidden_layers,
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.generator = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss = LabelSmoothingLoss(config.vocab_size, config.smoothing)
        self.init_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings[0]

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embeddings[0] = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.generator

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[Optional[torch.Tensor]], CausalLMOutputWithCrossAttentions]:
        output = self._forward(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )

        # transformers style loss calculation
        decoder_outputs = output.transpose(0, 1)
        prediction_scores = self.generator(decoder_outputs)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            pred = shifted_prediction_scores.view(-1, self.config.vocab_size)
            target = labels.view(-1)

            valid_positions = target.ne(self.config.pad_token_id)
            pred = pred[valid_positions]
            target = target[valid_positions]
            lm_loss = self.loss(pred, target)

        if not return_dict:
            output = (prediction_scores, None, None, None, None)
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def _forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        tgt = self.embeddings(input_ids).transpose(0, 1)
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be passed.")
        memory = encoder_hidden_states.transpose(0, 1)

        # Create masks
        tgt_mask = torch.ones(
            (tgt.size(0), tgt.size(0)),
            dtype=torch.bool,
            device=tgt.device,
        ).triu_(1)
        tgt_key_padding_mask = (
            attention_mask ^ True if attention_mask is not None else None
        )
        memory_key_padding_mask = (
            encoder_attention_mask ^ True
            if encoder_attention_mask is not None
            else None
        )

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output
