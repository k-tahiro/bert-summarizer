from logging import getLogger
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import (
    BertConfig,
    BertPreTrainedModel,
    BertModel,
    BertLMHeadModel,
    EncoderDecoderModel
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, SequenceClassifierOutput

from .embeddings import PositionalEncoding
from .loss import LabelSmoothingLoss
from ..config import BertSumExtConfig, BertSumAbsConfig

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
            nn.LayerNorm(config.hidden_size, eps=config.encoder.layer_norm_eps)
        )
        self.classifier = nn.Linear(config.hidden_size, 1, bias=True)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        if config.encoder.initializer_range != 0.0:
            for p in self.encoder.layers.parameters():
                p.data.uniform_(
                    -config.encoder.initializer_range,
                    config.encoder.initializer_range
                )
        if config.encoder.xavier_initialization:
            for p in self.encoder.layers.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        cls_mask=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs
        )

        sequence_output = outputs[0].transpose(0, 1)
        cls_output = self.encoder(
            sequence_output,
            src_key_padding_mask=cls_mask.bool() ^ True,
        )
        cls_output = cls_output.transpose(0, 1)

        logits = self.classifier(cls_output).squeeze(2)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())
            loss = (loss * cls_mask.float()).sum(1).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertSumAbsDecoder(BertLMHeadModel):
    def __init__(self, config: BertConfig):
        super(BertPreTrainedModel, self).__init__(config)

        self.embeddings = nn.Sequential(
            nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            ),
            PositionalEncoding(
                config.hidden_size,
                dropout=config.hidden_dropout_prob
            )
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
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.generator = nn.Linear(config.hidden_size, config.vocab_size)

        self.loss = LabelSmoothingLoss(
            config.vocab_size,
            config.smoothing
        )

        self.init_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings[0]

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.embeddings[0] = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.generator

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        tgt = self.embeddings(input_ids).transpose(0, 1)
        memory = encoder_hidden_states.transpose(0, 1)
        tgt_mask = torch.ones(
            (tgt.size(0), tgt.size(0)),
            dtype=torch.bool,
            device=tgt.device,
        ).triu_(1)
        tgt_key_padding_mask = attention_mask ^ True
        memory_key_padding_mask = encoder_attention_mask ^ True

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # transformers style loss calculation
        decoder_outputs = output.transpose(0, 1)
        prediction_scores = self.generator(decoder_outputs)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :] \
                .contiguous()
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


class BertSumAbs(EncoderDecoderModel):
    config_class = BertSumAbsConfig

    def __init__(
        self,
        config: Optional[BertSumAbsConfig] = None,
        encoder: Optional[BertPreTrainedModel] = None,
        decoder: Optional[BertPreTrainedModel] = None
    ):
        if config is not None:
            if encoder is None:
                encoder = BertModel.from_pretrained(
                    config.encoder_model_name_or_path
                )
            if decoder is None:
                decoder = BertSumAbsDecoder(config.decoder)

        super().__init__(config=config, encoder=encoder, decoder=decoder)

        logger.debug(f'self.config={self.config}')

        decoder_embeddings = self.encoder._get_resized_embeddings(
            nn.Embedding.from_pretrained(
                self.encoder.get_input_embeddings().weight,
                freeze=False,
                padding_idx=self.config.encoder.pad_token_id
            ),
            self.config.decoder.vocab_size
        )
        self.decoder.set_input_embeddings(decoder_embeddings)
        self.decoder.tie_weights()
