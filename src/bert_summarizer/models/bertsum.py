from logging import getLogger
from typing import Dict, List, Optional

from onmt.decoders import TransformerDecoder
from onmt.modules import Embeddings
from onmt.utils.loss import LabelSmoothingLoss
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
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, SequenceClassifierOutput

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
    # TODO: Replace with transformers decoder.
    # TODO: Control decoder and loss function arguments
    def __init__(self, config: BertConfig):
        super(BertPreTrainedModel, self).__init__(config)

        self.decoder = TransformerDecoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            False,
            'scaled-dot',
            config.hidden_dropout_prob,
            config.attention_probs_dropout_prob,
            Embeddings(
                config.hidden_size,
                config.vocab_size,
                config.pad_token_id,
                position_encoding=True
            ),
            0,
            False,
            False,
            0,
            0
        )

        self.generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()
        self.generator[0].bias.requires_grad = False

        self.loss = LabelSmoothingLoss(
            0.1,
            config.vocab_size,
            ignore_index=config.pad_token_id
        )

        self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.embeddings.make_embedding.emb_luts[0]

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.decoder.embeddings.make_embedding.emb_luts[0] = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.generator[0]

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
        logger.debug(f'kwargs={kwargs}')

        # Use OpenNMT-py TransformerDecoder
        src = kwargs['encoder_input_ids'].unsqueeze(-1).transpose(0, 1)
        tgt = input_ids.unsqueeze(-1).transpose(0, 1)
        memory_bank = enc_hidden = encoder_hidden_states.transpose(0, 1)

        self.decoder.init_state(src, memory_bank, enc_hidden)
        dec_outs, attns = self.decoder(
            tgt,
            memory_bank,
            memory_lengths=encoder_attention_mask.sum(axis=1)
        )

        # transformers style loss calculation
        decoder_outputs = dec_outs.transpose(0, 1)
        prediction_scores = self.generator[0](decoder_outputs)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :] \
                .contiguous()
            labels = labels[:, 1:].contiguous()

            output = self.generator[1](
                shifted_prediction_scores
            ).view(-1, self.config.vocab_size)
            target = labels.view(-1)

            normalization = target.ne(self.config.pad_token_id).sum().item()

            lm_loss = self.loss(output, target).div(float(normalization))

        if not return_dict:
            output = (prediction_scores, None, None, attns['std'], None)
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=attns['std'],
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

        enc_emb_weight = self.encoder.get_input_embeddings().weight.clone().detach()
        dec_emb_weight = self.decoder.get_input_embeddings().weight.clone().detach()
        enc_row, enc_col = enc_emb_weight.size()
        dec_emb_weight[:enc_row, :enc_col] = enc_emb_weight
        input_embeddings = nn.Embedding.from_pretrained(
            dec_emb_weight,
            freeze=False,
            padding_idx=self.config.decoder.pad_token_id
        )
        self.decoder.set_input_embeddings(input_embeddings)
        self.decoder.tie_weights()

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, encoder_outputs=None, **kwargs):
        input_dict = super().prepare_inputs_for_generation(
            input_ids,
            past,
            attention_mask,
            encoder_outputs,
            **kwargs
        )

        if 'decoder_encoder_input_ids' in kwargs:
            input_dict['decoder_encoder_input_ids'] = kwargs['decoder_encoder_input_ids']

        return input_dict
