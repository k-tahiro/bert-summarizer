from logging import getLogger
from typing import Dict, List, Optional, Union

from onmt.decoders import TransformerDecoder
from onmt.modules import Embeddings
from onmt.utils.loss import LabelSmoothingLoss
import torch
from torch import nn
from transformers import (
    BertConfig,
    BertPreTrainedModel,
    BertModel,
    BertLMHeadModel,
    EncoderDecoderModel
)
from transformers.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead
)

from ..config import BertSumExtConfig, BertSumAbsConfig

logger = getLogger(__name__)


class BertSumExt(BertPreTrainedModel):
    config_class = BertSumExtConfig

    def __init__(self, config: BertSumExtConfig):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(config.base_model_name_or_path)

        self.encoder = nn.Sequential(
            BertEncoder(config),
            BertPooler(config)
        )
        self.cls = BertOnlyMLMHead(config)

    def forward(self,
                src: Dict[str, torch.Tensor],
                cls_idxs: Union[None, List[List[int]], torch.Tensor] = None):
        if cls_idxs is None:
            cls_idxs = src['input_ids'] == self.config.cls_token_id

        x = self.bert(**src)[0]
        x = self.encoder(x, encoder_attention_mask=cls_idxs)
        x = self.cls(x)
        return x


class BertSumAbsDecoder(BertPreTrainedModel):
    # TODO: Replace with transformers decoder.
    # TODO: Control decoder and loss function arguments
    def __init__(self, config: BertConfig):
        super().__init__(config)

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

        self.loss = LabelSmoothingLoss(
            0.1,
            config.vocab_size,
            ignore_index=config.pad_token_id
        )

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
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        self.decoder.init_state(
            kwargs['encoder_input_ids'],
            encoder_hidden_states,
            encoder_hidden_states
        )
        decoder_outputs, _ = self.decoder(
            input_ids.T.unsqueeze(-1),
            encoder_hidden_states,
            memory_lengths=encoder_attention_mask.sum(axis=1)
        )
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

        output = (prediction_scores, None, None)
        return ((lm_loss,) + output) if lm_loss is not None else output


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

        logger.debug(f'{self.config=}')

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
