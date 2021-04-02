from logging import getLogger
from typing import Any, Optional, Tuple, Union

import torch
from onmt.decoders import TransformerDecoder
from onmt.modules import Embeddings
from torch import nn
from transformers import BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ...loss import LabelSmoothingLoss
from .decoder import BertSumAbsDecoder

logger = getLogger(__name__)


class BertSumAbsOpenNMTDecoder(BertSumAbsDecoder):
    def __init__(self, config: BertConfig):
        super(BertPreTrainedModel, self).__init__(config)

        self.decoder = TransformerDecoder(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            False,
            "scaled-dot",
            config.hidden_dropout_prob,
            config.attention_probs_dropout_prob,
            Embeddings(
                config.hidden_size,
                config.vocab_size,
                config.pad_token_id,
                position_encoding=True,
            ),
            0,
            False,
            False,
            0,
            0,
        )
        self.generator = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss = LabelSmoothingLoss(config.vocab_size, config.smoothing)
        self.init_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.decoder.embeddings.make_embedding.emb_luts[0]

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.decoder.embeddings.make_embedding.emb_luts[0] = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.generator

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
        if input_ids is None:
            raise ValueError
        if encoder_hidden_states is None:
            raise ValueError
        if encoder_attention_mask is None:
            raise ValueError

        src = encoder_attention_mask.unsqueeze(-1).transpose(0, 1)
        tgt = input_ids.unsqueeze(-1).transpose(0, 1)
        memory_bank = enc_hidden = encoder_hidden_states.transpose(0, 1)

        self.decoder.init_state(src, memory_bank, enc_hidden)
        output, _ = self.decoder(
            tgt, memory_bank, memory_lengths=encoder_attention_mask.sum(axis=1)
        )

        return output
