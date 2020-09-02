from logging import getLogger
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import BertConfig, BertModel, EncoderDecoderConfig, EncoderDecoderModel

from .common import PositionalEncoding

logger = getLogger(__name__)


class BertSum(nn.Module):
    def __init__(self, model_type: str):
        super(BertSum, self).__init__()
        self.model_type = model_type


class BertSumExt(BertSum):
    def __init__(self,
                 model_type: str,
                 cls_token_id: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 2,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 bias: bool = True):
        super(BertSumExt, self).__init__(model_type)

        # sentence embedding layer
        self.bert = BertModel.from_pretrained(model_type)
        hidden_size = self.bert.config.hidden_size
        self.cls_token_id = cls_token_id  # cls_token_id is not contained in BertConfig

        # inter-sentence contextual embedding layer
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        if norm is None and eps:
            norm = nn.LayerNorm(hidden_size, eps=eps)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers,
                                             norm=norm)

        # classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self,
                src: Dict[str, torch.Tensor],
                cls_idxs: Union[None, List[List[int]], torch.Tensor] = None):
        if cls_idxs is None:
            cls_idxs = src['input_ids'] == self.cls_token_id

        x = self.bert(**src)[0]
        x = self.pos_emb(x)

        x = x.permute(1, 0, 2).contiguous()
        x = self.encoder(x, src_key_padding_mask=cls_idxs)
        x = x.permute(1, 0, 2).contiguous()

        x = self.classifier(x)
        return x


class BertSumAbs(EncoderDecoderModel):
    def __init__(self,
                 model_type: str,
                 vocab_size: Optional[int] = None,
                 num_hidden_layers: int = 6,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 2048,
                 hidden_act: str = 'gelu',
                 hidden_dropout_prob: float = 0.2,
                 attention_probs_dropout_prob: float = 0.2,
                 layer_norm_eps: float = 1e-6):
        encoder = BertModel.from_pretrained(model_type)
        decoder_config = BertConfig(vocab_size=vocab_size or encoder.config.vocab_size,
                                    hidden_size=encoder.config.hidden_size,
                                    num_hidden_layers=num_hidden_layers,
                                    num_attention_heads=num_attention_heads,
                                    intermediate_size=intermediate_size,
                                    hidden_act=hidden_act,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                    max_position_embeddings=encoder.config.max_position_embeddings,
                                    type_vocab_size=encoder.config.type_vocab_size,
                                    initializer_range=encoder.config.initializer_range,
                                    layer_norm_eps=layer_norm_eps)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config,
                                                                   decoder_config)
        super(BertSumAbs, self).__init__(config=config, encoder=encoder)
