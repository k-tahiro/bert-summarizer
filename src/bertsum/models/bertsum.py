from logging import getLogger
from typing import Any, Dict, List, Optional

from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

logger = getLogger(__name__)


class BertSumExt(nn.Module):
    def __init__(self,
                 model_type: str,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 2,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 bias: bool = True):
        super(BertSumExt, self).__init__()

        # sentence embedding layer
        self.bert = AutoModel.from_pretrained(model_type)
        hidden_size = self.bert.config.hidden_size

        # inter-sentence contextual embedding layer
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
        self.linear = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self,
                src: torch.Tensor,
                cls_idxs: List[int] = None,
                bert_args: Dict[str, Any] = {},
                encoder_args: Dict[str, Any] = {}):
        if cls_idxs is None:
            cls_idxs = [i for i in range(src.size()[1])]

        x = self.bert(src, **bert_args)[0]

        x = x[:, cls_idxs, :]
        x = x.permute(1, 0, 2)
        x = self.encoder(x, **encoder_args)

        x = x.permute(1, 0, 2)
        x = self.linear(x)
        return torch.sigmoid(x)


class BertSumAbs(nn.Module):
    def __init__(self,
                 model_type: str,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6):
        super(BertSumAbs, self).__init__()

        # encoder
        self.encoder = AutoModel.from_pretrained(model_type)
        hidden_size = self.encoder.config.hidden_size

        # decoder
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        pad_token = tokenizer.special_tokens_map_extended['pad_token']
        padding_idx = tokenizer.vocab[pad_token]
        self.embeddings = nn.Embedding(self.encoder.config.vocab_size,
                                       hidden_size,
                                       padding_idx=padding_idx)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        if norm is None and eps:
            norm = nn.LayerNorm(hidden_size, eps=eps)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_layers,
                                             norm=norm)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                encoder_args: Dict[str, Any] = {},
                decoder_args: Dict[str, Any] = {}) -> torch.Tensor:
        # encode
        memory = self.encoder(src, **encoder_args)[0]
        memory = memory.permute(1, 0, 2)

        # decode
        tgt = self.embeddings(tgt)
        tgt = tgt.permute(1, 0, 2)
        x = self.decoder(tgt, memory, **decoder_args)
        x = x.permute(1, 0, 2)
        return x
