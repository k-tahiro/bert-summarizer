from logging import getLogger
from typing import Any, Dict, List, Optional

from transformers import AutoModel
import torch
from torch import nn

from .common import PositionalEncoding

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
        self.linear = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self,
                src: torch.Tensor,
                cls_idxs: List[int] = None,
                bert_args: Dict[str, Any] = {},
                encoder_args: Dict[str, Any] = {}):
        if cls_idxs is None:
            cls_idxs = [i for i in range(src.size()[1])]

        x = self.bert(src, **bert_args)[0]

        x = self.pos_emb(x)
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

        # decoder embedding
        self.pad_token_id = self.encoder.config.pad_token_id
        self.embeddings = nn.Embedding(self.encoder.config.vocab_size,
                                       hidden_size,
                                       padding_idx=self.pad_token_id)
        self.pos_emb = PositionalEncoding(dropout,
                                          self.embeddings.embedding_dim)

        # decoder
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
        # create masks
        attention_mask = self._mask(src)
        tgt_key_padding_mask = self._mask(tgt) == 0
        memory_key_padding_mask = attention_mask == 0

        encoder_args.setdefault('attention_mask',
                                attention_mask)
        decoder_args.setdefault('tgt_key_padding_mask',
                                tgt_key_padding_mask)
        decoder_args.setdefault('memory_key_padding_mask',
                                memory_key_padding_mask)

        # encode
        memory = self.encoder(src, **encoder_args)[0]
        memory = memory.permute(1, 0, 2)

        # decode
        tgt = self.embeddings(tgt)
        tgt = self.pos_emb(tgt)
        tgt = tgt.permute(1, 0, 2)

        x = self.decoder(tgt, memory, **decoder_args)
        x = x.permute(1, 0, 2)
        return x

    def _mask(self, token_ids_batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([
            [
                0 if token_id == self.pad_token_id else 1
                for token_id in token_ids
            ]
            for token_ids in token_ids_batch
        ])
