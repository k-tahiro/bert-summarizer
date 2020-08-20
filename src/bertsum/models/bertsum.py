from logging import getLogger
from typing import Dict, List, Optional, Union

from transformers import AutoModel
import torch
from torch import nn

from .common import PositionalEncoding

logger = getLogger(__name__)


class BertSum(nn.Module):
    def __init__(self, model_type: str):
        super(BertSum, self).__init__()
        self.model_type = model_type


class BertSumExt(BertSum):
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
        super(BertSumExt, self).__init__(model_type)

        # sentence embedding layer
        self.bert = AutoModel.from_pretrained(model_type)
        hidden_size = self.bert.config.hidden_size
        self.cls_token_id = self.bert.config.cls_token_id
        self.pad_token_id = self.bert.config.pad_token_id

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

        x = x.permute(1, 0, 2)
        x = self.encoder(x, src_key_padding_mask=cls_idxs)
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x


class BertSumAbs(BertSum):
    def __init__(self,
                 model_type: str,
                 num_embeddings: int = None,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 vocab_size: Optional[int] = None,
                 bias: bool = False):
        super(BertSumAbs, self).__init__(model_type)

        # encoder
        self.encoder = AutoModel.from_pretrained(model_type)
        hidden_size = self.encoder.config.hidden_size

        # decoder embedding
        self.embeddings = nn.Embedding(num_embeddings or self.encoder.config.vocab_size,
                                       hidden_size,
                                       padding_idx=self.encoder.config.pad_token_id)
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

        # generator
        vocab_size = vocab_size or self.encoder.config.vocab_size
        self.generator = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=bias),
            nn.Softmax(dim=-1)
        )

    def forward(self,
                src: Dict[str, torch.Tensor],
                tgt: Dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        memory = self.encoder(**src)[0]

        # decode
        tgt_key_padding_mask = tgt['attention_mask'] == 0
        memory_key_padding_mask = src['attention_mask'] == 0
        tgt = self.embeddings(tgt['input_ids'])
        tgt = self.pos_emb(tgt)

        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        x = self.decoder(tgt,
                         memory,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        x = x.permute(1, 0, 2)

        x = self.generator(x)
        return x
