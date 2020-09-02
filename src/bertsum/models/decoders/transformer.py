from typing import Optional

import torch
from torch import nn

from ..common import PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 768,
                 pad_token_id: int = 0,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 bias: bool = False):
        super(TransformerDecoder, self).__init__()

        self.embeddings = nn.Embedding(vocab_size,
                                       hidden_size,
                                       padding_idx=pad_token_id)
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

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=bias),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self,
                input_ids: torch.Tensor,
                memory: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt = self.embeddings(input_ids)
        tgt = self.pos_emb(tgt)

        tgt = tgt.permute(1, 0, 2).contiguous()
        x = self.decoder(tgt,
                         memory,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        x = x.permute(1, 0, 2).contiguous()

        return self.classifier(x)
