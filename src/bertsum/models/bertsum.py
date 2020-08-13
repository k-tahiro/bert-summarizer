from transformers import AutoModel
import torch
from torch import nn


class BertSumExt(nn.Module):
    def __init__(self,
                 model_type: str,
                 num_encoder_heads: int = 8,
                 num_encoder_layers: int = 2):
        super().__init__()

        # sentence embedding layer
        self.bert = AutoModel.from_pretrained(model_type)

        # inter-sentence contextual embedding layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size,
                                                   nhead=num_encoder_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_encoder_layers)

        # classifier layer
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, src, cls_idxs: list = None):
        if cls_idxs is None:
            cls_idxs = [i for i in range(src.shape[1])]

        x = self.bert(src)[0]

        x = x[:, cls_idxs, :]
        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        x = x.permute(1, 0, 2)
        x = self.linear(x)
        return torch.sigmoid(x)


class BertSumAbs(nn.Module):
    def __init__(self,
                 model_type: str,
                 num_decoder_heads: int = 8,
                 num_decoder_layers: int = 6):
        super(BertSumAbs, self).__init__()

        # encoder
        self.encoder = AutoModel.from_pretrained(model_type)

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.encoder.config.hidden_size,
                                                   nhead=num_decoder_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        x = self.encoder.forward(src)[0]

        tgt = tgt.permute(1, 0)
        memory = x.permute(1, 0, 2)
        return self.decoder.forward(tgt, memory).permute(1, 0, 2)
