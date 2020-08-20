from logging import getLogger
from typing import List

from torch.utils.data import Dataset
from transformers import AutoModel

from ..preprocessing.sentencization import Sentencizer
from ..preprocessing.tokenization import BertSumTokenizer

logger = getLogger(__name__)


class BertSumDataset(Dataset):
    TGT_CLS_TOKEN = '[unused1]'
    TGT_SEP_TOKEN = '[unused2]'
    TGT_ADDITIONAL_SPECIAL_TOKENS = ['[unused3]']

    def __init__(self, src: List[str], tgt: List[str], model_type: str):
        if len(src) != len(tgt):
            raise RuntimeError('Different length src v.s. tgt pair is given.')

        self.n_len = len(src)
        self.src = src
        self.tgt = tgt

        self.sentencizer = Sentencizer(model_type)

        model = AutoModel.from_pretrained(model_type)
        model_max_length = model.config.max_position_embeddings
        self.src_tokenizer = BertSumTokenizer.from_pretrained(model_type,
                                                              model_max_length=model_max_length)
        self.tgt_tokenizer = BertSumTokenizer.from_pretrained(
            model_type,
            model_max_length=model_max_length,
            cls_token=self.TGT_CLS_TOKEN,
            sep_token=self.TGT_SEP_TOKEN,
            additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS
        )

        vocab_size = self.tgt_tokenizer.vocab_size
        for token in [self.TGT_CLS_TOKEN, self.TGT_SEP_TOKEN] + self.TGT_ADDITIONAL_SPECIAL_TOKENS:
            if token not in self.tgt_tokenizer.vocab:
                vocab_size += 1
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.n_len

    def __getitem__(self, idx: int):
        src_txt = self.src[idx]
        tgt_txt = self.tgt[idx]
        return self.transform(src_txt, tgt_txt)

    def transform(self, src_txt: str, tgt_txt: str):
        raise NotImplementedError('BertSumDataset cannot be used directly.')

    def _transform(self, src_txt: str, tgt_txt: str):
        # transform src
        src = self.sentencizer(src_txt)
        if len(src) > 1:
            src = [src]
        src = self.src_tokenizer(src,
                                 padding='max_length',
                                 return_tensors='pt')
        src = {
            k: v[0]
            for k, v in src.items()
        }

        # transform tgt
        tgt = self.sentencizer(tgt_txt)
        if len(tgt) > 1:
            tgt = [tgt]
        tgt = self.tgt_tokenizer(tgt,
                                 padding='max_length',
                                 return_tensors='pt')
        tgt = {
            k: v[0]
            for k, v in tgt.items()
        }

        return src, tgt


class BertSumExtDataset(BertSumDataset):
    def transform(self, src_txt: str, tgt_txt: str):
        pass


class BertSumAbsDataset(BertSumDataset):
    def transform(self, src_txt: str, tgt_txt: str):
        return self._transform(src_txt, tgt_txt)
