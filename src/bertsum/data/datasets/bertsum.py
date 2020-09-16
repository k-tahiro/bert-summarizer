from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from ..tokenizer import BertSumTokenizer

logger = getLogger(__name__)


class BertSumDataset(Dataset):
    TGT_CLS_TOKEN = '[unused1]'
    TGT_SEP_TOKEN = '[unused2]'
    TGT_ADDITIONAL_SPECIAL_TOKENS = ['[unused3]']

    def __init__(
        self,
        model_type: str,
        src: List[str],
        tgt: Optional[List[str]] = None,
        return_tensors: Optional[str] = None
    ):
        if tgt is not None and len(src) != len(tgt):
            raise RuntimeError('Different length src v.s. tgt pair is given.')

        self.n_len = len(src)
        self.src = src
        self.tgt = tgt
        self.return_tensors = return_tensors

        self._init_nlp(model_type)

        self.src_tokenizer = BertSumTokenizer(model_type)
        self.tgt_tokenizer = BertSumTokenizer(
            model_type,
            cls_token=self.TGT_CLS_TOKEN,
            sep_token=self.TGT_SEP_TOKEN,
            additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS
        )

        vocab_size = self.tgt_tokenizer.tokenizer.vocab_size
        for token in [self.TGT_CLS_TOKEN, self.TGT_SEP_TOKEN] + self.TGT_ADDITIONAL_SPECIAL_TOKENS:
            if token not in self.tgt_tokenizer.tokenizer.vocab:
                vocab_size += 1
        self.vocab_size = vocab_size

    def _init_nlp(self, model_type: str):
        if 'japanese' in model_type:
            import spacy
            nlp = spacy.load('ja_ginza')
        else:
            from spacy.lang.en import English
            nlp = English()
            sentencizer = nlp.create_pipe("sentencizer")
            nlp.add_pipe(sentencizer)

        self.nlp = nlp

    def __len__(self) -> int:
        return self.n_len

    def __getitem__(self, idx: int):
        src_txt = self.src[idx]
        tgt_txt = self.tgt[idx] if self.tgt is not None else None
        return self.transform(src_txt, tgt_txt)

    def transform(self, src_txt: str, tgt_txt: Optional[str] = None):
        raise NotImplementedError('BertSumDataset cannot be used directly.')

    def _transform(
        self,
        src_txt: str,
        tgt_txt: Optional[str] = None
    ) -> Tuple[Dict[str, Union[List[int], torch.Tensor]], Optional[Dict[str, Union[List[int], torch.Tensor]]]]:
        # transform src
        src = list(map(str, self.nlp(src_txt).sents))

        src = self.src_tokenizer(src, **self.tokenize_args)
        src = {
            k: v[0]
            for k, v in src.items()
        }

        # transform tgt
        if tgt_txt is None:
            tgt = None
        else:
            tgt = list(map(str, self.nlp(tgt_txt).sents))
            tgt = self.tgt_tokenizer(tgt, **self.tokenize_args)
            tgt = {
                k: v[0]
                for k, v in tgt.items()
            }

        return src, tgt

    @property
    def tokenize_args(self):
        tokenize_args = dict(truncation=True)
        if self.return_tensors is not None:
            tokenize_args.update(dict(padding='max_length',
                                      return_tensors=self.return_tensors))
        return tokenize_args

    @property
    def tokenizer(self):
        return self.tgt_tokenizer.tokenizer

    @property
    def translator(self):
        def translator(token_ids: List[int]) -> str:
            return self.tokenizer.decode(token_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)

        return translator


class BertSumExtDataset(BertSumDataset):
    def transform(self, src_txt: str, tgt_txt: Optional[str] = None):
        pass


class BertSumAbsDataset(BertSumDataset):
    def transform(self, src_txt: str, tgt_txt: Optional[str] = None) -> Dict[str, Union[List[int], torch.Tensor]]:
        src, tgt = self._transform(src_txt, tgt_txt)
        data = src

        if tgt is not None:
            data.update({
                f'decoder_{k}': v
                for k, v in tgt.items()
            })

        return data
