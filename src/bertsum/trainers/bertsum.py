from logging import getLogger
from typing import Callable, List

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from ..models import BertSumExt, BertSumAbs
from ..preprocessing.sentencization import Sentencizer
from ..preprocessing.tokenization import BertSumTokenizer

logger = getLogger(__name__)


class BertSumDataset(Dataset):
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
            cls_token='[unused0]',
            sep_token='[unused1]',
            additional_special_tokens=['[unused2]']
        )

    def __len__(self) -> int:
        return self.n_len

    def __getitem__(self, idx: int):
        src = self.src[idx]
        tgt = self.tgt[idx]
        return self.transform(src, tgt)

    def transform(self, src: str, tgt: str):
        raise NotImplementedError('BertSumDataset cannot be used directly.')

    def _transform(self, src: str, tgt: str):
        # transform src
        src = self.sentencizer(src)
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
        tgt = self.sentencizer(tgt)
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
    def transform(self, src: str, tgt: str):
        pass


class BertSumAbsDataset(BertSumDataset):
    def transform(self, src: str, tgt: str):
        return self._transform(src, tgt)


class BertSumExtTrainer:
    def __init__(self, model: BertSumExt):
        self.model = model


class BertSumAbsTrainer:
    def __init__(self, model: BertSumAbs):
        self.model = model

    def train(self, train_data, train_steps: int):
        for step in range(train_steps):
            for i, batch in enumerate(train_data):
