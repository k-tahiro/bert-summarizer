from functools import partial, reduce
from logging import getLogger
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from ...tokenizers import BertSumTokenizer, BertSumJapaneseTokenizer

logger = getLogger(__name__)


class BertSumDataset(Dataset):
    TGT_CLS_TOKEN = '[unused1]'
    TGT_SEP_TOKEN = '[unused2]'
    TGT_ADDITIONAL_SPECIAL_TOKENS = ['[unused3]']

    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[List[str]] = None
    ):
        if tgt is not None and len(src) != len(tgt):
            raise RuntimeError('Different length src v.s. tgt pair is given.')

        # keep inputs
        self.model_name = model_name
        self.src = src
        self.tgt = tgt

        # load nlp
        if self.is_japanese:
            import spacy
            nlp = spacy.load('ja_ginza')
        else:
            from spacy.lang.en import English
            nlp = English()
            sentencizer = nlp.create_pipe("sentencizer")
            nlp.add_pipe(sentencizer)
        self.nlp = nlp

        # create data
        encoded_src = self._encode(self.src_tokenizer, src)
        self.data = encoded_src
        if tgt is not None:
            encoded_tgt = self._encode(self.tgt_tokenizer, tgt)
            data = []
            for e_src, e_tgt in zip(encoded_src, encoded_tgt):
                sample = e_src
                sample.update({
                    f'decoder_{k}': v
                    for k, v in e_tgt.items()
                })
                data.append(sample)
            self.data = data

        # set meta objects
        tokenizer = self.tgt_tokenizer
        vocab_size = tokenizer.vocab_size
        for token in [self.TGT_CLS_TOKEN, self.TGT_SEP_TOKEN] + self.TGT_ADDITIONAL_SPECIAL_TOKENS:
            if token not in tokenizer.vocab:
                vocab_size += 1
        self.vocab_size = vocab_size

    @property
    def is_japanese(self):
        return 'bert-base-japanese' in self.model_name

    @property
    def src_tokenizer(self):
        if self.is_japanese:
            return BertSumJapaneseTokenizer.from_pretrained(self.model_name)
        else:
            return BertSumTokenizer.from_pretrained(self.model_name)

    @property
    def tgt_tokenizer(self):
        if self.is_japanese:
            return BertSumJapaneseTokenizer.from_pretrained(
                self.model_name,
                cls_token=self.TGT_CLS_TOKEN,
                sep_token=self.TGT_SEP_TOKEN,
                additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS
            )
        else:
            return BertSumTokenizer.from_pretrained(
                self.model_name,
                cls_token=self.TGT_CLS_TOKEN,
                sep_token=self.TGT_SEP_TOKEN,
                additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS
            )

    def _encode(
        self,
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        data: List[str],
    ) -> List[Dict[str, List[int]]]:
        concat_sents = partial(self._concat_sents, tokenizer)
        truncate = partial(self._truncate, tokenizer)

        encoded_data = []
        for text in data:
            sents = list(map(str, self.nlp(text).sents))
            n_sents = len(sents)
            sent_pairs = sum(divmod(n_sents, 2))
            outputs = []
            for i in range(sent_pairs):
                index_0 = 2 * i
                index_1 = index_0 + 1
                sent_0 = sents[index_0]
                if index_1 < n_sents:
                    sent_1 = sents[index_1]
                    output = tokenizer(sent_0, sent_1)
                else:
                    output = tokenizer(sent_0)
                outputs.append(output)

            encoded_data.append(truncate(reduce(concat_sents, outputs)))
        return encoded_data

    @staticmethod
    def _concat_sents(
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        output_0: Dict[str, List[int]],
        output_1: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        input_ids_0 = output_0['input_ids'][1:-1]
        input_ids_1 = output_1['input_ids'][1:-1]
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids_0,
                                                               input_ids_1)

        sent_sep_token_ids = tokenizer.get_sent_sep_token_ids()
        i = sum(divmod(len(sent_sep_token_ids), 2))
        token_type_ids = output_0['token_type_ids'][:-1] \
            + [1] * i \
            + [0] * (len(sent_sep_token_ids) - i) \
            + output_1['token_type_ids'][1:]

        assert len(input_ids) == len(token_type_ids), \
            'concatenated length mismatch: ' \
            f'{len(input_ids)=} != {len(token_type_ids)=}'

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
        }

    @staticmethod
    def _truncate(
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        encoded_inputs: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        input_ids = encoded_inputs['input_ids'][1:-1]
        num_tokens_to_remove = len(input_ids) \
            - tokenizer.model_max_length \
            + 2  # for metatoken
        input_ids, _, _ = tokenizer.truncate_sequences(
            input_ids,
            num_tokens_to_remove=num_tokens_to_remove,
            truncation_strategy='longest_first',
        )
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        token_type_ids = encoded_inputs['token_type_ids'][:len(input_ids) - 1]
        token_type_ids.append(token_type_ids[-1])

        assert len(input_ids) == len(token_type_ids), \
            'truncated length mismatch: ' \
            f'{len(input_ids)=} != {len(token_type_ids)=}'

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]


class BertSumExtDataset(BertSumDataset):
    # TODO: add labels
    pass


class BertSumAbsDataset(BertSumDataset):
    pass
