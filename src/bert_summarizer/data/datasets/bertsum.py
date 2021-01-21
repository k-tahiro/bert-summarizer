from functools import partial, reduce
from logging import getLogger
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from ...tokenizers import BertSumTokenizer, BertSumJapaneseTokenizer
from ...utils.bertsum import GreedySelector

logger = getLogger(__name__)


class BertSumDataset(Dataset):
    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[Union[List[str], List[List[str]]]] = None
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
        encoded_src = self._encode(self.tokenizer, src)
        self.data = encoded_src

    @property
    def is_japanese(self):
        return 'bert-base-japanese' in self.model_name

    @property
    def tokenizer(self):
        if self.is_japanese:
            return BertSumJapaneseTokenizer.from_pretrained(self.model_name)
        else:
            return BertSumTokenizer.from_pretrained(self.model_name)

    def _encode(
        self,
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        data: List[str],
        keep_sents: bool = True,
    ) -> List[Dict[str, List[int]]]:
        concat_sents = partial(self._concat_sents, tokenizer)
        truncate = partial(self._truncate, tokenizer)

        sentences = []
        encoded_data = []
        for text in data:
            sents = list(map(str, self.nlp(text).sents))
            if keep_sents:
                sentences.append(sents)
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

        if keep_sents:
            self.sentences = sentences
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
    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[Union[List[str], List[List[str]]]] = None
    ):
        super().__init__(model_name, src, tgt)

        if self.tgt is None:
            return

        tokenizer = self.tokenizer
        bos_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.sep_token_id
        self.gs = GreedySelector(tokenizer)

        generate_tgt = isinstance(self.tgt[0], str)

        for data, sents_src, sents_tgt in zip(self.data, self.sentences, self.tgt):
            if generate_tgt:
                sents_tgt = [
                    str(sent)
                    for sent in self.nlp(sents_tgt).sents
                ]
                sents_tgt = self.gs(sents_src, sents_tgt)
            else:
                sents_tgt = [
                    str(sent)
                    for text in sents_tgt
                    for sent in self.nlp(text).sents
                ]  # to support multiple sentences in one sentence

            data['cls_mask'] = [
                1 * (id_ == bos_token_id)
                for id_ in data['input_ids']
            ]

            data['label'] = []
            index = 0
            for m in data['cls_mask']:
                if m:
                    sent = sents_src[index]
                    data['label'].append(1 * (sent in sents_tgt))
                    index += 1
                else:
                    data['label'].append(0)


class BertSumAbsDataset(BertSumDataset):
    TGT_CLS_TOKEN = '[unused1]'
    TGT_SEP_TOKEN = '[unused2]'
    TGT_ADDITIONAL_SPECIAL_TOKENS = ['[unused3]']

    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[List[str]] = None
    ):
        super().__init__(model_name, src, tgt)

        if tgt is None:
            return

        tokenizer = self.tgt_tokenizer
        encoded_tgt = self._encode(tokenizer, tgt, False)
        for data, e_tgt in zip(self.data, encoded_tgt):
            data.update({
                f'decoder_{k}': v
                for k, v in e_tgt.items()
            })

        # set meta objects
        vocab_size = tokenizer.vocab_size
        for token in [self.TGT_CLS_TOKEN, self.TGT_SEP_TOKEN] + self.TGT_ADDITIONAL_SPECIAL_TOKENS:
            if token not in tokenizer.vocab:
                vocab_size += 1
        self.vocab_size = vocab_size

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
