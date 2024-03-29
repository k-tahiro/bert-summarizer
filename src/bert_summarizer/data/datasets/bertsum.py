from functools import partial, reduce
from logging import getLogger
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from ...tokenizers import BertSumJapaneseTokenizer, BertSumTokenizer
from ...utils.bertsum import GreedySelector, SentenceSplitter

logger = getLogger(__name__)


class BertSumDataset(Dataset):
    def __init__(
        self,
        model_name: str,
        src: Union[List[str], List[List[str]]],
        tgt: Optional[Union[List[str], List[List[str]]]] = None,
        model_max_length: Optional[int] = None,
    ):
        if tgt is not None and len(src) != len(tgt):
            raise RuntimeError("Different length src v.s. tgt pair is given.")

        # keep inputs
        self.model_name = model_name
        self.src = src
        self.tgt = tgt
        self.model_max_length = model_max_length

        self.sentence_splitter = SentenceSplitter(self.is_japanese)

        # create data
        encoded_src = self._encode(self.tokenizer, src)
        self.data = encoded_src

    @property
    def is_japanese(self) -> bool:
        return "bert-base-japanese" in self.model_name

    @property
    def tokenizer(self) -> BertSumTokenizer:
        tokenizer: BertSumTokenizer
        if self.is_japanese:
            tokenizer = BertSumJapaneseTokenizer.from_pretrained(self.model_name)
        else:
            tokenizer = BertSumTokenizer.from_pretrained(self.model_name)

        if self.model_max_length is not None:
            tokenizer.model_max_length = self.model_max_length
        return tokenizer

    def _encode(
        self,
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        data: Union[List[str], List[List[str]]],
        keep_sents: bool = True,
    ) -> List[Dict[str, List[int]]]:
        concat_sents = partial(self._concat_sents, tokenizer)
        truncate = partial(self._truncate, tokenizer)

        sentences = []
        encoded_data = []
        for text in data:
            if isinstance(text, str):
                sents = self.sentence_splitter(text)
            else:
                sents = text
            if len(sents) == 0:
                continue
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
        output_1: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        input_ids_0 = output_0["input_ids"][1:-1]
        input_ids_1 = output_1["input_ids"][1:-1]
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids_0, input_ids_1)

        sent_sep_token_ids = tokenizer.get_sent_sep_token_ids()
        i = sum(divmod(len(sent_sep_token_ids), 2))
        token_type_ids = (
            output_0["token_type_ids"][:-1]
            + [1] * i
            + [0] * (len(sent_sep_token_ids) - i)
            + output_1["token_type_ids"][1:]
        )

        assert len(input_ids) == len(token_type_ids), (
            "concatenated length mismatch: "
            f"len(input_ids)={len(input_ids)} != len(token_type_ids)={len(token_type_ids)}"
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }

    @staticmethod
    def _truncate(
        tokenizer: Union[BertSumTokenizer, BertSumJapaneseTokenizer],
        encoded_inputs: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        input_ids = encoded_inputs["input_ids"][1:-1]
        num_tokens_to_remove = (
            len(input_ids) - tokenizer.model_max_length + 2
        )  # for metatoken
        input_ids, _, _ = tokenizer.truncate_sequences(
            input_ids,
            num_tokens_to_remove=num_tokens_to_remove,
            truncation_strategy="longest_first",
        )
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        token_type_ids = encoded_inputs["token_type_ids"][: len(input_ids) - 1]
        token_type_ids.append(token_type_ids[-1])

        assert len(input_ids) == len(token_type_ids), (
            "truncated length mismatch: "
            f"len(input_ids)={len(input_ids)} != len(token_type_ids)={len(token_type_ids)}"
        )

        return {"input_ids": input_ids, "token_type_ids": token_type_ids}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]


class BertSumExtDataset(BertSumDataset):
    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[Union[List[str], List[List[str]]]] = None,
        model_max_length: Optional[int] = None,
    ):
        super().__init__(model_name, src, tgt, model_max_length)

        tokenizer = self.tokenizer
        bos_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.sep_token_id
        self.gs = GreedySelector(tokenizer)

        for data in self.data:
            data["cls_mask"] = [1 * (id_ == bos_token_id) for id_ in data["input_ids"]]

        if self.tgt is None:
            return

        valid_data = []
        valid_sentences = []
        valid_tgt = []
        invalid_data = []
        for i in range(len(self.data)):
            data = self.data[i]
            sents_src = self.sentences[i]
            tgt_i = self.tgt[i]

            if isinstance(tgt_i, str):
                sents_tgt = self.sentence_splitter(tgt_i)
                sents_tgt = self.gs(sents_src, sents_tgt)
            else:
                sents_tgt = tgt_i

            data["label"] = []
            index = 0
            for m in data["cls_mask"]:
                if m:
                    sent = sents_src[index]
                    data["label"].append(1 * any(st in sent for st in sents_tgt))
                    index += 1
                else:
                    data["label"].append(0)

            if sum(data["label"]):
                valid_data.append(data)
                valid_sentences.append(sents_src)
                valid_tgt.append(sents_tgt)
            else:
                logger.warning("Invalid src-tgt pair was given. There are no labels.")
                logger.debug(
                    f"src sentences: {sents_src}\n" f"tgt sentences: {sents_tgt}"
                )
                invalid_data.append(
                    {
                        "src": sents_src,
                        "tgt": sents_tgt,
                    }
                )

        self.data = valid_data
        self.sentences = valid_sentences
        self.tgt = valid_tgt
        self.invalid_data = invalid_data


class BertSumAbsDataset(BertSumDataset):
    TGT_CLS_TOKEN = "[unused1]"
    TGT_SEP_TOKEN = "[unused2]"
    TGT_ADDITIONAL_SPECIAL_TOKENS = ["[unused3]"]

    def __init__(
        self,
        model_name: str,
        src: List[str],
        tgt: Optional[List[str]] = None,
        model_max_length: Optional[int] = None,
    ):
        super().__init__(model_name, src, tgt, model_max_length)

        if tgt is None:
            return

        tokenizer = self.tgt_tokenizer
        encoded_tgt = self._encode(tokenizer, tgt, False)
        for data, e_tgt in zip(self.data, encoded_tgt):
            data.update({f"decoder_{k}": v for k, v in e_tgt.items()})

        # set meta objects
        vocab_size = tokenizer.vocab_size
        for token in [
            self.TGT_CLS_TOKEN,
            self.TGT_SEP_TOKEN,
        ] + self.TGT_ADDITIONAL_SPECIAL_TOKENS:
            if token not in tokenizer.vocab:
                vocab_size += 1
        self.vocab_size = vocab_size

    @property
    def tgt_tokenizer(self) -> BertSumTokenizer:
        tokenizer: BertSumTokenizer
        if self.is_japanese:
            tokenizer = BertSumJapaneseTokenizer.from_pretrained(
                self.model_name,
                cls_token=self.TGT_CLS_TOKEN,
                sep_token=self.TGT_SEP_TOKEN,
                additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS,
            )
        else:
            tokenizer = BertSumTokenizer.from_pretrained(
                self.model_name,
                cls_token=self.TGT_CLS_TOKEN,
                sep_token=self.TGT_SEP_TOKEN,
                additional_special_tokens=self.TGT_ADDITIONAL_SPECIAL_TOKENS,
            )
        return tokenizer
