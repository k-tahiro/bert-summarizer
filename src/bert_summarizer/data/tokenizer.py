from typing import List, Optional

from transformers import BertTokenizer, BertJapaneseTokenizer


class BertSumMixin:
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return cls + token_ids_0 + sep

        sent_sep_token_ids = self.get_sent_sep_token_ids()

        return cls \
            + token_ids_0 \
            + sent_sep_token_ids \
            + token_ids_1 \
            + sep

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        sent_sep_token_ids = self.get_sent_sep_token_ids()
        i = sum(divmod(len(sent_sep_token_ids), 2))
        former_sent_sep_token_ids = sent_sep_token_ids[:i]
        latter_sent_sep_token_ids = sent_sep_token_ids[i:]

        return len(cls + token_ids_0 + former_sent_sep_token_ids) * [0] \
            + len(latter_sent_sep_token_ids + token_ids_1 + sep) * [1]

    def get_sent_sep_token_ids(self) -> List[int]:
        if self.additional_special_tokens_ids:
            sent_sep_token_ids = self.additional_special_tokens_ids
        else:
            sent_sep_token_ids = [self.sep_token_id, self.cls_token_id]

        return sent_sep_token_ids


class BertSumTokenizer(BertSumMixin, BertTokenizer):
    pass


class BertSumJapaneseTokenizer(BertSumMixin, BertJapaneseTokenizer):
    pass
