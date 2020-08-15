from typing import List, Optional

from transformers import BertTokenizer


class BertSumTokenizer(BertTokenizer):
    def build_inputs_with_special_tokens(self,
                                         token_ids_0: List[int],
                                         token_ids_1: Optional[List[int]] = None) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return cls + token_ids_0 + sep

        if self.additional_special_tokens_ids:
            insert_token_ids = self.additional_special_tokens_ids
        else:
            insert_token_ids = sep + cls

        return cls \
            + token_ids_0 \
            + insert_token_ids \
            + token_ids_1 \
            + sep

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0: List[int],
                                             token_ids_1: Optional[List[int]] = None) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        if self.additional_special_tokens_ids:
            insert_token_ids = self.additional_special_tokens_ids
        else:
            insert_token_ids = sep + cls

        i = sum(divmod(len(insert_token_ids), 2))
        former_insert_token_ids = insert_token_ids[:i]
        latter_insert_token_ids = insert_token_ids[i:]

        return len(cls + token_ids_0 + former_insert_token_ids) * [0] \
            + len(latter_insert_token_ids + token_ids_1 + sep) * [1]
