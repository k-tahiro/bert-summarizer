from functools import partial, reduce
from typing import Dict, List, Optional, Union

import torch
from transformers import BertTokenizer

from ..utils.common import reduce_dict


class InternalBertSumTokenizer(BertTokenizer):
    def build_inputs_with_special_tokens(self,
                                         token_ids_0: List[int],
                                         token_ids_1: Optional[List[int]] = None) -> List[int]:
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

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0: List[int],
                                             token_ids_1: Optional[List[int]] = None) -> List[int]:
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


class BertSumTokenizer:
    INPUT_IDS_KEY = 'input_ids'
    TOKEN_TYPE_IDS_KEY = 'token_type_ids'
    ATTENTION_MASK_KEY = 'attention_mask'
    ENCODED_OUTPUT_KEYS = [INPUT_IDS_KEY,
                           TOKEN_TYPE_IDS_KEY,
                           ATTENTION_MASK_KEY]
    RETURN_TENSORS = {'tf', 'pt', 'np'}
    MODEL_MAX_LENGTH = 512

    def __init__(self, model_type: str, *args, **kwargs):
        self.tokenizer = InternalBertSumTokenizer.from_pretrained(model_type,
                                                                  *args, **kwargs)
        self.max_length = self.tokenizer.max_model_input_sizes.get(model_type,
                                                                   self.MODEL_MAX_LENGTH)

    def __call__(self, text: Union[List[str], List[List[str]]], *args, **kwargs) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        if not (isinstance(text, list) or isinstance(text[0], (str, list))):
            raise RuntimeError(
                'text should be sentences or batch of sentences.')

        padding = kwargs.pop('padding', False)
        truncation = kwargs.pop('truncation', False)
        return_tensors = kwargs.pop('return_tensors', None)
        if return_tensors is not None and return_tensors not in self.RETURN_TENSORS:
            raise ValueError(f"'{return_tensors}' is not a valid TensorType, "
                             f"please select one of {self.RETURN_TENSORS}")

        if isinstance(text[0], str):
            outputs = [self._encode(text, *args, **kwargs)]
        else:
            outputs = [
                self._encode(sents, *args, **kwargs)
                for sents in text
            ]
        output = reduce(partial(reduce_dict, lambda x, y: x + [y]),
                        [{k: [] for k in self.ENCODED_OUTPUT_KEYS}] + outputs)

        if truncation:
            output = self._truncate(output)
        output = self.tokenizer.pad(output,
                                    padding=padding,
                                    max_length=self.max_length)
        output = self._return_tensors(output, return_tensors)
        return output

    def _encode(self, text: List[str], *args, **kwargs):
        outputs = []

        text_len = len(text)
        sent_pairs = sum(divmod(text_len, 2))
        for i in range(sent_pairs):
            index_0 = 2 * i
            index_1 = index_0 + 1
            sent_0 = text[index_0]
            if index_1 < text_len:
                sent_1 = text[index_1]
                output = self.tokenizer(sent_0, sent_1,
                                        padding=False, truncation=False, return_tensors=None,
                                        *args, **kwargs)
            else:
                output = self.tokenizer(sent_0,
                                        padding=False, truncation=False, return_tensors=None,
                                        *args, **kwargs)
            outputs.append(output)

        output = reduce(self._concat_sents, outputs)
        return output

    def _concat_sents(self,
                      output_0: Dict[str, List[int]],
                      output_1: Dict[str, List[int]]) -> Dict[str, List[int]]:
        input_ids_0 = output_0[self.INPUT_IDS_KEY][1:-1]
        input_ids_1 = output_1[self.INPUT_IDS_KEY][1:-1]
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids_0,
                                                                    input_ids_1)

        sent_sep_token_ids = self.tokenizer.get_sent_sep_token_ids()
        i = sum(divmod(len(sent_sep_token_ids), 2))
        token_type_ids = output_0[self.TOKEN_TYPE_IDS_KEY][:-1] \
            + [1] * i \
            + [0] * (len(sent_sep_token_ids) - i) \
            + output_1[self.TOKEN_TYPE_IDS_KEY][1:]

        attention_mask = [1] * len(input_ids)

        return {
            self.INPUT_IDS_KEY: input_ids,
            self.TOKEN_TYPE_IDS_KEY: token_type_ids,
            self.ATTENTION_MASK_KEY: attention_mask,
        }

    def _truncate(self, output: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        batch_size = len(output[self.INPUT_IDS_KEY])
        for i in range(batch_size):
            input_ids = output[self.INPUT_IDS_KEY][i]
            seq_len = len(input_ids)
            if seq_len <= self.max_length:
                continue

            input_ids = input_ids[1:self.max_length - 1]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(
                input_ids)

            output[self.INPUT_IDS_KEY][i] = input_ids
            output[self.TOKEN_TYPE_IDS_KEY][i] = output[self.TOKEN_TYPE_IDS_KEY][i][:self.max_length]
            output[self.ATTENTION_MASK_KEY][i] = [1] * self.max_length

        return output

    @staticmethod
    def _return_tensors(output: Dict[str, List[List[int]]],
                        return_tensors: Optional[str] = None) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        if return_tensors == 'tf':
            # TODO: support tf tensor
            raise RuntimeError('tf tensor does not implemented.')
        elif return_tensors == 'pt':
            return {
                k: torch.tensor(v)
                for k, v in output.items()
            }
        elif return_tensors == 'np':
            # TODO: support np tensor
            raise RuntimeError('np tensor does not implemented.')

        return output
