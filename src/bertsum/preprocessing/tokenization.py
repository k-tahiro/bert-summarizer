from typing import List

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_type: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

    def __call__(self, text: str) -> List[int]:
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
