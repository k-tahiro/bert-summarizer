from typing import List


class SegmentEmbedding:
    def __init__(self, sep_token_id: int):
        self.sep_token_id = sep_token_id

    def __call__(self, token_ids: List[int]) -> List[int]:
        sep_indices = [-1] + [
            i
            for i, t in enumerate(token_ids)
            if t == self.sep_token_id
        ]
        n_tokens = [
            sep_indices[i] - sep_indices[i - 1]
            for i in range(1, len(sep_indices))
        ]
        segments_ids = []
        for i, s in enumerate(n_tokens):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        return segments_ids
