from typing import Dict, List, Union


class BertRouge:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self,
        hyps: Union[str, List[str]],
        refs: Union[str, List[str]]
    ) -> List[Dict[str, Dict[str, float]]]:
        if isinstance(hyps, str):
            hyps, refs = [hyps], [refs]

        scores: List[Dict[str, Dict[str, float]]] = []
        for hyp, ref in zip(hyps, refs):
            tokens_hyp = [token for token in self.tokenize(hyp)]
            tokens_ref = [token for token in self.tokenize(ref)]
            score: Dict[str, Dict[str, float]] = {}

            score['rouge-1'] = self.get_score(tokens_hyp, tokens_ref)
            score['rouge-2'] = self.get_score(tokens_hyp, tokens_ref, 2)

            scores.append(score)

        return scores

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []

        for subtoken in self.tokenizer.tokenize(text):
            if subtoken.startswith('##'):
                tokens[-1] += subtoken[2:]
            else:
                tokens.append(subtoken)

        return tokens

    def get_score(
        self,
        hyp: List[str],
        ref: List[str],
        n: int = 1
    ) -> Dict[str, float]:
        score: Dict[str, float] = {}
        ngram_hyp = {
            tuple(hyp[i:i + n])
            for i in range(len(hyp) - (n - 1))
        }
        ngram_ref = {
            tuple(ref[i:i + n])
            for i in range(len(ref) - (n - 1))
        }
        ngram_inter = ngram_hyp.intersection(ngram_ref)

        count_hyp = len(ngram_hyp)
        count_ref = len(ngram_ref)
        count_inter = len(ngram_inter)

        if count_hyp == 0:
            p = 0.0
        else:
            p = count_inter / count_hyp

        if count_ref == 0:
            r = 0.0
        else:
            r = count_inter / count_ref

        if p == 0 and r == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)

        score['p'] = p
        score['r'] = r
        score['f'] = f

        return score


class GreedySelector:
    def __init__(self, tokenizer, n: int = 3):
        self._rouge = BertRouge(tokenizer)
        self._n = n

    def __call__(
        self,
        sents_src: List[str],
        sents_tgt: List[str]
    ) -> list:
        reference = '\n'.join(sents_tgt)
        max_rouge = 0.0
        selected: List[int] = []
        for _ in range(self._n):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(sents_src)):
                if i in selected:
                    continue
                c = selected + [i]
                hypothesis = '\n'.join([
                    sents_src[idx]
                    for idx in c
                ])
                score = self._rouge(hypothesis, reference)
                rouge_1 = score[0]['rouge-1']['f']
                rouge_2 = score[0]['rouge-2']['f']
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                break
            selected.append(cur_id)
            max_rouge = cur_max_rouge

        return [
            sents_src[i]
            for i in sorted(selected)
        ]
