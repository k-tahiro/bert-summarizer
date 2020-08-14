from logging import getLogger
from typing import List

logger = getLogger(__name__)


class BertSumTextizer:
    def __init__(self,
                 model_type: str,
                 prefix_token: str = '[CLS]',
                 suffix_token: str = '[SEP]',
                 interpolation_token: str = '[SEP] [CLS]'):
        self.seps = self._infer_seps(model_type)
        self.prefix_token = prefix_token
        self.suffix_token = suffix_token
        self.interpolation_token = interpolation_token

    def _infer_seps(self, model_type: str) -> List[str]:
        # FIXME: infer separators depends on model_type
        logger.info('inferring separators...')
        logger.warning('This step returns fixed separators!'
                       'Please fix me...')
        return ['\n', '. ', '。']

    def __call__(self, text: str):
        sents = [text]
        temp = []
        for sep in self.seps:
            for sent in sents:
                sub_sents = sent.split(sep)
                temp.extend(sub_sents)
            sents = temp
            temp = []

        text = f'{self.prefix_token} '
        + f' {self.interpolation_token} '.join(sents)
        + f' {self.suffix_token}'

        return text
