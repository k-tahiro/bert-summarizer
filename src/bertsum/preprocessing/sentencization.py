from logging import getLogger
from typing import List

logger = getLogger(__name__)


class Sentencizer:
    def __init__(self, model_type: str):
        self.seps = self._infer_seps(model_type)

    def _infer_seps(self, model_type: str) -> List[str]:
        # FIXME: infer separators depends on model_type
        logger.info('inferring separators...')
        logger.warning('This step returns fixed separators!\n'
                       'Please fix me...')
        return ['\n', '. ', '。']

    def __call__(self, text: str) -> List[str]:
        sents = [text]
        temp = []
        for sep in self.seps:
            for sent in sents:
                sub_sents = sent.split(sep)
                temp.extend(sub_sents)
            sents = temp
            temp = []

        return sents
