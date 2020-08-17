from logging import getLogger

from ..models.bertsum import BertSumExt, BertSumAbs

logger = getLogger(__name__)


class BertSumExtTrainer:
    def __init__(self, model: BertSumExt):
        self.model = model


class BertSumAbsTrainer:
    def __init__(self, model: BertSumAbs):
        self.model = model

    def train(self, train_data, train_steps: int):
        for step in range(train_steps):
            for i, batch in enumerate(train_data):
