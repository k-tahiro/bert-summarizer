from logging import getLogger

from torch.utils.data import DataLoader

from ..datasets.bertsum import BertSumExtDataset, BertSumAbsDataset
from ..models.bertsum import BertSumExt, BertSumAbs

logger = getLogger(__name__)


class BertSumExtTrainer:
    def __init__(self, steps: int = 10000):
        self.steps = steps

    def train(self, model: BertSumExt, dataset: BertSumExtDataset):
        data_loader = DataLoader(dataset)
        for step in range(self.steps):
            for i, batch in enumerate(data_loader):
                pass


class BertSumAbsTrainer:
    def __init__(self, steps: int = 10000):
        self.steps = steps

    def train(self, model: BertSumAbs, dataset: BertSumAbsDataset):
        data_loader = DataLoader(dataset)
        for step in range(self.steps):
            for i, batch in enumerate(data_loader):
                pass
