from logging import basicConfig, getLogger

from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bert_summarizer.config import BertSumExtConfig
from bert_summarizer.data import BertSumExtDataset
from bert_summarizer.models import BertSumExt

logger = getLogger(__name__)


def create_dataset(model_name: str = 'bert-base-uncased', n: int = 1000) -> BertSumExtDataset:
    src = [
        'This is the first text for testing. This text contains two sentences.',
        'This is the second text for testing. This text contains two sentences.'
    ] * n
    tgt = [
        ['This is the first text for testing.'],
        ['This is the second text for testing.'],
    ] * n

    return BertSumExtDataset(model_name, src, tgt)


def create_model(model_name: str = 'bert-base-uncased') -> BertSumExt:
    encoder = BertConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=2048,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6
    )
    config = BertSumExtConfig(
        model_name,
        encoder=encoder,
    )
    return BertSumExt(config)


def main():
    basicConfig(level='INFO')

    dataset = create_dataset()
    data_collator = DataCollatorWithPadding(dataset.src_tokenizer)
    model = create_model(dataset.model_name)

    args = TrainingArguments('BertSumExt')
    trainer = Trainer(
        model,
        args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()

    data = next(iter(DataLoader(
        dataset,
        batch_size=1,
        collate_fn=data_collator
    )))
    loss, logits = model(**data)
    logger.info(f'{loss=}')
    logger.info(f'{logits[data["cls_mask"] == 1]=}')
    logger.info(f'{data["labels"][data["cls_mask"] == 1]=}')


if __name__ == '__main__':
    main()
