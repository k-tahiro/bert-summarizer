from logging import basicConfig, getLogger

from torch.utils.data import DataLoader
from transformers import TrainingArguments

from bert_summarizer.config import BertSumAbsConfig
from bert_summarizer.data import BertSumAbsDataset, EncoderDecoderDataCollatorWithPadding
from bert_summarizer.models import BertSumAbs
from bert_summarizer.trainers import EncoderDecoderTrainer

logger = getLogger(__name__)


def create_dataset(model_name: str = 'bert-base-uncased', n: int = 1000) -> BertSumAbsDataset:
    """TODO: Use CNN/DM dataset"""
    src = [
        'This is the first text for testing. This text contains two sentences.',
        'This is the second text for testing. This text contains two sentences.'
    ] * n
    tgt = [
        'First test text',
        'Second test text',
    ] * n

    return BertSumAbsDataset(model_name, src, tgt)


def create_model(dataset: BertSumAbsDataset) -> BertSumAbs:
    config = BertSumAbsConfig(
        dataset.model_name,
        vocab_size=dataset.vocab_size,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act='gelu',
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        layer_norm_eps=1e-6
    )
    return BertSumAbs(config)


def main():
    basicConfig(level='INFO')

    dataset = create_dataset(n=10)
    tokenizer = dataset.tgt_tokenizer

    model = create_model(dataset)

    args = TrainingArguments('BertSumAbs')
    trainer = EncoderDecoderTrainer(
        model,
        args,
        data_collator=EncoderDecoderDataCollatorWithPadding(
            dataset.src_tokenizer,
            decoder_tokenizer=tokenizer,
            generate_labels=True
        ),
        train_dataset=dataset
    )
    trainer.train()

    data = next(iter(DataLoader(
        dataset,
        batch_size=1,
        collate_fn=EncoderDecoderDataCollatorWithPadding(
            dataset.src_tokenizer,
            decoder_tokenizer=tokenizer,
            generate_labels=False
        )
    )))
    outputs = model.generate(
        **data,
        num_beams=5,
        repetition_penalty=1.2,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        length_penalty=0.6,
        no_repeat_ngram_size=3,
    )
    results = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]

    logger.info(f'{results=}')


if __name__ == '__main__':
    main()
