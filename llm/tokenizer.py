from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(
    vocab_size=12800,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

