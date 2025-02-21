from dataclasses import dataclass
import llm.model as model

@dataclass
class AkatheV1Config:
    n_vocabs: int = 12800
    max_seq_len = 20000
    embed_dim: int = 1024
    q_head: int = 8
    kv_head: int = 2
    n_layers: int = 10
    att_bias: bool = False
    att_dropout: float = 0.1

    ffn_hidden: int = 4096
    ffn_dropout: float = 0.1

    att_layer: type = model.GroupQueryAttention
    ffn_layer: type = model.FeedForward


@dataclass
class AkatheTrainConfig:
    name: str = "AkatheV1"
    train_batch_size: int = 8
    val_batch_size: int = 8
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    checkpoint_steps: int = 1000
