from dataclasses import dataclass
import llm.model as model

@dataclass
class AkatheV1Config:
    n_vocabs: int = 12800
    max_seq_len = 20000
    embed_dim: int = 1024
    n_head: int = 8
    n_layers: int = 10
    n_group: int = 4
    att_bias: bool = False
    att_dropout: float = 0.1

    ffn_hidden: int = 4096
    ffn_dropout: float = 0.1

    att_layer: type = model.GroupQueryAttention
    ffn_layer: type = model.FeedForward


