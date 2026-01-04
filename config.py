from dataclasses import dataclass


@dataclass(frozen=True)
class LongDocConfig:
    base_model_name: str = "distilbert-base-uncased"
    max_chunks_per_doc: int = 32
    hidden_dim: int = 768
    shared_dim: int = 512
    dropout: float = 0.2
    lr: float = 5e-4
