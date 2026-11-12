"""Training pipeline components: tokenizer, 500M-token streaming/memmap datasets, trainer."""

from sciminilm.training.dataset import (
    MemmapTokenDataset,
    PackedTokenStreamingDataset,
    iter_text_files,
    write_token_shard,
)
from sciminilm.training.tokenizer import ScientificTokenizer
from sciminilm.training.trainer import (
    StepLog,
    TrainSummary,
    Trainer,
    build_optimizer,
    cosine_lr_with_warmup,
)

__all__ = [
    "MemmapTokenDataset",
    "PackedTokenStreamingDataset",
    "ScientificTokenizer",
    "StepLog",
    "TrainSummary",
    "Trainer",
    "build_optimizer",
    "cosine_lr_with_warmup",
    "iter_text_files",
    "write_token_shard",
]
