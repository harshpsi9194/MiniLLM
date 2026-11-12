"""Streaming and memory-mapped datasets for 500M-token scientific-text training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def iter_text_files(
    paths: Sequence[Union[str, Path]],
    text_field: str = "text",
) -> Iterator[str]:
    """Stream text documents from .jsonl or plain .txt files."""
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get(text_field, "")
                    if text:
                        yield text
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                yield text


def write_token_shard(
    texts: Iterable[str],
    tokenizer,
    output_path: Union[str, Path],
    dtype: np.dtype = np.uint16,
    add_eos: bool = True,
    max_tokens: Optional[int] = None,
) -> int:
    """Pre-tokenize a stream of scientific texts into a flat numpy memmap binary shard."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    token_buf: List[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_eos=add_eos)
        token_buf.extend(ids)
        if max_tokens is not None and len(token_buf) >= max_tokens:
            token_buf = token_buf[:max_tokens]
            break

    arr = np.asarray(token_buf, dtype=dtype)
    arr.tofile(out_path)
    return int(arr.size)


class PackedTokenStreamingDataset(IterableDataset):
    """Streaming iterable dataset that packs tokenized scientific documents into `seq_len` blocks.

    Designed for large-scale streaming corpora (e.g., 500M tokens of arXiv/PeS2o/PubMed)
    with zero padding waste, multi-worker sharding, and optional token-budget capping.
    """

    def __init__(
        self,
        text_stream_factory: Callable[[], Iterable[str]],
        tokenizer,
        seq_len: int = 2048,
        add_eos: bool = True,
        shuffle_buffer_size: int = 0,
        max_tokens: Optional[int] = 500_000_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.text_stream_factory = text_stream_factory
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_tokens = max_tokens
        self.seed = seed

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        rng = random.Random(self.seed + worker_id)
        doc_stream = self.text_stream_factory()

        def _sharded_docs() -> Iterator[str]:
            for idx, doc in enumerate(doc_stream):
                if idx % num_workers == worker_id:
                    yield doc

        def _shuffled_docs() -> Iterator[str]:
            if self.shuffle_buffer_size <= 1:
                yield from _sharded_docs()
                return
            buf: List[str] = []
            for doc in _sharded_docs():
                buf.append(doc)
                if len(buf) >= self.shuffle_buffer_size:
                    pick = rng.randrange(len(buf))
                    yield buf.pop(pick)
            while buf:
                pick = rng.randrange(len(buf))
                yield buf.pop(pick)

        token_buffer: List[int] = []
        emitted_tokens = 0
        worker_token_limit = (
            None if self.max_tokens is None else self.max_tokens // num_workers
        )

        for doc in _shuffled_docs():
            ids = self.tokenizer.encode(doc, add_eos=self.add_eos)
            token_buffer.extend(ids)
            while len(token_buffer) >= self.seq_len:
                if worker_token_limit is not None and emitted_tokens + self.seq_len > worker_token_limit:
                    return
                chunk = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]
                emitted_tokens += self.seq_len
                yield torch.tensor(chunk, dtype=torch.long)


class MemmapTokenDataset(Dataset):
    """Random-access dataset backed by a pre-tokenized flat binary shard (.bin)."""

    def __init__(
        self,
        bin_path: Union[str, Path],
        seq_len: int = 2048,
        dtype: np.dtype = np.uint16,
    ) -> None:
        super().__init__()
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        file_bytes = self.bin_path.stat().st_size
        self.total_tokens = file_bytes // self.dtype.itemsize
        self.num_samples = self.total_tokens // self.seq_len
        if self.num_samples == 0:
            raise ValueError(
                f"Shard {bin_path} has {self.total_tokens} tokens, fewer than seq_len={seq_len}"
            )
        self._memmap: Optional[np.memmap] = None

    def _get_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        return self._memmap

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        mm = self._get_memmap()
        start = idx * self.seq_len
        chunk = np.asarray(mm[start : start + self.seq_len], dtype=np.int64)
        return torch.from_numpy(chunk)
