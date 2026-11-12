"""Byte-level BPE scientific text tokenizer with zero external dependencies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


class ScientificTokenizer:
    """Lossless byte-level BPE tokenizer tailored for scientific text (LaTeX, math, code).

    Token IDs 0..3 are reserved for <pad>, <bos>, <eos>, <unk>.
    Token IDs 4..259 represent raw UTF-8 bytes 0..255.
    Token IDs >= 260 represent learned BPE merge tokens.
    """

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        self.pad_id = self.special_tokens["<pad>"]
        self.bos_id = self.special_tokens["<bos>"]
        self.eos_id = self.special_tokens["<eos>"]
        self.unk_id = self.special_tokens["<unk>"]

        self.byte_offset = len(self.special_tokens)
        self.merges: List[Tuple[int, int]] = list(merges) if merges else []
        self.merge_ranks: Dict[Tuple[int, int], int] = {
            pair: self.byte_offset + 256 + idx for idx, pair in enumerate(self.merges)
        }
        self._id_to_bytes: Dict[int, bytes] = {
            self.byte_offset + b: bytes([b]) for b in range(256)
        }
        for pair, new_id in self.merge_ranks.items():
            self._id_to_bytes[new_id] = self._id_to_bytes[pair[0]] + self._id_to_bytes[pair[1]]

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 1024,
        max_docs: Optional[int] = None,
    ) -> "ScientificTokenizer":
        """Train byte-level BPE merges from an iterator of scientific documents."""
        min_vocab = len(SPECIAL_TOKENS) + 256
        if vocab_size < min_vocab:
            raise ValueError(f"vocab_size must be >= {min_vocab}, got {vocab_size}")

        byte_offset = len(SPECIAL_TOKENS)
        corpus_ids: List[List[int]] = []
        for idx, text in enumerate(texts):
            if max_docs is not None and idx >= max_docs:
                break
            raw = text.encode("utf-8")
            if raw:
                corpus_ids.append([byte_offset + b for b in raw])

        n_merges = vocab_size - min_vocab
        merges: List[Tuple[int, int]] = []

        for step in range(n_merges):
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for seq in corpus_ids:
                if len(seq) < 2:
                    continue
                for a, b in zip(seq[:-1], seq[1:]):
                    pair_counts[(a, b)] += 1

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            new_id = min_vocab + step
            merges.append(best_pair)

            # Apply merge across corpus
            updated_corpus: List[List[int]] = []
            p0, p1 = best_pair
            for seq in corpus_ids:
                merged: List[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i + 1 < n and seq[i] == p0 and seq[i + 1] == p1:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                updated_corpus.append(merged)
            corpus_ids = updated_corpus

        return cls(merges=merges)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode string into token IDs using greedy BPE merge application."""
        raw = text.encode("utf-8")
        tokens = [self.byte_offset + b for b in raw]

        if self.merges and len(tokens) >= 2:
            while len(tokens) >= 2:
                # Find the earliest-learned merge present in tokens
                best_pair: Optional[Tuple[int, int]] = None
                best_new_id = float("inf")
                for a, b in zip(tokens[:-1], tokens[1:]):
                    pair = (a, b)
                    if pair in self.merge_ranks and self.merge_ranks[pair] < best_new_id:
                        best_new_id = self.merge_ranks[pair]
                        best_pair = pair
                if best_pair is None:
                    break
                p0, p1 = best_pair
                new_id = int(best_new_id)
                merged: List[int] = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i + 1 < n and tokens[i] == p0 and tokens[i + 1] == p1:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(tokens[i])
                        i += 1
                tokens = merged

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into a UTF-8 string."""
        chunks: List[bytes] = []
        for tid in ids:
            tid_int = int(tid)
            if tid_int in self.id_to_special:
                if not skip_special_tokens:
                    chunks.append(self.id_to_special[tid_int].encode("utf-8"))
                continue
            byte_seq = self._id_to_bytes.get(tid_int)
            if byte_seq is not None:
                chunks.append(byte_seq)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save tokenizer merges to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScientificTokenizer":
        """Load tokenizer merges from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(int(a), int(b)) for a, b in payload.get("merges", [])]
        return cls(merges=merges)
