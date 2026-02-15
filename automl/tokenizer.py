"""Tokenizer design phase powered by AST-generated code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TokenizerDesign:
    vocabulary: List[str]

    def encode(self, text: str) -> List[int]:
        tokens = text.replace("(", " ( ").replace(")", " ) ").split()
        return [self.vocabulary.index(tok) if tok in self.vocabulary else -1 for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        pieces = [self.vocabulary[idx] if 0 <= idx < len(self.vocabulary) else "<unk>" for idx in ids]
        return " ".join(pieces)
