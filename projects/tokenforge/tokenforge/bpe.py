"""The BPE engine — training, encoding, decoding, special tokens, stats.

Byte-level base vocabulary (ids 0-255 are raw bytes); learned merges occupy
ids 256+. A merge is a pair (a, b) with a rank: to encode, repeatedly merge
the pair with the LOWEST rank present; to decode, expand ids in rank order.
Everything is deterministic — no randomness anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import TokenizerError

_BASE = 256
_UNPRINTABLE = "·"


class Tokenizer:
    """A trained byte-pair-encoding tokenizer."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}   # pair -> rank (0-based)
        self.special: dict[str, int] = {}              # name -> id
        self._trained_on = 0

    # ---------- training ----------
    def train(self, text: str, vocab_size: int) -> "Tokenizer":
        if vocab_size < _BASE + 1:
            raise TokenizerError(f"vocab_size must be >= {_BASE + 1}, got {vocab_size}")
        ids = list(text.encode("utf-8"))
        self._trained_on = len(ids)
        self.merges = {}
        for rank in range(vocab_size - _BASE):
            counts: dict[tuple[int, int], int] = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            # most frequent pair wins; ties break by first-seen order (deterministic)
            best, best_count = None, 1
            for pair, count in counts.items():
                if count > best_count:
                    best, best_count = pair, count
            if best is None:
                break  # nothing repeats — further merges are meaningless
            new_id = _BASE + rank
            ids = self._merge_ids(ids, best, new_id)
            self.merges[best] = rank
        return self

    @staticmethod
    def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ---------- encoding / decoding ----------
    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if allow_special and self.special:
            return self._encode_with_specials(text)
        return self._encode_raw(text)

    def _encode_with_specials(self, text: str) -> list[int]:
        names = sorted(self.special, key=len, reverse=True)  # longest first
        ids: list[int] = []
        buf = ""
        i = 0
        while i < len(text):
            hit = next((n for n in names if text.startswith(n, i)), None)
            if hit is not None:
                ids.extend(self._encode_raw(buf))
                ids.append(self.special[hit])
                buf = ""
                i += len(hit)
            else:
                buf += text[i]
                i += 1
        ids.extend(self._encode_raw(buf))
        return ids

    def _encode_raw(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            ids = self._merge_ids(ids, best_pair, _BASE + best_rank)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        id_to_pair = {_BASE + rank: pair for pair, rank in self.merges.items()}
        special_by_id = {sid: name for name, sid in self.special.items()}
        for i in ids:
            if not isinstance(i, int) or i < 0:
                raise TokenizerError(f"invalid token id: {i!r}")
            if i in special_by_id:
                out.extend(special_by_id[i].encode("utf-8"))
            elif i < _BASE:
                out.append(i)
            else:
                out.extend(bytes(self._expand(i, id_to_pair)))
        return out.decode("utf-8", errors="replace")

    def _expand(self, token: int, id_to_pair: dict[int, tuple[int, int]]) -> list[int]:
        seq = [token]
        while True:
            new: list[int] = []
            grew = False
            for t in seq:
                if t >= _BASE and t in id_to_pair:
                    new.extend(id_to_pair[t])
                    grew = True
                else:
                    new.append(t)
            seq = new
            if not grew:
                return seq

    # ---------- special tokens ----------
    def register_special(self, name: str) -> int:
        if not name:
            raise TokenizerError("special token name must be non-empty")
        if name in self.special:
            return self.special[name]
        sid = _BASE + len(self.merges) + len(self.special)
        self.special[name] = sid
        return sid

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "version": 1,
            "merges": [list(p) for p, _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
            "special": self.special,
            "trained_on": self._trained_on,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for rank, pair in enumerate(data["merges"]):
            tok.merges[(pair[0], pair[1])] = rank
        tok.special = dict(data.get("special", {}))
        tok._trained_on = data.get("trained_on", 0)
        return tok

    # ---------- inspection ----------
    def vocab_size(self) -> int:
        return _BASE + len(self.merges) + len(self.special)

    def vocab(self) -> list[str]:
        """Every token as a printable string (undecodable bytes → ·)."""
        id_to_pair = {_BASE + rank: pair for pair, rank in self.merges.items()}
        out = []
        for i in range(_BASE):
            b = bytes([i])
            out.append(b.decode("ascii") if 32 <= i < 127 else _UNPRINTABLE)
        for t in range(_BASE, _BASE + len(self.merges)):
            bs = bytes(self._expand(t, id_to_pair))
            try:
                out.append(bs.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(_UNPRINTABLE * 2)
        out.extend(self.special.keys())
        return out

    def stats(self, text: str) -> dict:
        chars = len(text)
        raw = len(text.encode("utf-8"))
        toks = len(self.encode(text))
        return {"chars": chars, "bytes": raw, "tokens": toks,
                "chars_per_token": round(chars / toks, 2) if toks else 0.0,
                "bytes_per_token": round(raw / toks, 2) if toks else 0.0,
                "compression_vs_bytes": round(raw / toks, 3) if toks else 0.0}
