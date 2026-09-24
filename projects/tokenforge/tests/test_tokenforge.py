"""tokenforge tests — round-trips, determinism, merge order, persistence, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tokenforge import Tokenizer, TokenizerError

PY = sys.executable
CORPUS = ("the quick brown fox jumps over the lazy dog. " * 20
          + "hello world, hello tokenizer. tokens, tokens everywhere. " * 20)


def trained(vocab: int = 300, corpus: str = CORPUS) -> Tokenizer:
    return Tokenizer().train(corpus, vocab)


# ---------- round-trips ----------
def test_round_trip_ascii():
    t = trained()
    for s in ["the quick brown fox", "hello, tokens!", "x"]:
        assert t.decode(t.encode(s)) == s


def test_round_trip_unicode_and_emoji():
    t = trained()
    for s in ["héllo wörld 🌟", "georgian: გამარჯობა", "日本語のテスト"]:
        assert t.decode(t.encode(s)) == s


def test_round_trip_whitespace_heavy():
    t = trained()
    s = "    spaced\n\n\tout\t\ttext\n"
    assert t.decode(t.encode(s)) == s


# ---------- training ----------
def test_training_is_deterministic():
    a, b = trained(), trained()
    assert a.merges == b.merges
    assert a.encode(CORPUS[:200]) == b.encode(CORPUS[:200])


def test_most_frequent_pair_merges_first():
    # "ab" occurs 5 times, "bc" twice → the first learned token must be "ab"
    corpus = "ababababab" + "bcbc"
    t = Tokenizer().train(corpus, 257)
    first_pair = next(iter(t.merges))
    ids = list(corpus.encode())
    first_merged = Tokenizer._merge_ids(ids, first_pair, 256)
    assert t.decode([256]) == "ab"
    assert first_merged != ids  # a merge definitely happened


def test_vocab_size_respected():
    # a rich-enough corpus reaches the exact target…
    assert trained(300).vocab_size() == 300
    # …a small repetitive corpus may exhaust earlier (documented behavior):
    t = trained(400)
    assert 257 <= t.vocab_size() <= 400


def test_train_rejects_tiny_vocab():
    with pytest.raises(TokenizerError):
        trained(100)


# ---------- specials ----------
def test_special_tokens_round_trip_and_stay_whole():
    t = trained()
    sid = t.register_special("<|endoftext|>")
    text = "hello <|endoftext|> world"
    ids = t.encode(text)
    assert sid in ids                      # the special survived as ONE id
    assert t.decode(ids) == text


def test_special_register_is_idempotent():
    t = trained()
    assert t.register_special("<s>") == t.register_special("<s>")


def test_encode_can_ignore_specials():
    t = trained()
    t.register_special("<s>")
    ids = t.encode("a <s> b", allow_special=False)
    assert t.special["<s>"] not in ids


# ---------- persistence ----------
def test_save_load_round_trip(tmp_path: Path):
    t = trained()
    p = tmp_path / "tf.json"
    t.save(p)
    t2 = Tokenizer.load(p)
    for s in ["the quick brown fox", "unknown 🌟 text"]:
        assert t2.encode(s) == t.encode(s)
        assert t2.decode(t2.encode(s)) == s
    data = json.loads(p.read_text())
    assert data["version"] == 1 and data["merges"]


# ---------- compression ----------
def test_bigger_vocab_compresses_more():
    small = trained(280).stats(CORPUS)["tokens"]
    big = trained(400).stats(CORPUS)["tokens"]
    assert big < small  # more merges → fewer tokens on the training corpus


def test_stats_math_on_known_string():
    t = trained()
    s = "abcd"  # 4 chars, 4 bytes
    st = t.stats(s)
    assert st["chars"] == 4 and st["bytes"] == 4 and st["tokens"] >= 1


# ---------- errors ----------
def test_decode_rejects_invalid_ids():
    t = trained()
    with pytest.raises(TokenizerError):
        t.decode([-1])
    with pytest.raises(TokenizerError):
        t.decode(["x"])


# ---------- CLI ----------
def test_cli_train_encode_decode(tmp_path: Path):
    corpus = tmp_path / "c.txt"
    vocab = tmp_path / "tf.json"
    corpus.write_text(CORPUS)
    assert subprocess.run([PY, "-m", "tokenforge", "train", "--corpus", str(corpus),
                           "--vocab", "320", "--out", str(vocab)]).returncode == 0
    enc = subprocess.run([PY, "-m", "tokenforge", "encode", "--vocab", str(vocab),
                          "--text", "the quick brown fox"], capture_output=True, text=True)
    ids = json.loads(enc.stdout)
    dec = subprocess.run([PY, "-m", "tokenforge", "decode", "--vocab", str(vocab),
                          "--ids", ",".join(map(str, ids))], capture_output=True, text=True)
    assert dec.stdout.strip() == "the quick brown fox"
