"""A self-contained walkthrough: train on an embedded corpus, show the funnel."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .bpe import Tokenizer

_CORPUS = (
    "the little robot walked through the garden. the garden was full of lights, "
    "and the robot loved the lights. every night the robot counted the lights, "
    "and every night the lights counted the robot.\n\n"
    "def greet(name):\n    return f'hello {name}'\n\n"
    "the quick brown fox jumps over the lazy dog near the river bank. "
    "the dog did not mind the fox or the river or the bank.\n\n"
    "a wizard looked at the stars and wrote down what he saw. the stars wrote "
    "back. the notebook filled with starlight.\n\n"
    "she sold seashells by the seashore, and the seashore sold shells right back. "
)


def main() -> int:
    print("== tokenforge demo — train, encode, round-trip ==\n")
    for vocab in (300, 400, 512):
        tok = Tokenizer().train(_CORPUS, vocab)
        st = tok.stats(_CORPUS)
        print(f"vocab {vocab:>4} · merges {len(tok.merges):>3} · "
              f"{st['bytes']} bytes → {st['tokens']} tokens "
              f"({st['bytes_per_token']} bytes/token)")
    tok = Tokenizer().train(_CORPUS, 512)
    learned = [v for v in tok.vocab()[256:] if len(v) > 2][:10]
    print("\nexample learned tokens:", learned)
    sentence = "the robot counted the lights 🌟"
    ids = tok.encode(sentence)
    back = tok.decode(ids)
    print(f"\nround-trip: {sentence!r} → {len(ids)} ids → {back!r}")
    print("lossless:", back == sentence)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tf.json"
        tok.save(p)
        again = Tokenizer.load(p)
        print("save/load round-trip:", again.decode(again.encode(sentence)) == sentence)
    print("\ndone — pure stdlib, fully deterministic.")
    return 0
