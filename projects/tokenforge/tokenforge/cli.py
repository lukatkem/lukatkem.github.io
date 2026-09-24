"""Command-line interface: train, encode, decode, stats, demo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bpe import Tokenizer
from .errors import TokenizerError


def _train(a: argparse.Namespace) -> int:
    text = Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    tok = Tokenizer().train(text, a.vocab)
    tok.save(a.out)
    st = tok.stats(text)
    print(f"trained: vocab {tok.vocab_size()} · merges {len(tok.merges)} · saved → {a.out}")
    print(f"corpus:  {st['bytes']} bytes → {st['tokens']} tokens "
          f"({st['bytes_per_token']} bytes/token)")
    return 0


def _encode(a: argparse.Namespace) -> int:
    tok = Tokenizer.load(a.vocab)
    text = Path(a.text).read_text(encoding="utf-8", errors="replace") if a.text == "-" or Path(a.text).exists() else a.text
    print(json.dumps(tok.encode(text, allow_special=not a.no_special)))
    return 0


def _decode(a: argparse.Namespace) -> int:
    tok = Tokenizer.load(a.vocab)
    ids = [int(x) for x in a.ids.replace(" ", "").split(",") if x]
    print(tok.decode(ids))
    return 0


def _stats(a: argparse.Namespace) -> int:
    tok = Tokenizer.load(a.vocab)
    text = Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    print(json.dumps(tok.stats(text), indent=1))
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tokenforge",
                                 description="byte-pair encoding, trained from scratch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train a tokenizer on a corpus")
    tr.add_argument("--corpus", required=True)
    tr.add_argument("--vocab", type=int, default=500)
    tr.add_argument("--out", default="tf.json")
    tr.set_defaults(func=_train)

    en = sub.add_parser("encode", help="encode text (or a file path with --file)")
    en.add_argument("--vocab", required=True)
    en.add_argument("--text", required=True)
    en.add_argument("--no-special", action="store_true")
    en.set_defaults(func=_encode)

    de = sub.add_parser("decode", help="decode comma-separated ids")
    de.add_argument("--vocab", required=True)
    de.add_argument("--ids", required=True)
    de.set_defaults(func=_decode)

    st = sub.add_parser("stats", help="compression stats of a corpus")
    st.add_argument("--corpus", required=True)
    st.add_argument("--vocab", required=True)
    st.set_defaults(func=_stats)

    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] == "demo":
        from . import demo
        return demo.main()

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except TokenizerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
