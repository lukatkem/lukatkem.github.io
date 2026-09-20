"""Hydro-1 — a GPT trained from scratch, nothing downloaded.

Every component lives in this package: char tokenizer, causal self-attention,
MLP blocks, the full transformer, and the sampling loop. The weights in
checkpoints/ were born on the machine that ran train.py — never fetched.
"""
__version__ = "0.1.0"
