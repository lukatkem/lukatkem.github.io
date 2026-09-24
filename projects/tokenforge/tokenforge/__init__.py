"""tokenforge — byte-pair encoding trained from scratch.

The piece of infrastructure every LLM uses, in ~300 lines: learn merge ranks
from a corpus, encode any text to token ids, decode back losslessly. Byte-level
base vocabulary, so nothing is ever out of vocabulary.
"""
from .bpe import Tokenizer
from .errors import TokenizerError

__all__ = ["Tokenizer", "TokenizerError"]
__version__ = "1.0.0"
