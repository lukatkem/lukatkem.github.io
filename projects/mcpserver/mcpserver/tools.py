"""Text-toolkit tools exposed by the MCP server.

Four standalone tools implemented on the Python standard library only:

* ``dedupe_corpus``  — near-duplicate detection (word 5-gram shingles,
  64-permutation MinHash, 16-band LSH, union-find clustering),
* ``scrub_pii``      — redaction of emails, phone-like digit sequences, long
  digit runs, API keys and JSON Web Tokens,
* ``quality_score``  — 0..1 text quality verdict with named reasons,
* ``corpus_stats``   — character/word/line statistics, MD5 fingerprint, top words.

Handlers receive the ``arguments`` object of a ``tools/call`` request and
return the text block the server places into ``content``. Argument problems
raise :class:`ToolArgumentError`, which the protocol layer converts into an
MCP tool error (a result with ``isError: true``), not a JSON-RPC error.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["Tool", "ToolArgumentError", "TOOLS", "TOOL_MAP"]


class ToolArgumentError(ValueError):
    """Raised when tool arguments are missing or malformed.

    The protocol layer turns this into an MCP tool error: a normal result
    with ``isError: true`` and a human-readable message inside ``content``.
    """


@dataclass(frozen=True)
class Tool:
    """A callable MCP tool: name, LLM-visible description, JSON schema, handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``tools/list`` wire shape."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _require_text(arguments: dict[str, Any], tool_name: str) -> str:
    """Fetch and validate the required ``text`` string argument."""
    value = arguments.get("text")
    if value is None:
        raise ToolArgumentError(
            f"{tool_name} requires 'text' (a string); none was provided"
        )
    if not isinstance(value, str):
        raise ToolArgumentError(
            f"{tool_name} argument 'text' must be a string, got {type(value).__name__}"
        )
    return value


_WORD_RE = re.compile(r"\w+")

# ---------------------------------------------------------------------------
# dedupe_corpus — MinHash + LSH near-duplicate detection
# ---------------------------------------------------------------------------

_NGRAM_SIZE = 5                       # word 5-gram shingles
_MINHASH_PERMUTATIONS = 64            # signature length
_LSH_BANDS = 16                       # 16 bands x 4 rows over the 64-row signature
_LSH_ROWS = _MINHASH_PERMUTATIONS // _LSH_BANDS
_DEFAULT_THRESHOLD = 0.8
_MERSENNE_PRIME = (1 << 61) - 1       # modulus for the affine permutations
_HASH_KEY = b"mcpserver/minhash/v1"   # deterministic blake2b seed


def _word_shingles(text: str, n: int = _NGRAM_SIZE) -> set[tuple[str, ...]]:
    """Return the set of word ``n``-gram shingles of *text* (lowercased).

    Texts shorter than ``n`` words collapse to a single shingle (their whole
    word tuple) so tiny documents remain comparable.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _shingle_hash(shingle: tuple[str, ...]) -> int:
    """Deterministic 64-bit blake2b hash of one shingle."""
    digest = hashlib.blake2b(
        " ".join(shingle).encode("utf-8"), digest_size=8, key=_HASH_KEY
    )
    return int.from_bytes(digest.digest(), "big")


def _permutation_parameters() -> list[tuple[int, int]]:
    """Derive the 64 affine MinHash permutation coefficients ``(a, b)``.

    Each signature row uses ``(a * h + b) mod p`` over the shingle hash ``h``.
    The coefficients are fixed constants derived from keyed blake2b, so
    signatures are reproducible across runs and processes.
    """
    parameters = []
    for index in range(_MINHASH_PERMUTATIONS):
        digest = hashlib.blake2b(
            f"perm/{index}".encode("ascii"), digest_size=16, key=_HASH_KEY
        ).digest()
        a = int.from_bytes(digest[:8], "big") % _MERSENNE_PRIME
        b = int.from_bytes(digest[8:], "big") % _MERSENNE_PRIME
        parameters.append((a or 1, b))
    return parameters


_PERMUTATIONS = _permutation_parameters()


def _minhash_signature(shingles: set[tuple[str, ...]]) -> list[int]:
    """MinHash signature: per permutation, the minimum permuted shingle hash."""
    if not shingles:
        return [0] * _MINHASH_PERMUTATIONS
    base_hashes = [_shingle_hash(shingle) for shingle in shingles]
    return [
        min((a * base + b) % _MERSENNE_PRIME for base in base_hashes)
        for a, b in _PERMUTATIONS
    ]


def _estimated_jaccard(first: list[int], second: list[int]) -> float:
    """Estimate Jaccard similarity as the fraction of equal signature rows."""
    matches = sum(1 for a, b in zip(first, second) if a == b)
    return matches / _MINHASH_PERMUTATIONS


def _candidate_pairs(signatures: list[list[int]]) -> set[tuple[int, int]]:
    """LSH candidate generation: pairs sharing at least one 16-band bucket."""
    pairs: set[tuple[int, int]] = set()
    for band in range(_LSH_BANDS):
        start = band * _LSH_ROWS
        buckets: dict[tuple[int, ...], list[int]] = {}
        for doc_id, signature in enumerate(signatures):
            key = tuple(signature[start:start + _LSH_ROWS])
            buckets.setdefault(key, []).append(doc_id)
        for members in buckets.values():
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    pairs.add((left, right))
    return pairs


class _UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        if self._rank[first_root] < self._rank[second_root]:
            first_root, second_root = second_root, first_root
        self._parent[second_root] = first_root
        if self._rank[first_root] == self._rank[second_root]:
            self._rank[first_root] += 1


def _tool_dedupe_corpus(arguments: dict[str, Any]) -> str:
    """Cluster near-duplicate documents; return a human-readable summary."""
    texts = arguments.get("texts")
    if texts is None:
        raise ToolArgumentError(
            "dedupe_corpus requires 'texts' (a list of strings); none was provided"
        )
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ToolArgumentError(
            "dedupe_corpus argument 'texts' must be a list of strings"
        )
    threshold = arguments.get("threshold", _DEFAULT_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ToolArgumentError(
            "dedupe_corpus argument 'threshold' must be a number between 0.0 and 1.0"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ToolArgumentError(
            "dedupe_corpus argument 'threshold' must be between 0.0 and 1.0, "
            f"got {threshold}"
        )
    threshold = float(threshold)

    signatures = [_minhash_signature(_word_shingles(text)) for text in texts]
    forest = _UnionFind(len(texts))
    for first, second in _candidate_pairs(signatures):
        if _estimated_jaccard(signatures[first], signatures[second]) >= threshold:
            forest.union(first, second)

    clusters: dict[int, list[int]] = {}
    for doc_id in range(len(texts)):
        clusters.setdefault(forest.find(doc_id), []).append(doc_id)
    duplicate_clusters = sorted(
        (members for members in clusters.values() if len(members) > 1),
        key=lambda members: members[0],
    )
    removed = sum(len(members) - 1 for members in duplicate_clusters)
    unique = len(texts) - removed

    documents_word = "document" if len(texts) == 1 else "documents"
    cluster_word = "cluster" if len(duplicate_clusters) == 1 else "clusters"
    lines = [
        f"{len(texts)} {documents_word} → {unique} unique "
        f"({len(duplicate_clusters)} duplicate {cluster_word} removed)",
        f"threshold: {threshold:.2f}",
    ]
    for number, members in enumerate(duplicate_clusters, start=1):
        lines.append(
            f"cluster {number}: docs {members} · representative: doc {members[0]}"
        )
    if not duplicate_clusters:
        lines.append("no duplicate clusters detected")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# scrub_pii — PII redaction
# ---------------------------------------------------------------------------

_REDACTED = "[REDACTED]"

# Named alternatives matched in one pass; the order encodes priority — e.g.
# dates are consumed before the phone scanner can ever see them.
_PII_RULES: list[tuple[str, str]] = [
    (
        "email",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    ),
    (
        "api_key",
        r"\b(?:"
        r"sk-or-v1-[A-Za-z0-9_-]{8,}"       # OpenRouter
        r"|sk-[A-Za-z0-9_-]{16,}"           # OpenAI-style
        r"|nvapi-[A-Za-z0-9_-]{8,}"         # NVIDIA
        r"|gh[pousr]_[A-Za-z0-9]{20,}"      # GitHub
        r"|xox[aepbrs]-[A-Za-z0-9-]{10,}"   # Slack
        r")\b",
    ),
    ("jwt", r"\b[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    (
        # Dates are kept verbatim; they must never be redacted as phones.
        "date",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b"
        r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        r"|\b\d{1,2}-\d{1,2}-\d{4}\b",
    ),
    (
        # Digit groups joined by spaces/dots/dashes/parens; classified in code.
        "phone",
        r"\+?\b\d[\d\s.\-()]{7,}\d(?!\w)",
    ),
]

_PII_MASTER = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _PII_RULES)
)
_NON_DIGITS = re.compile(r"\D")
_PHONE_SEPARATORS = re.compile(r"[\s.\-()]")

_PII_LABELS = {
    "email": "emails",
    "api_key": "api_keys",
    "jwt": "jwts",
    "phone": "phones",
    "digit_run": "long_digit_runs",
}
_PII_DISPLAY_ORDER = ("email", "phone", "digit_run", "api_key", "jwt")


def _classify_phone_match(match_text: str) -> str | None:
    """Classify a phone-pattern match: ``'phone'``, ``'digit_run'`` or None.

    Digit groups joined by separators count as a phone at 9+ digits; a bare
    run only counts at 10+ digits (cards, ids). Anything shorter is kept.
    """
    digits = _NON_DIGITS.sub("", match_text)
    if _PHONE_SEPARATORS.search(match_text):
        return "phone" if len(digits) >= 9 else None
    return "digit_run" if len(digits) >= 10 else None


def _tool_scrub_pii(arguments: dict[str, Any]) -> str:
    """Redact PII from ``text``; return the cleaned text plus per-kind counts."""
    text = _require_text(arguments, "scrub_pii")

    counts: Counter[str] = Counter()
    pieces: list[str] = []
    cursor = 0
    for match in _PII_MASTER.finditer(text):
        pieces.append(text[cursor:match.start()])
        kind = match.lastgroup or ""
        matched = match.group()
        if kind == "date":
            pieces.append(matched)  # dates are preserved, never redacted
        elif kind == "phone":
            classification = _classify_phone_match(matched)
            if classification is None:
                pieces.append(matched)  # too few digits to be phone-like
            else:
                counts[classification] += 1
                pieces.append(_REDACTED)
        else:
            counts[kind] += 1
            pieces.append(_REDACTED)
        cursor = match.end()
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)

    summary = ", ".join(
        f"{_PII_LABELS[kind]}={count}"
        for kind in _PII_DISPLAY_ORDER
        if (count := counts.get(kind, 0)) > 0
    )
    return f"redactions: {summary or 'none'}\n\n{cleaned}"


# ---------------------------------------------------------------------------
# quality_score — text quality verdict
# ---------------------------------------------------------------------------

_MIN_CHARS = 200             # hard: fewer characters than this is a reject
_MIN_WORDS = 25              # hard: fewer words than this is a reject
_MAX_SYMBOL_RATIO = 0.15     # hard: non-alphanumeric, non-space character ratio
_MAX_LINE_REPEATS = 3        # hard: any identical line appearing this often
_SOFT_PENALTY = 0.15         # score deducted per soft reason
_MIN_PASS_SCORE = 0.85       # soft-penalized scores below this are WARN
_SHOUTING_CAPS_RATIO = 0.5   # soft: share of uppercase letters
_MAX_MEAN_WORD_LENGTH = 9    # soft: mean word length
_LINK_FARM_LIMIT = 4         # soft: more http(s) links than this is a farm

_URL_RE = re.compile(r"https?://\S+")
_LOREM_RE = re.compile(r"lorem\s+ipsum", re.IGNORECASE)


def _tool_quality_score(arguments: dict[str, Any]) -> str:
    """Score text quality 0..1; return a verdict line plus the reason list."""
    text = _require_text(arguments, "quality_score")
    words = _WORD_RE.findall(text)
    hard: list[tuple[str, str]] = []
    soft: list[tuple[str, str]] = []

    if len(text) < _MIN_CHARS:
        hard.append(
            ("too-short", f"text has {len(text)} characters, minimum is {_MIN_CHARS}")
        )
    if len(words) < _MIN_WORDS:
        hard.append(
            ("too-few-words", f"text has {len(words)} words, minimum is {_MIN_WORDS}")
        )
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    symbol_ratio = symbols / len(text) if text else 1.0
    if symbol_ratio > _MAX_SYMBOL_RATIO:
        hard.append(
            (
                "symbol-ratio",
                f"symbol ratio {symbol_ratio:.0%} exceeds {_MAX_SYMBOL_RATIO:.0%}",
            )
        )
    line_counts = Counter(line.strip() for line in text.splitlines() if line.strip())
    repeated = [line for line, count in line_counts.items() if count >= _MAX_LINE_REPEATS]
    if repeated:
        hard.append(
            (
                "repeated-lines",
                f"a line appears {_MAX_LINE_REPEATS}+ times, e.g. {repeated[0][:60]!r}",
            )
        )

    if hard:
        score, verdict, reasons = 0.0, "FAIL", hard
    else:
        letters = [char for char in text if char.isalpha()]
        caps_ratio = (
            sum(1 for char in letters if char.isupper()) / len(letters)
            if letters
            else 0.0
        )
        if caps_ratio > _SHOUTING_CAPS_RATIO:
            soft.append(("shouting-caps", f"{caps_ratio:.0%} of letters are uppercase"))
        mean_word_length = sum(len(word) for word in words) / len(words)
        if mean_word_length > _MAX_MEAN_WORD_LENGTH:
            soft.append(
                (
                    "mean-word-length",
                    f"mean word length {mean_word_length:.1f} exceeds "
                    f"{_MAX_MEAN_WORD_LENGTH}",
                )
            )
        if _LOREM_RE.search(text):
            soft.append(("lorem-ipsum", "placeholder text detected"))
        links = _URL_RE.findall(text)
        if len(links) > _LINK_FARM_LIMIT:
            soft.append(
                ("link-farm", f"{len(links)} links found, more than {_LINK_FARM_LIMIT}")
            )
        score = round(max(0.0, 1.0 - _SOFT_PENALTY * len(soft)), 2)
        verdict = "PASS" if score >= _MIN_PASS_SCORE else "WARN"
        reasons = soft

    kinds = ", ".join(kind for kind, _ in reasons) if reasons else "none"
    lines = [f"score {score:.2f} · {verdict} · reasons: {kinds}"]
    lines.extend(f"- {kind}: {detail}" for kind, detail in reasons)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# corpus_stats — basic corpus profile
# ---------------------------------------------------------------------------


def _tool_corpus_stats(arguments: dict[str, Any]) -> str:
    """Return character/word/line statistics, an MD5 fingerprint and top words."""
    text = _require_text(arguments, "corpus_stats")

    words = _WORD_RE.findall(text)
    frequencies = Counter(word.lower() for word in words)
    top_words = ", ".join(
        f"{word} ({count})" for word, count in frequencies.most_common(3)
    )
    # MD5 here is a non-cryptographic content fingerprint only.
    md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    lines = [
        f"chars: {len(text)}",
        f"words: {len(words)}",
        f"unique-words: {len(frequencies)}",
        f"lines: {len(text.splitlines())}",
        f"md5: {md5}",
        f"top-words: {top_words if top_words else 'none'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry — the MCP-visible surface
# ---------------------------------------------------------------------------

_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The text to process."}
    },
    "required": ["text"],
}

TOOLS: list[Tool] = [
    Tool(
        name="dedupe_corpus",
        description=(
            "Detect near-duplicate documents in a list of texts. Builds word "
            "5-gram shingles, 64-permutation MinHash signatures (deterministic "
            "blake2b seeding) and 16-band LSH to find candidate pairs, estimates "
            "Jaccard similarity, then clusters duplicates with union-find. "
            "Returns a summary (documents in, unique out, duplicate clusters "
            "removed) and each cluster's member ids with a representative."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The documents to compare.",
                },
                "threshold": {
                    "type": "number",
                    "description": (
                        "Minimum estimated Jaccard similarity (0..1) for two "
                        "documents to count as duplicates."
                    ),
                    "default": 0.8,
                },
            },
            "required": ["texts"],
        },
        handler=_tool_dedupe_corpus,
    ),
    Tool(
        name="scrub_pii",
        description=(
            "Redact personally identifiable information from free text. Replaces "
            "email addresses, phone-like digit sequences (9+ digits; dates like "
            "2026-09-20 are preserved), long bare digit runs (10+, e.g. card or "
            "account numbers), API keys (sk-…, sk-or-v1-…, nvapi-…, ghp_…, xox…) "
            "and JSON Web Tokens with [REDACTED]. Returns the cleaned text plus "
            "a redaction count per kind."
        ),
        input_schema=_TEXT_SCHEMA,
        handler=_tool_scrub_pii,
    ),
    Tool(
        name="quality_score",
        description=(
            "Score text quality from 0 to 1 with named reasons. Hard rejects "
            "(score 0): under 200 characters, under 25 words, symbol ratio above "
            "15%, or any line repeated 3+ times. Soft penalties (-0.15 each): "
            "SHOUTING-style caps, mean word length above 9, lorem ipsum "
            "placeholders, link farms (more than 4 links). Returns a one-line "
            "verdict 'score X · PASS/WARN/FAIL · reasons: …' followed by the "
            "reason list."
        ),
        input_schema=_TEXT_SCHEMA,
        handler=_tool_quality_score,
    ),
    Tool(
        name="corpus_stats",
        description=(
            "Compute basic statistics for a text: character, word, unique-word "
            "and line counts, an MD5 fingerprint (non-cryptographic), and the "
            "three most common words. Useful as a quick corpus profile before "
            "dedupe_corpus or quality_score."
        ),
        input_schema=_TEXT_SCHEMA,
        handler=_tool_corpus_stats,
    ),
]

TOOL_MAP: dict[str, Tool] = {tool.name: tool for tool in TOOLS}
